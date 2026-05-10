#!/usr/bin/env python3
"""Evaluate whether action-near LeRobot windows are latent-near for LAMs.

For each cache/dataset and fixed offset k, this script compares:
  1. Spearman corr between pairwise action distances and latent distances.
  2. Action error of latent nearest neighbors vs random neighbors.
  3. Triplet AUC: action-close positives should be latent-closer than action-far negatives.
"""

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from torchvision import transforms

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = Path(__file__).resolve().parents[1]
LATENT_ROOT = Path(__file__).resolve().parent
for path in (str(REPO_ROOT), str(LATENT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from eval_ksg_mi import _load_model  # noqa: E402


def _parse_dataset(spec: str) -> tuple[str, Path]:
    parts = spec.split("|")
    if len(parts) != 2:
        raise ValueError("dataset spec must be name|cache_dir")
    return parts[0], Path(parts[1])


def _parse_model(spec: str) -> tuple[str, str, str, str]:
    parts = spec.split("|")
    if len(parts) != 4:
        raise ValueError("model spec must be name|kind|config|ckpt")
    return parts[0], parts[1], parts[2], parts[3]


def _load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _load_records(cache_dir: Path) -> List[dict]:
    records = []
    with open(cache_dir / "windows.jsonl", "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _action_stats(cache_dir: Path, dataset_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stats = _load_json(cache_dir / "dataset_statistics.json")[dataset_name]["action"]
    low = np.asarray(stats["q01"], dtype=np.float32)
    high = np.asarray(stats["q99"], dtype=np.float32)
    zero_mask = np.asarray(stats["min"], dtype=np.float32) == np.asarray(stats["max"], dtype=np.float32)
    return low, high, zero_mask


def _normalize_actions(actions: np.ndarray, low: np.ndarray, high: np.ndarray, zero_mask: np.ndarray) -> np.ndarray:
    out = 2.0 * (actions - low) / (high - low + 1e-8) - 1.0
    out = np.clip(out, -1.0, 1.0)
    if np.any(zero_mask):
        out[:, zero_mask] = 0.0
    return out.astype(np.float32, copy=False)


def _sample_records(records: List[dict], n: int, seed: int, k: int) -> List[dict]:
    eligible = [record for record in records if len(record["frame_paths"]) > k and len(record["actions"]) > k]
    rng = random.Random(seed + 9973 * k)
    if len(eligible) <= n:
        out = list(eligible)
        rng.shuffle(out)
        return out
    return rng.sample(eligible, n)


def _zq_tokens(outputs: Dict, batch_size: int) -> torch.Tensor:
    z = outputs["z_q"].detach().float()
    if z.shape[0] != batch_size:
        z = z.reshape(batch_size, -1, *z.shape[1:])
    return z.reshape(batch_size, -1, z.shape[-1]).reshape(batch_size, -1)


def _encode_k(
    model: torch.nn.Module,
    cache_dir: Path,
    records: List[dict],
    k: int,
    low: np.ndarray,
    high: np.ndarray,
    zero_mask: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    to_tensor = transforms.ToTensor()
    z_batches, action_batches = [], []
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch_records = records[start : start + batch_size]
            videos, actions = [], []
            for record in batch_records:
                frame0 = Image.open(cache_dir / record["frame_paths"][0]).convert("RGB")
                framek = Image.open(cache_dir / record["frame_paths"][k]).convert("RGB")
                videos.append(torch.stack([to_tensor(frame0), to_tensor(framek)], dim=0))
                action_seq = np.asarray(record["actions"], dtype=np.float32)
                action_norm = _normalize_actions(action_seq[:k], low, high, zero_mask).sum(axis=0)
                actions.append(action_norm)
            video = torch.stack(videos, dim=0).to(device, non_blocking=True)
            outputs = model.lam.vq_encode(video)
            z_batches.append(_zq_tokens(outputs, len(batch_records)).cpu().numpy())
            action_batches.append(np.stack(actions).astype(np.float32))
    return np.concatenate(z_batches, axis=0), np.concatenate(action_batches, axis=0)


def _zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + eps)


def _latent_variant(z: np.ndarray, variant: str) -> np.ndarray:
    z = z.astype(np.float64, copy=False)
    if variant == "raw":
        return z
    if variant == "zscore":
        return _zscore(z)
    raise ValueError(variant)


def _nn_action_ratio(z: np.ndarray, a: np.ndarray, seed: int) -> float:
    tree = cKDTree(z)
    _, nn = tree.query(z, k=2, workers=-1)
    nn = nn[:, 1]
    latent_nn_action = np.linalg.norm(a - a[nn], axis=1).mean()
    rng = np.random.default_rng(seed)
    rand = rng.permutation(a.shape[0])
    same = rand == np.arange(a.shape[0])
    if np.any(same):
        rand[same] = (rand[same] + 1) % a.shape[0]
    random_action = np.linalg.norm(a - a[rand], axis=1).mean()
    return float(latent_nn_action / max(random_action, 1e-12))


def _triplet_auc(z: np.ndarray, a: np.ndarray, seed: int, triplets_per_query: int) -> float:
    rng = np.random.default_rng(seed)
    dz = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=-1)
    da = np.linalg.norm(a[:, None, :] - a[None, :, :], axis=-1)
    n = z.shape[0]
    pos_count = max(1, int(0.05 * (n - 1)))
    neg_count = max(1, int(0.50 * (n - 1)))
    correct = 0
    total = 0
    for i in range(n):
        order = np.argsort(da[i])
        order = order[order != i]
        pos_pool = order[:pos_count]
        neg_pool = order[-neg_count:]
        for _ in range(triplets_per_query):
            pos = int(rng.choice(pos_pool))
            neg = int(rng.choice(neg_pool))
            correct += int(dz[i, pos] < dz[i, neg])
            total += 1
    return float(correct / max(total, 1))


def _metrics(z: np.ndarray, action: np.ndarray, variant: str, seed: int, triplets_per_query: int) -> dict:
    zz = _latent_variant(z, variant)
    aa = _zscore(action.astype(np.float64, copy=False))
    dz = pdist(zz, metric="euclidean")
    da = pdist(aa, metric="euclidean")
    corr = spearmanr(da, dz).correlation
    return {
        "latent_preprocess": variant,
        "distance_spearman": float(corr) if corr == corr else float("nan"),
        "latent_nn_action_ratio": _nn_action_ratio(zz, aa, seed),
        "triplet_auc": _triplet_auc(zz, aa, seed, triplets_per_query),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", required=True, help="name|cache_dir")
    parser.add_argument("--model", action="append", required=True, help="name|kind|config|ckpt")
    parser.add_argument("--offsets", default="1,2,3,4,5,6,7,8,9")
    parser.add_argument("--samples-per-offset", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latent-preprocess", default="raw,zscore")
    parser.add_argument("--triplets-per-query", type=int, default=8)
    parser.add_argument("--feature-cache-dir", default="outputs/analysis/cache/lerobot_offset_geometry")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    offsets = [int(x) for x in args.offsets.split(",") if x]
    variants = [x for x in args.latent_preprocess.split(",") if x]
    dataset_specs = [_parse_dataset(spec) for spec in args.dataset]
    model_specs = [_parse_model(spec) for spec in args.model]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    for model_name, kind, config, ckpt in model_specs:
        print(f"[model] loading {model_name}", flush=True)
        model = _load_model(kind, config, ckpt, device)
        for dataset_name, cache_dir in dataset_specs:
            records_all = _load_records(cache_dir)
            low, high, zero_mask = _action_stats(cache_dir, dataset_name)
            for k in offsets:
                cache_path = Path(args.feature_cache_dir) / dataset_name / f"{model_name}_k{k}_n{args.samples_per_offset}.npz"
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                if cache_path.exists():
                    payload = np.load(cache_path)
                    z = payload["z"]
                    action = payload["action"]
                    n = int(z.shape[0])
                    print(f"[cache] {cache_path}", flush=True)
                else:
                    sampled = _sample_records(records_all, args.samples_per_offset, args.seed, k)
                    z, action = _encode_k(model, cache_dir, sampled, k, low, high, zero_mask, args.batch_size, device)
                    n = int(z.shape[0])
                    np.savez_compressed(cache_path, z=z, action=action)
                    print(f"[write] {cache_path} n={n}", flush=True)
                for variant in variants:
                    row = {
                        "dataset": dataset_name,
                        "model": model_name,
                        "kind": kind,
                        "offset_k": int(k),
                        "samples": int(n),
                        "latent_dim": int(z.shape[1]),
                        "action_dim": int(action.shape[1]),
                        **_metrics(z, action, variant, args.seed + k, args.triplets_per_query),
                    }
                    rows.append(row)
                    print(row, flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
