#!/usr/bin/env python3
"""Bridge action-direction alignment using factorized direction-code-only latents."""

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = Path(__file__).resolve().parents[1]
LATENT_ROOT = Path(__file__).resolve().parent
for path in (str(REPO_ROOT), str(LATENT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from eval_bridge_offset_geometry import _future_zq, _masked_action_sum  # noqa: E402
from eval_ksg_mi import _load_model  # noqa: E402
from eval_offset_norms import _collect_offset_batches, _to_device  # noqa: E402


def _parse_model(spec: str) -> tuple[str, str, str, str, str]:
    parts = spec.split("|")
    if len(parts) != 5:
        raise ValueError("model spec must be name|kind|config|ckpt|feature_view")
    return parts[0], parts[1], parts[2], parts[3], parts[4]


def _future_direction_indices(outputs: Dict, batch_size: int) -> torch.Tensor:
    if "direction_indices" not in outputs:
        raise ValueError("direction_only feature requires outputs['direction_indices'].")
    radius = outputs["radius_indices"].detach().reshape(-1).long()
    direction = outputs["direction_indices"].detach().reshape(radius.numel(), -1).long()
    return direction[-batch_size:]


def _direction_code_feature(model: torch.nn.Module, outputs: Dict, batch_size: int) -> torch.Tensor:
    direction_indices = _future_direction_indices(outputs, batch_size)
    directions = model.lam.vq.direction_values().detach().float()
    z = directions[direction_indices]
    return z.reshape(batch_size, -1)


def _feature(model: torch.nn.Module, outputs: Dict, batch_size: int, view: str) -> torch.Tensor:
    if view == "direction_code":
        return _direction_code_feature(model, outputs, batch_size)
    if view == "zq":
        return _future_zq(outputs, batch_size)
    raise ValueError(f"unknown feature view: {view}")


def _encode_batches(
    model: torch.nn.Module,
    batches: list[Dict],
    device: torch.device,
    view: str,
) -> tuple[np.ndarray, np.ndarray]:
    zs, actions = [], []
    with torch.no_grad():
        for idx, batch_cpu in enumerate(batches, start=1):
            batch = _to_device(batch_cpu, device)
            batch_size = int(batch["videos"].shape[0])
            outputs = model.lam.vq_encode(batch["videos"])
            zs.append(_feature(model, outputs, batch_size, view).cpu().numpy())
            actions.append(_masked_action_sum(batch, batch_size).cpu().numpy())
            print(f"encode view={view} batch={idx}/{len(batches)} samples={sum(x.shape[0] for x in zs)}", flush=True)
    return np.concatenate(zs, axis=0), np.concatenate(actions, axis=0)


def _unit(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def _upper_tri(mat: np.ndarray) -> np.ndarray:
    i, j = np.triu_indices(mat.shape[0], k=1)
    return mat[i, j]


def _cosine_metrics(z: np.ndarray, action: np.ndarray, top_k: int, action_min_percentile: float) -> dict:
    action_norm = np.linalg.norm(action, axis=1)
    if action_min_percentile > 0:
        keep = action_norm > np.percentile(action_norm, action_min_percentile)
        z = z[keep]
        action = action[keep]
        action_norm = action_norm[keep]

    z_unit = _unit(z.astype(np.float64, copy=False))
    action_unit = _unit(action.astype(np.float64, copy=False))
    latent_cos = z_unit @ z_unit.T
    action_cos = action_unit @ action_unit.T

    latent_pairs = _upper_tri(latent_cos)
    action_pairs = _upper_tri(action_cos)
    spearman = spearmanr(latent_pairs, action_pairs).correlation
    pearson = pearsonr(latent_pairs, action_pairs)[0]

    n = z.shape[0]
    k = min(top_k, n - 1)
    np.fill_diagonal(latent_cos, -np.inf)
    top = np.argpartition(-latent_cos, kth=k - 1, axis=1)[:, :k]
    top_action_cos = action_cos[np.arange(n)[:, None], top].mean(axis=1)

    rng = np.random.default_rng(123)
    random_scores = []
    for i in range(n):
        candidates = np.delete(np.arange(n), i)
        random_scores.append(action_cos[i, rng.choice(candidates, size=k, replace=False)].mean())
    random_action_cos = float(np.mean(random_scores))

    oracle = np.array(action_cos, copy=True)
    np.fill_diagonal(oracle, -np.inf)
    oracle_top = np.argpartition(-oracle, kth=k - 1, axis=1)[:, :k]
    oracle_action_cos = float(action_cos[np.arange(n)[:, None], oracle_top].mean())

    return {
        "samples_used": int(n),
        "action_norm_mean": float(action_norm.mean()),
        "pairwise_cosine_spearman": float(spearman) if spearman == spearman else float("nan"),
        "pairwise_cosine_pearson": float(pearson) if pearson == pearson else float("nan"),
        "topk": int(k),
        "topk_action_cosine": float(top_action_cos.mean()),
        "random_topk_action_cosine": random_action_cos,
        "topk_action_cosine_lift": float(top_action_cos.mean() - random_action_cos),
        "oracle_topk_action_cosine": oracle_action_cos,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, help="name|kind|config|ckpt|feature_view")
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--offsets", default="1,2,3,4,5,6,7,8,9")
    parser.add_argument("--samples-per-offset", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--action-min-percentiles", default="0,10")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    offsets = [int(x) for x in args.offsets.split(",") if x.strip()]
    percentiles = [float(x) for x in args.action_min_percentiles.split(",") if x.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} offsets={offsets}", flush=True)

    cached = _collect_offset_batches(
        args.data_config,
        offsets=offsets,
        samples_per_offset=args.samples_per_offset,
        batch_size=args.batch_size,
    )
    batches_by_offset: dict[int, list[Dict]] = defaultdict(list)
    for offset, batch in cached:
        batches_by_offset[int(offset)].append(batch)

    rows = []
    for spec in args.model:
        model_name, kind, config, ckpt, view = _parse_model(spec)
        print(f"[model] {model_name} view={view}", flush=True)
        model = _load_model(kind, config, ckpt, device)
        for offset in offsets:
            z, action = _encode_batches(model, batches_by_offset[offset], device, view)
            for percentile in percentiles:
                rows.append(
                    {
                        "dataset": "bridge",
                        "model": model_name,
                        "kind": kind,
                        "feature_view": view,
                        "offset_k": int(offset),
                        "samples": int(z.shape[0]),
                        "latent_dim": int(z.shape[1]),
                        "action_dim": int(action.shape[1]),
                        "action_min_percentile": percentile,
                        **_cosine_metrics(z, action, args.top_k, percentile),
                    }
                )
                print(rows[-1], flush=True)
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
