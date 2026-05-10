import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree
from scipy.special import digamma
from scipy.stats import spearmanr


os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = Path(__file__).resolve().parents[1]
LATENT_ROOT = Path(__file__).resolve().parent
for path in (str(REPO_ROOT), str(LATENT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    import wandb

    if not hasattr(wandb, "init"):
        wandb.init = lambda *args, **kwargs: None
except Exception:
    pass

from eval_ksg_mi import _load_model  # noqa: E402
from eval_offset_norms import _to_device  # noqa: E402


def _load_cache(path: str) -> list[Dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    batches = payload["batches"] if isinstance(payload, dict) and "batches" in payload else payload
    if batches and isinstance(batches[0], tuple):
        batches = [batch for _, batch in batches]
    print(f"loaded cache={path} batches={len(batches)}", flush=True)
    return batches


def _parse_model(spec: str) -> tuple[str, str, str, str]:
    parts = spec.split("|")
    if len(parts) != 4:
        raise ValueError("model spec: name|kind|config|ckpt")
    return parts[0], parts[1], parts[2], parts[3]


def _zq_tokens(outputs: Dict, batch_size: int) -> torch.Tensor:
    z = outputs["z_q"].detach().float()
    if z.shape[0] != batch_size:
        z = z.reshape(batch_size, -1, *z.shape[1:])
    return z.reshape(batch_size, -1, z.shape[-1])


def _extract_or_load(
    name: str,
    kind: str,
    config: str,
    ckpt: str,
    batches: list[Dict],
    device: torch.device,
    feature_cache_dir: Path,
) -> dict[str, np.ndarray]:
    path = feature_cache_dir / f"{name}.npz"
    if path.exists():
        data = np.load(path)
        print(f"loaded features={path}", flush=True)
        return {key: data[key] for key in data.files}

    model = _load_model(kind, config, ckpt, device)
    z_raw, z_token_unit, z_token_norms, actions = [], [], [], []
    with torch.no_grad():
        for idx, batch_cpu in enumerate(batches, start=1):
            batch = _to_device(batch_cpu, device)
            outputs = model.lam.vq_encode(batch["videos"])
            batch_size = int(batch["videos"].shape[0])
            tokens = _zq_tokens(outputs, batch_size)
            norms = torch.linalg.vector_norm(tokens, dim=-1, keepdim=True).clamp_min(1e-8)
            z_raw.append(tokens.reshape(batch_size, -1).cpu().numpy())
            z_token_unit.append((tokens / norms).reshape(batch_size, -1).cpu().numpy())
            z_token_norms.append(norms.squeeze(-1).cpu().numpy())
            actions.append(batch["radprog_action_sequence"].detach().float().reshape(batch_size, -1).cpu().numpy())
            print(f"extract name={name} batch={idx}/{len(batches)}", flush=True)

    out = {
        "z_raw": np.concatenate(z_raw, axis=0).astype(np.float64),
        "z_token_unit": np.concatenate(z_token_unit, axis=0).astype(np.float64),
        "z_token_norms": np.concatenate(z_token_norms, axis=0).astype(np.float64),
        "action_seq": np.concatenate(actions, axis=0).astype(np.float64),
    }
    feature_cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    print(f"wrote features={path}", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def _global_scale(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (x - float(x.mean())) / (float(x.std()) + eps)


def _zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + eps)


def _features(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    z_raw = data["z_raw"]
    z_unit = data["z_token_unit"]
    z_norms = data["z_token_norms"]
    return {
        "raw": z_raw,
        "raw_global": _global_scale(z_raw),
        "sample_unit": z_raw / np.maximum(np.linalg.norm(z_raw, axis=1, keepdims=True), 1e-8),
        "token_unit": z_unit,
        "token_norms": z_norms,
        "unit_plus_norms": np.concatenate([z_unit, _global_scale(z_norms)], axis=1),
        "zscore": _zscore(z_raw),
    }


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    sse = float(np.sum((y - pred) ** 2))
    sst = float(np.sum((y - y.mean(axis=0, keepdims=True)) ** 2))
    return 1.0 - sse / max(sst, 1e-12)


def _cosine(y: np.ndarray, pred: np.ndarray) -> float:
    denom = np.linalg.norm(y, axis=1) * np.linalg.norm(pred, axis=1)
    ok = denom > 1e-12
    return float(np.mean(np.sum(y[ok] * pred[ok], axis=1) / denom[ok])) if ok.any() else float("nan")


def _knn_regression(x: np.ndarray, y: np.ndarray, seed: int, k: int, metric: str) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(x.shape[0])
    split = int(0.8 * x.shape[0])
    tr, te = idx[:split], idx[split:]
    xtr, xte, ytr, yte = x[tr], x[te], y[tr], y[te]
    if metric == "cosine":
        xtr = xtr / np.maximum(np.linalg.norm(xtr, axis=1, keepdims=True), 1e-8)
        xte = xte / np.maximum(np.linalg.norm(xte, axis=1, keepdims=True), 1e-8)
        dist = 1.0 - xte @ xtr.T
    else:
        aa = np.sum(xte * xte, axis=1, keepdims=True)
        bb = np.sum(xtr * xtr, axis=1, keepdims=True).T
        dist = aa + bb - 2.0 * (xte @ xtr.T)
    nn = np.argpartition(dist, kth=min(k, dist.shape[1] - 1), axis=1)[:, :k]
    pred = ytr[nn].mean(axis=1)
    return {
        "knn_k": int(k),
        "knn_metric": metric,
        "knn_r2": _r2(yte, pred),
        "knn_cosine": _cosine(yte, pred),
        "knn_mse": float(np.mean((yte - pred) ** 2)),
    }


def _distance_alignment(x: np.ndarray, y: np.ndarray, sample: int, seed: int, metric: str) -> float:
    rng = np.random.default_rng(seed)
    n = min(sample, x.shape[0])
    idx = rng.choice(x.shape[0], size=n, replace=False)
    xx = x[idx]
    yy = y[idx]
    if metric == "cosine":
        xx = xx / np.maximum(np.linalg.norm(xx, axis=1, keepdims=True), 1e-8)
        dx = pdist(xx, metric="cosine")
    else:
        dx = pdist(xx, metric="euclidean")
    dy = pdist(yy, metric="euclidean")
    corr = spearmanr(dx, dy).correlation
    return float(corr)


def _random_project(x: np.ndarray, dim: int, seed: int) -> np.ndarray:
    if x.shape[1] <= dim:
        return x
    rng = np.random.default_rng(seed)
    proj = rng.normal(size=(x.shape[1], dim)).astype(np.float64) / math.sqrt(float(dim))
    return x @ proj


def _ksg_bits(x: np.ndarray, y: np.ndarray, k: int) -> float:
    n = x.shape[0]
    xy = np.concatenate([x, y], axis=1)
    tree_xy = cKDTree(xy)
    tree_x = cKDTree(x)
    tree_y = cKDTree(y)
    dist, _ = tree_xy.query(xy, k=k + 1, p=np.inf, workers=-1)
    eps = np.nextafter(dist[:, k], 0.0)
    nx = np.empty(n, dtype=np.int64)
    ny = np.empty(n, dtype=np.int64)
    for i in range(n):
        nx[i] = len(tree_x.query_ball_point(x[i], eps[i], p=np.inf)) - 1
        ny[i] = len(tree_y.query_ball_point(y[i], eps[i], p=np.inf)) - 1
    mi = digamma(k) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1))
    return float(mi / math.log(2.0))


def _ksg_eval(x: np.ndarray, y: np.ndarray, samples: int, dim: int, seed: int, preprocess: str) -> float:
    rng = np.random.default_rng(seed)
    n = min(samples, x.shape[0])
    idx = rng.choice(x.shape[0], size=n, replace=False)
    xx = x[idx]
    yy = y[idx]
    xx = _random_project(xx, dim, seed)
    yy = _random_project(yy, min(dim, yy.shape[1]), seed + 1)
    if preprocess == "zscore":
        xx = _zscore(xx)
        yy = _zscore(yy)
    elif preprocess == "global":
        xx = _global_scale(xx)
        yy = _global_scale(yy)
    return _ksg_bits(xx, yy, k=5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-cache", required=True)
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distance-sample", type=int, default=1024)
    parser.add_argument("--ksg-samples", type=int, default=1024)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batches = _load_cache(args.batch_cache)
    rows = []
    for spec in args.model:
        name, kind, config, ckpt = _parse_model(spec)
        data = _extract_or_load(name, kind, config, ckpt, batches, device, Path(args.feature_cache_dir))
        y = data["action_seq"]
        y_norm = np.linalg.norm(y, axis=1)
        for feat_name, x in _features(data).items():
            x_norm = np.linalg.norm(x, axis=1)
            base = {
                "name": name,
                "kind": kind,
                "feature": feat_name,
                "samples": int(x.shape[0]),
                "feature_dim": int(x.shape[1]),
                "target": "full_action_sequence",
                "z_norm_action_norm_pearson": float(np.corrcoef(x_norm, y_norm)[0, 1])
                if x_norm.std() > 1e-12 and y_norm.std() > 1e-12
                else float("nan"),
            }
            for metric in ("euclidean", "cosine"):
                row = dict(base, metric_type="distance_alignment", distance_metric=metric)
                row["distance_spearman"] = _distance_alignment(x, y, args.distance_sample, args.seed, metric)
                rows.append(row)
                print(row, flush=True)
            for metric in ("euclidean", "cosine"):
                for k in (1, 5, 20):
                    row = dict(base, metric_type="knn_regression")
                    row.update(_knn_regression(x, y, args.seed, k, metric))
                    rows.append(row)
                    print(row, flush=True)
            for preprocess in ("raw", "global", "zscore"):
                for dim in (16, 32, 64):
                    row = dict(base, metric_type="ksg_mi", ksg_preprocess=preprocess, ksg_dim=dim)
                    row["ksg_mi_bits"] = _ksg_eval(x, y, args.ksg_samples, dim, args.seed, preprocess)
                    rows.append(row)
                    print(row, flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row.keys()}))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
