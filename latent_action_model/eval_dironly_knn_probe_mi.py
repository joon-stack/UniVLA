#!/usr/bin/env python3
"""Compare direction-only hyper LAM features against baseline z_q features.

This is intentionally narrow:
- fixed Bridge offset k via _collect_offset_batches
- hyper factorized view can use direction code vectors only
- baseline views use future z_q
- reports KNN regression, linear ridge probe, and KSG MI
"""

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.special import digamma

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = Path(__file__).resolve().parents[1]
LATENT_ROOT = Path(__file__).resolve().parent
for path in (str(REPO_ROOT), str(LATENT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from eval_bridge_offset_geometry import _future_zq  # noqa: E402
from eval_ksg_mi import _load_model  # noqa: E402
from eval_offset_norms import _collect_offset_batches, _to_device  # noqa: E402


def _parse_model(spec: str) -> tuple[str, str, str, str, str]:
    parts = spec.split("|")
    if len(parts) != 5:
        raise ValueError("model spec must be name|kind|config|ckpt|feature_view")
    return parts[0], parts[1], parts[2], parts[3], parts[4]


def _future_direction_code(model: torch.nn.Module, outputs: Dict, batch_size: int) -> torch.Tensor:
    if "direction_indices" not in outputs or "radius_indices" not in outputs:
        raise ValueError("direction_code view requires factorized outputs with direction_indices.")
    radius = outputs["radius_indices"].detach().reshape(-1).long()
    direction = outputs["direction_indices"].detach().reshape(radius.numel(), -1).long()
    direction = direction[-batch_size:]
    directions = model.lam.vq.direction_values().detach().float()
    return directions[direction].reshape(batch_size, -1)


def _future_radius_direction_code(model: torch.nn.Module, outputs: Dict, batch_size: int) -> torch.Tensor:
    if "direction_indices" not in outputs or "radius_indices" not in outputs:
        raise ValueError("radius_direction_code view requires factorized outputs.")
    radius = outputs["radius_indices"].detach().reshape(-1).long()
    direction = outputs["direction_indices"].detach().reshape(radius.numel(), -1).long()
    radius = radius[-batch_size:]
    direction = direction[-batch_size:]
    radii = model.lam.vq.radius_values().detach().float()[radius].reshape(batch_size, 1)
    directions = model.lam.vq.direction_values().detach().float()
    direction_code = directions[direction].reshape(batch_size, -1)
    return torch.cat([radii, direction_code], dim=1)


def _feature(model: torch.nn.Module, outputs: Dict, batch_size: int, view: str) -> torch.Tensor:
    if view == "direction_code":
        return _future_direction_code(model, outputs, batch_size)
    if view == "radius_direction_code":
        return _future_radius_direction_code(model, outputs, batch_size)
    if view == "zq":
        return _future_zq(outputs, batch_size)
    raise ValueError(f"unknown feature_view={view!r}")


def _masked_action_seq(batch: Dict, batch_size: int) -> torch.Tensor:
    if "radprog_action_sequence" not in batch:
        return batch["action"].detach().float().reshape(batch_size, -1, batch["action"].shape[-1])
    seq = batch["radprog_action_sequence"].detach().float().reshape(batch_size, -1, batch["radprog_action_sequence"].shape[-1])
    mask = batch.get("radprog_action_sequence_mask")
    if mask is None:
        return seq
    mask = mask.detach().float().reshape(batch_size, -1)
    steps = min(seq.shape[1], mask.shape[1])
    return seq[:, :steps] * mask[:, :steps, None]


def _extract_or_load(
    name: str,
    kind: str,
    config: str,
    ckpt: str,
    view: str,
    batches: list[tuple[int, Dict]],
    device: torch.device,
    feature_cache_dir: Path,
) -> dict[str, np.ndarray]:
    path = feature_cache_dir / f"{name}_{view}.npz"
    if path.exists():
        data = np.load(path)
        print(f"loaded features={path}", flush=True)
        return {key: data[key] for key in data.files}

    model = _load_model(kind, config, ckpt, device)
    xs, seqs = [], []
    with torch.no_grad():
        for idx, (_, batch_cpu) in enumerate(batches, start=1):
            batch = _to_device(batch_cpu, device)
            batch_size = int(batch["videos"].shape[0])
            outputs = model.lam.vq_encode(batch["videos"])
            xs.append(_feature(model, outputs, batch_size, view).detach().cpu().numpy())
            seqs.append(_masked_action_seq(batch, batch_size).detach().cpu().numpy())
            print(f"extract name={name} view={view} batch={idx}/{len(batches)}", flush=True)

    seq = np.concatenate(seqs, axis=0).astype(np.float64)
    out = {
        "x": np.concatenate(xs, axis=0).astype(np.float64),
        "action_seq": seq.reshape(seq.shape[0], -1),
        "action_sum": seq.sum(axis=1),
    }
    feature_cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    print(f"wrote features={path}", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def _split(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    a = int(0.70 * n)
    b = int(0.85 * n)
    return idx[:a], idx[a:b], idx[b:]


def _zscore_fit(x: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    return x.mean(axis=0, keepdims=True), x.std(axis=0, keepdims=True) + eps


def _zscore_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def _zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean, std = _zscore_fit(x, eps)
    return _zscore_apply(x, mean, std)


def _global_scale(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (x - float(x.mean())) / (float(x.std()) + eps)


def _unit(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def _feature_variants(x: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "raw": x,
        "global": _global_scale(x),
        "zscore": _zscore(x),
        "sample_unit": _unit(x),
    }


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    sse = float(np.sum((y - pred) ** 2))
    sst = float(np.sum((y - y.mean(axis=0, keepdims=True)) ** 2))
    return 1.0 - sse / max(sst, 1e-12)


def _mse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean((y - pred) ** 2))


def _cosine(y: np.ndarray, pred: np.ndarray) -> float:
    denom = np.linalg.norm(y, axis=1) * np.linalg.norm(pred, axis=1)
    ok = denom > 1e-12
    if int(ok.sum()) == 0:
        return float("nan")
    return float(np.mean(np.sum(y[ok] * pred[ok], axis=1) / denom[ok]))


def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    xtx = x_aug.T @ x_aug
    reg = np.eye(xtx.shape[0], dtype=x.dtype) * alpha
    reg[-1, -1] = 0.0
    rhs = x_aug.T @ y
    try:
        return np.linalg.solve(xtx + reg, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(xtx + reg) @ rhs


def _ridge_predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    return x_aug @ w


def _ridge_eval(x: np.ndarray, y: np.ndarray, split: tuple[np.ndarray, np.ndarray, np.ndarray]) -> dict:
    tr, va, te = split
    mean, std = _zscore_fit(x[tr])
    xtr = _zscore_apply(x[tr], mean, std)
    xva = _zscore_apply(x[va], mean, std)
    xte = _zscore_apply(x[te], mean, std)
    best = None
    for alpha in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0):
        w = _ridge_fit(xtr, y[tr], alpha)
        pred = _ridge_predict(xva, w)
        val_mse = _mse(y[va], pred)
        if best is None or val_mse < best[0]:
            best = (val_mse, alpha, w)
    assert best is not None
    pred = _ridge_predict(xte, best[2])
    return {
        "metric": "ridge_probe",
        "k": "",
        "distance": "",
        "alpha": float(best[1]),
        "test_r2": _r2(y[te], pred),
        "test_mse": _mse(y[te], pred),
        "test_cosine": _cosine(y[te], pred),
    }


def _knn_eval(
    x: np.ndarray,
    y: np.ndarray,
    split: tuple[np.ndarray, np.ndarray, np.ndarray],
    k: int,
    distance: str,
) -> dict:
    tr, _, te = split
    xtr, xte = x[tr], x[te]
    ytr, yte = y[tr], y[te]
    if distance == "cosine":
        xtr = _unit(xtr)
        xte = _unit(xte)
        dist = 1.0 - xte @ xtr.T
    elif distance == "euclidean":
        aa = np.sum(xte * xte, axis=1, keepdims=True)
        bb = np.sum(xtr * xtr, axis=1, keepdims=True).T
        dist = aa + bb - 2.0 * (xte @ xtr.T)
    else:
        raise ValueError(distance)
    nn = np.argpartition(dist, kth=min(k, dist.shape[1] - 1), axis=1)[:, :k]
    pred = ytr[nn].mean(axis=1)
    return {
        "metric": "knn_regression",
        "k": int(k),
        "distance": distance,
        "alpha": "",
        "test_r2": _r2(yte, pred),
        "test_mse": _mse(yte, pred),
        "test_cosine": _cosine(yte, pred),
    }


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


def _ksg_eval(
    x: np.ndarray,
    y: np.ndarray,
    samples: int,
    dim: int,
    seed: int,
    preprocess: str,
    ksg_k: int,
) -> float:
    rng = np.random.default_rng(seed)
    n = min(samples, x.shape[0])
    idx = rng.choice(x.shape[0], size=n, replace=False)
    xx = _random_project(x[idx], dim, seed)
    yy = y[idx]
    if preprocess == "zscore":
        xx = _zscore(xx)
        yy = _zscore(yy)
    elif preprocess == "global":
        xx = _global_scale(xx)
        yy = _global_scale(yy)
    elif preprocess != "raw":
        raise ValueError(preprocess)
    return _ksg_bits(xx, yy, k=ksg_k)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, help="name|kind|config|ckpt|feature_view")
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--offset", type=int, default=9)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ksg-samples", type=int, default=1024)
    parser.add_argument("--ksg-k", type=int, default=5)
    parser.add_argument("--features", default="raw,zscore")
    parser.add_argument("--targets", default="action_sum,full_seq")
    parser.add_argument("--knn-ks", default="5,20")
    parser.add_argument("--ksg-preprocesses", default="global,zscore")
    parser.add_argument("--ksg-dims", default="32")
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} offset={args.offset} samples={args.samples}", flush=True)
    feature_cache_dir = Path(args.feature_cache_dir)
    feature_cache_dir.mkdir(parents=True, exist_ok=True)
    batch_cache = feature_cache_dir / f"batches_offset{args.offset}_n{args.samples}_bs{args.batch_size}.pt"
    if batch_cache.exists():
        batches = torch.load(batch_cache, map_location="cpu", weights_only=False)
        print(f"loaded batch_cache={batch_cache}", flush=True)
    else:
        batches = _collect_offset_batches(
            args.data_config,
            offsets=[args.offset],
            samples_per_offset=args.samples,
            batch_size=args.batch_size,
            respect_valid=True,
        )
        torch.save(batches, batch_cache)
        print(f"wrote batch_cache={batch_cache}", flush=True)
    print(f"cached_batches={len(batches)}", flush=True)

    feature_names = [x.strip() for x in args.features.split(",") if x.strip()]
    target_names = [x.strip() for x in args.targets.split(",") if x.strip()]
    knn_ks = [int(x) for x in args.knn_ks.split(",") if x.strip()]
    ksg_preprocesses = [x.strip() for x in args.ksg_preprocesses.split(",") if x.strip()]
    ksg_dims = [int(x) for x in args.ksg_dims.split(",") if x.strip()]
    rows = []
    for spec in args.model:
        name, kind, config, ckpt, view = _parse_model(spec)
        data = _extract_or_load(name, kind, config, ckpt, view, batches, device, feature_cache_dir)
        split = _split(data["x"].shape[0], args.seed)
        targets = {
            "action_sum": data["action_sum"],
            "full_seq": data["action_seq"],
        }
        variants = _feature_variants(data["x"])
        for target_name in target_names:
            y = targets[target_name]
            for feature_name in feature_names:
                x = variants[feature_name]
                base = {
                    "name": name,
                    "kind": kind,
                    "feature_view": view,
                    "feature": feature_name,
                    "target": target_name,
                    "samples": int(x.shape[0]),
                    "feature_dim": int(x.shape[1]),
                    "target_dim": int(y.shape[1]),
                }
                row = dict(base)
                row.update(_ridge_eval(x, y, split))
                rows.append(row)
                print(row, flush=True)
                for distance in ("euclidean", "cosine"):
                    for knn_k in knn_ks:
                        row = dict(base)
                        row.update(_knn_eval(x, y, split, knn_k, distance))
                        rows.append(row)
                        print(row, flush=True)
                for preprocess in ksg_preprocesses:
                    for dim in ksg_dims:
                        row = dict(base)
                        row.update(
                            {
                                "metric": "ksg_mi",
                                "k": "",
                                "distance": "",
                                "alpha": "",
                                "ksg_preprocess": preprocess,
                                "ksg_dim": int(dim),
                                "ksg_mi_bits": _ksg_eval(
                                    x,
                                    y,
                                    args.ksg_samples,
                                    dim,
                                    args.seed,
                                    preprocess,
                                    args.ksg_k,
                                ),
                            }
                        )
                        rows.append(row)
                        print(row, flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({key for row in rows for key in row.keys()}))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
