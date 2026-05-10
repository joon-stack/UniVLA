#!/usr/bin/env python3
"""Probe cached z_q features against action targets with ridge and kNN.

This intentionally uses z_q-derived continuous features, not token ids.
"""

import argparse
import csv
from pathlib import Path

import numpy as np


def _global_scale(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (x - float(x.mean())) / (float(x.std()) + eps)


def _zscore_fit(x: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    return x.mean(axis=0, keepdims=True), x.std(axis=0, keepdims=True) + eps


def _zscore_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def _sample_unit(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def _features(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    z_raw = data["z_raw"].astype(np.float64)
    z_unit = data["z_token_unit"].astype(np.float64)
    z_norms = data["z_token_norms"].astype(np.float64)
    return {
        "zq_raw": z_raw,
        "zq_raw_global": _global_scale(z_raw),
        "zq_sample_unit": _sample_unit(z_raw),
        "zq_token_unit": z_unit,
        "zq_token_norms": _global_scale(z_norms),
        "zq_unit_plus_norms": np.concatenate([z_unit, _global_scale(z_norms)], axis=1),
    }


def _targets(action_seq_flat: np.ndarray) -> dict[str, np.ndarray]:
    seq = action_seq_flat.astype(np.float64).reshape(action_seq_flat.shape[0], -1, 7)
    action_sum = seq.sum(axis=1)
    action_mean = seq.mean(axis=1)
    first = seq[:, 0]
    last = seq[:, -1]
    return {
        "full_seq": seq.reshape(seq.shape[0], -1),
        "action_sum": action_sum,
        "action_mean": action_mean,
        "first_action": first,
        "last_action": last,
        "translation_sum": action_sum[:, :3],
        "rotation_sum": action_sum[:, 3:6],
        "gripper_sum": action_sum[:, 6:7],
        "seq_norm": np.linalg.norm(seq.reshape(seq.shape[0], -1), axis=1, keepdims=True),
        "sum_norm": np.linalg.norm(action_sum, axis=1, keepdims=True),
        "translation_norm": np.linalg.norm(action_sum[:, :3], axis=1, keepdims=True),
        "rotation_norm": np.linalg.norm(action_sum[:, 3:6], axis=1, keepdims=True),
    }


def _split(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    a = int(0.70 * n)
    b = int(0.85 * n)
    return idx[:a], idx[a:b], idx[b:]


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
    ytr = y[tr]
    best = None
    for alpha in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0):
        w = _ridge_fit(xtr, ytr, alpha)
        pred = _ridge_predict(xva, w)
        val_r2 = _r2(y[va], pred)
        if best is None or val_r2 > best[0]:
            best = (val_r2, alpha, w)
    assert best is not None
    pred = _ridge_predict(xte, best[2])
    return {
        "metric": "ridge",
        "k": "",
        "distance": "",
        "alpha": float(best[1]),
        "val_r2": float(best[0]),
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
        xtr = _sample_unit(xtr)
        xte = _sample_unit(xte)
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
        "metric": "knn",
        "k": int(k),
        "distance": distance,
        "alpha": "",
        "val_r2": "",
        "test_r2": _r2(yte, pred),
        "test_mse": _mse(yte, pred),
        "test_cosine": _cosine(yte, pred),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="outputs/analysis/cache/zq_full_action_k9_all_2048")
    parser.add_argument("--names", default="factorized_hyper_30k,euclidean_visual_vq_30k,univla_stage2_60k")
    parser.add_argument("--output", default="outputs/analysis/zq_probe_knn_from_cache_k9_all_2048.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = []
    cache_dir = Path(args.cache_dir)
    for name in [x.strip() for x in args.names.split(",") if x.strip()]:
        data = dict(np.load(cache_dir / f"{name}.npz"))
        split = _split(data["action_seq"].shape[0], args.seed)
        for target_name, y in _targets(data["action_seq"]).items():
            for feature_name, x in _features(data).items():
                base = {
                    "name": name,
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
                    for k in (1, 5, 20, 50):
                        row = dict(base)
                        row.update(_knn_eval(x, y, split, k, distance))
                        rows.append(row)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "feature",
        "target",
        "metric",
        "k",
        "distance",
        "alpha",
        "samples",
        "feature_dim",
        "target_dim",
        "val_r2",
        "test_r2",
        "test_mse",
        "test_cosine",
    ]
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
