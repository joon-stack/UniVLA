import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy.special import digamma
from scipy.spatial import cKDTree


def _global_scale(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (x - float(x.mean())) / (float(x.std()) + eps)


def _zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + eps)


def _project_x_only(x: np.ndarray, dim: int, seed: int) -> np.ndarray:
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


def _feature(data: dict[str, np.ndarray], name: str) -> np.ndarray:
    z_raw = data["z_raw"].astype(np.float64)
    if name == "raw":
        return z_raw
    if name == "raw_global":
        return _global_scale(z_raw)
    if name == "zscore":
        return _zscore(z_raw)
    if name == "sample_unit":
        return z_raw / np.maximum(np.linalg.norm(z_raw, axis=1, keepdims=True), 1e-8)
    if name == "token_unit":
        return data["z_token_unit"].astype(np.float64)
    if name == "token_norms":
        return data["z_token_norms"].astype(np.float64)
    if name == "unit_plus_norms":
        return np.concatenate(
            [data["z_token_unit"].astype(np.float64), _global_scale(data["z_token_norms"].astype(np.float64))],
            axis=1,
        )
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument("--name", action="append", required=True)
    parser.add_argument("--features", default="raw,raw_global,zscore,sample_unit,token_unit,unit_plus_norms")
    parser.add_argument("--dims", default="16,32,64,128,256")
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--ksg-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preprocess-y", choices=["raw", "global", "zscore"], default="raw")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = []
    dims = [int(x) for x in args.dims.split(",") if x]
    features = [x for x in args.features.split(",") if x]
    rng = np.random.default_rng(args.seed)

    for name in args.name:
        data = dict(np.load(Path(args.feature_cache_dir) / f"{name}.npz"))
        y_all = data["action_seq"].astype(np.float64)
        n = min(args.samples, y_all.shape[0])
        idx = rng.choice(y_all.shape[0], size=n, replace=False)
        y = y_all[idx]
        if args.preprocess_y == "global":
            y = _global_scale(y)
        elif args.preprocess_y == "zscore":
            y = _zscore(y)

        for feat in features:
            x_all = _feature(data, feat)
            x_base = x_all[idx]
            for dim in dims:
                x = _project_x_only(x_base, dim, args.seed)
                mi = _ksg_bits(x, y, args.ksg_k)
                row = {
                    "name": name,
                    "feature": feat,
                    "samples": int(n),
                    "x_dim_raw": int(x_base.shape[1]),
                    "x_dim": int(x.shape[1]),
                    "y_dim": int(y.shape[1]),
                    "preprocess_y": args.preprocess_y,
                    "ksg_k": int(args.ksg_k),
                    "ksg_mi_bits": mi,
                }
                rows.append(row)
                print(row, flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
