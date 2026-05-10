#!/usr/bin/env python3
"""Evaluate z_q neighborhood geometry against action sequence geometry.

This script intentionally uses continuous z_q-derived features, not token ids.
It reports:

1. Action neighborhood preservation:
   - overlap between top-k z_q neighbors and top-k action neighbors
   - action cosine of z_q neighbors
   - lift over random same-action-norm-bin neighbors

2. Local action prediction from neighbors:
   - action_hat_i = mean action of top-k z_q neighbors
   - MSE, cosine, R2
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch


MODEL_NAMES = [
    "factorized_hyper_30k",
    "euclidean_visual_vq_30k",
    "univla_stage2_60k",
]


def unit(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def topk_from_score(score: np.ndarray, k: int) -> np.ndarray:
    score = score.copy()
    np.fill_diagonal(score, -np.inf)
    idx = np.argpartition(-score, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(score.shape[0])[:, None]
    order = np.argsort(-score[rows, idx], axis=1)
    return idx[rows, order]


def pairwise_score(x: np.ndarray, metric: str) -> np.ndarray:
    if metric == "cosine":
        xu = unit(x)
        return xu @ xu.T
    if metric == "euclidean":
        x2 = np.sum(x * x, axis=1, keepdims=True)
        return -(x2 + x2.T - 2.0 * (x @ x.T))
    raise ValueError(f"unknown metric: {metric}")


def action_cosines(action: np.ndarray, nn: np.ndarray) -> np.ndarray:
    au = unit(action)
    sim = au @ au.T
    rows = np.arange(len(action))[:, None]
    return sim[rows, nn].mean(axis=1)


def action_mse_to_neighbors(action: np.ndarray, nn: np.ndarray) -> np.ndarray:
    rows = np.arange(len(action))[:, None]
    diff = action[rows] - action[nn]
    return np.mean(diff * diff, axis=(1, 2))


def overlap_metrics(nn_z: np.ndarray, nn_a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    recalls = []
    jaccards = []
    k = nn_z.shape[1]
    for z, a in zip(nn_z, nn_a):
        inter = len(set(z.tolist()).intersection(a.tolist()))
        recalls.append(inter / k)
        jaccards.append(inter / (2 * k - inter))
    return np.asarray(recalls), np.asarray(jaccards)


def load_tasks(batch_cache: str | None, n: int) -> list[str] | None:
    if not batch_cache:
        return None
    obj = torch.load(batch_cache, map_location="cpu")
    tasks: list[str] = []
    for _, batch in obj["batches"]:
        tasks.extend([str(x) for x in batch["task_instruction"]])
    return tasks[:n]


def random_neighbors(
    action: np.ndarray,
    k: int,
    bins: int,
    tasks: list[str] | None,
    seed: int,
    require_task: bool,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    n = len(action)
    action_norm = np.linalg.norm(action, axis=1)
    qs = np.quantile(action_norm, np.linspace(0, 1, bins + 1))
    qs[0] -= 1e-9
    qs[-1] += 1e-9
    bin_id = np.searchsorted(qs[1:-1], action_norm, side="right")
    nn = np.empty((n, k), dtype=np.int64)
    strict = 0
    all_idx = np.arange(n)
    for i in range(n):
        mask = bin_id == bin_id[i]
        mask[i] = False
        if require_task and tasks is not None:
            task_mask = np.asarray([t == tasks[i] for t in tasks], dtype=bool)
            task_mask[i] = False
            strict_mask = mask & task_mask
            if strict_mask.sum() >= k:
                pool = all_idx[strict_mask]
                strict += 1
            else:
                pool = all_idx[mask]
        else:
            pool = all_idx[mask]
        if len(pool) == 0:
            pool = np.delete(all_idx, i)
        replace = len(pool) < k
        nn[i] = rng.choice(pool, size=k, replace=replace)
    return nn, strict / n


def prediction_metrics(action: np.ndarray, nn: np.ndarray) -> dict[str, float]:
    pred = action[nn].mean(axis=1)
    err = pred - action
    mse_sample = np.mean(err * err, axis=1)
    cos_sample = np.sum(unit(pred) * unit(action), axis=1)
    sse = np.sum(err * err)
    centered = action - action.mean(axis=0, keepdims=True)
    sst = np.sum(centered * centered)
    return {
        "mse": float(mse_sample.mean()),
        "cosine": float(cos_sample.mean()),
        "r2": float(1.0 - sse / max(sst, 1e-12)),
    }


def bootstrap_diff(a: np.ndarray, b: np.ndarray, seed: int, samples: int) -> tuple[float, float, float, float]:
    rng = np.random.default_rng(seed)
    diff = a - b
    vals = []
    for _ in range(samples):
        idx = rng.integers(0, len(diff), len(diff))
        vals.append(float(diff[idx].mean()))
    vals = np.asarray(vals)
    return float(diff.mean()), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)), float((vals > 0).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="outputs/analysis/cache/zq_full_action_k9_all_2048")
    parser.add_argument("--batch-cache", default="outputs/analysis/cache/bridge_k9_all_2048_bs16.pt")
    parser.add_argument("--output-dir", default="outputs/analysis")
    parser.add_argument("--ks", default="5,10,20,50")
    parser.add_argument("--norm-bins", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ks = [int(x) for x in args.ks.split(",") if x]

    data = {}
    for name in MODEL_NAMES:
        d = np.load(cache_dir / f"{name}.npz")
        data[name] = {
            "z_raw": np.asarray(d["z_raw"], dtype=np.float64),
            "action": np.asarray(d["action_seq"], dtype=np.float64),
        }

    n = len(next(iter(data.values()))["action"])
    tasks = load_tasks(args.batch_cache, n) if args.batch_cache and os.path.exists(args.batch_cache) else None
    action = next(iter(data.values()))["action"]

    random_by_k = {}
    for k in ks:
        random_by_k[(k, "normbin")] = (*random_neighbors(action, k, args.norm_bins, tasks, 1000 + k, False),)
        random_by_k[(k, "task_normbin_or_normbin")] = (*random_neighbors(action, k, args.norm_bins, tasks, 2000 + k, True),)

    rows = []
    per_sample = {}
    for name, d in data.items():
        z = d["z_raw"]
        action_score_e = pairwise_score(action, "euclidean")
        action_score_c = pairwise_score(action, "cosine")
        z_scores = {
            "euclidean": pairwise_score(z, "euclidean"),
            "cosine": pairwise_score(z, "cosine"),
        }
        for k in ks:
            nn_action = {
                "euclidean": topk_from_score(action_score_e, k),
                "cosine": topk_from_score(action_score_c, k),
            }
            for z_metric, z_score in z_scores.items():
                nn_z = topk_from_score(z_score, k)
                pred = prediction_metrics(action, nn_z)
                z_action_cos = action_cosines(action, nn_z)
                z_action_mse = action_mse_to_neighbors(action, nn_z)

                for action_metric, nn_a in nn_action.items():
                    recall, jaccard = overlap_metrics(nn_z, nn_a)
                    rows.append({
                        "section": "neighborhood_preservation",
                        "name": name,
                        "z_feature": "z_raw",
                        "z_metric": z_metric,
                        "action_metric": action_metric,
                        "k": k,
                        "baseline": "",
                        "strict_task_baseline_frac": "",
                        "recall_at_k": float(recall.mean()),
                        "jaccard_at_k": float(jaccard.mean()),
                        "action_cos_at_k": float(z_action_cos.mean()),
                        "action_mse_at_k": float(z_action_mse.mean()),
                        "lift_action_cos": "",
                        "pred_mse": "",
                        "pred_cosine": "",
                        "pred_r2": "",
                    })
                    per_sample[(name, z_metric, action_metric, k, "recall")] = recall
                    per_sample[(name, z_metric, action_metric, k, "jaccard")] = jaccard

                for baseline_name in ["normbin", "task_normbin_or_normbin"]:
                    nn_rand, strict_frac = random_by_k[(k, baseline_name)]
                    rand_cos = action_cosines(action, nn_rand)
                    lift = z_action_cos - rand_cos
                    rows.append({
                        "section": "action_cos_lift",
                        "name": name,
                        "z_feature": "z_raw",
                        "z_metric": z_metric,
                        "action_metric": "cosine",
                        "k": k,
                        "baseline": baseline_name,
                        "strict_task_baseline_frac": strict_frac,
                        "recall_at_k": "",
                        "jaccard_at_k": "",
                        "action_cos_at_k": float(z_action_cos.mean()),
                        "action_mse_at_k": float(z_action_mse.mean()),
                        "lift_action_cos": float(lift.mean()),
                        "pred_mse": "",
                        "pred_cosine": "",
                        "pred_r2": "",
                    })
                    per_sample[(name, z_metric, baseline_name, k, "lift")] = lift
                    per_sample[(name, z_metric, baseline_name, k, "action_cos")] = z_action_cos

                rows.append({
                    "section": "local_action_prediction",
                    "name": name,
                    "z_feature": "z_raw",
                    "z_metric": z_metric,
                    "action_metric": "sequence",
                    "k": k,
                    "baseline": "z_neighbors",
                    "strict_task_baseline_frac": "",
                    "recall_at_k": "",
                    "jaccard_at_k": "",
                    "action_cos_at_k": "",
                    "action_mse_at_k": "",
                    "lift_action_cos": "",
                    "pred_mse": pred["mse"],
                    "pred_cosine": pred["cosine"],
                    "pred_r2": pred["r2"],
                })
                per_sample[(name, z_metric, "sequence", k, "pred_mse")] = np.mean((action[nn_z].mean(axis=1) - action) ** 2, axis=1)
                per_sample[(name, z_metric, "sequence", k, "pred_cosine")] = np.sum(unit(action[nn_z].mean(axis=1)) * unit(action), axis=1)

    result_csv = out_dir / "zq_neighbor_geometry_k9_2048.csv"
    fieldnames = list(rows[0].keys())
    with result_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    compare_rows = []
    hyper = "factorized_hyper_30k"
    for other in ["euclidean_visual_vq_30k", "univla_stage2_60k"]:
        for key, hvals in per_sample.items():
            name, z_metric, context, k, metric = key
            if name != hyper:
                continue
            other_key = (other, z_metric, context, k, metric)
            if other_key not in per_sample:
                continue
            ovals = per_sample[other_key]
            # For MSE, lower is better, so report other - hyper as positive if hyper wins.
            if metric == "pred_mse":
                diff, lo, hi, prob = bootstrap_diff(ovals, hvals, 3000 + k, args.bootstrap)
                direction = "positive_means_hyper_lower"
            else:
                diff, lo, hi, prob = bootstrap_diff(hvals, ovals, 3000 + k, args.bootstrap)
                direction = "positive_means_hyper_higher"
            compare_rows.append({
                "other": other,
                "z_metric": z_metric,
                "context": context,
                "k": k,
                "metric": metric,
                "direction": direction,
                "hyper_minus_or_advantage": diff,
                "ci95_low": lo,
                "ci95_high": hi,
                "boot_prob_hyper_better": prob,
            })

    compare_csv = out_dir / "zq_neighbor_geometry_k9_2048_compare.csv"
    with compare_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(compare_rows[0].keys()))
        writer.writeheader()
        writer.writerows(compare_rows)

    print(f"wrote {result_csv}")
    print(f"wrote {compare_csv}")
    print("\nHyper wins vs Euclidean with bootstrap prob >= 0.95:")
    for row in compare_rows:
        if row["other"] == "euclidean_visual_vq_30k" and row["boot_prob_hyper_better"] >= 0.95:
            print(row)

    print("\nBest compact rows, k=20, cosine z metric:")
    for row in rows:
        if row["k"] == 20 and row["z_metric"] == "cosine" and row["section"] in {"action_cos_lift", "local_action_prediction"}:
            if row["baseline"] in {"normbin", "z_neighbors"}:
                print(row)


if __name__ == "__main__":
    main()
