#!/usr/bin/env python3
"""Inspect factorized LAM radius-token distribution at fixed offset k=9."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = Path(__file__).resolve().parents[1]
LATENT_ROOT = Path(__file__).resolve().parent
for path in (str(REPO_ROOT), str(LATENT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from eval_ksg_mi import _load_model  # noqa: E402
from eval_offset_norms import _collect_offset_batches, _to_device  # noqa: E402


DEFAULT_CONFIG = "latent_action_model/config/lam-visual-vq-bridge-hyperbolic-factorized-50k.yaml"
DEFAULT_CKPT = (
    "/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong/outputs/lam_bridge/logs/"
    "visual_vq_lam_bridge_hyperbolic_factorized_rad16_0to4_dir16_rad1_dir4tokens_prelift_hmax9_50k_workers2/"
    "epoch=0-step=30000.ckpt"
)


def _future_factorized(outputs: Dict[str, torch.Tensor], batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if "radius_indices" not in outputs or "direction_indices" not in outputs:
        raise ValueError("model outputs do not contain factorized radius/direction indices")
    radius = outputs["radius_indices"].detach().reshape(-1).long()
    direction = outputs["direction_indices"].detach().reshape(radius.numel(), -1).long()
    return radius[-batch_size:], direction[-batch_size:]


def _action_sequence_sum_norm(batch: Dict, batch_size: int) -> np.ndarray:
    if "radprog_action_sequence" not in batch:
        return np.full((batch_size,), np.nan, dtype=np.float64)
    action = batch["radprog_action_sequence"].detach().float()
    action = action.reshape(batch_size, -1, action.shape[-1])
    if "radprog_valid" in batch:
        # For forced k=9 this should be valid for all retained samples, but keep the mask explicit.
        valid = batch["radprog_valid"].detach().reshape(batch_size).float().cpu().numpy() > 0.5
    else:
        valid = np.ones(batch_size, dtype=bool)
    summed = action.sum(dim=1)
    norms = torch.linalg.vector_norm(summed, dim=-1).cpu().numpy().astype(np.float64)
    norms[~valid] = np.nan
    return norms


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if x.size < 2 or x.std() <= 1e-12 or y.std() <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(x.size, dtype=np.float64)
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if x.size < 2:
        return float("nan")
    return _pearson(_rankdata(x), _rankdata(y))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--offset", type=int, default=9)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--respect-valid", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/analysis/radius_k9"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valid_tag = "valid" if args.respect_valid else "novalid"
    print(
        f"device={device} offset={args.offset} samples={args.samples} respect_valid={args.respect_valid}",
        flush=True,
    )

    batches = _collect_offset_batches(
        args.config,
        offsets=[args.offset],
        samples_per_offset=args.samples,
        batch_size=args.batch_size,
        respect_valid=args.respect_valid,
    )
    print(f"cached_batches={len(batches)}", flush=True)

    model = _load_model("visual_vq", args.config, args.ckpt, device)
    radii = model.lam.vq.radius_values().detach().cpu().float().numpy()

    radius_ids = []
    radius_values = []
    direction_ids = []
    action_sum_norms = []
    with torch.no_grad():
        for batch_idx, (_, batch_cpu) in enumerate(batches, start=1):
            batch = _to_device(batch_cpu, device)
            batch_size = int(batch["videos"].shape[0])
            outputs = model.lam.vq_encode(batch["videos"])
            radius, direction = _future_factorized(outputs, batch_size)
            radius_np = radius.cpu().numpy().astype(np.int64)
            direction_np = direction.cpu().numpy().astype(np.int64)
            radius_ids.append(radius_np)
            radius_values.append(radii[radius_np])
            direction_ids.append(direction_np)
            action_sum_norms.append(_action_sequence_sum_norm(batch, batch_size))
            print(
                f"batch={batch_idx}/{len(batches)} collected={sum(x.size for x in radius_ids)}",
                flush=True,
            )

    radius_ids_np = np.concatenate(radius_ids, axis=0)[: args.samples]
    radius_values_np = np.concatenate(radius_values, axis=0)[: args.samples]
    direction_ids_np = np.concatenate(direction_ids, axis=0)[: args.samples]
    action_norm_np = np.concatenate(action_sum_norms, axis=0)[: args.samples]

    counts = np.bincount(radius_ids_np, minlength=len(radii))
    rows = []
    total = int(radius_ids_np.shape[0])
    for idx, count in enumerate(counts):
        rows.append(
            {
                "offset": int(args.offset),
                "radius_id": int(idx),
                "radius_value": float(radii[idx]),
                "count": int(count),
                "frac": float(count / max(total, 1)),
            }
        )
    _write_csv(args.out_dir / f"hyper_radius_distribution_k{args.offset}_{valid_tag}_n{total}.csv", rows)

    sample_rows = []
    for i in range(total):
        row = {
            "sample_idx": int(i),
            "offset": int(args.offset),
            "radius_id": int(radius_ids_np[i]),
            "radius_value": float(radius_values_np[i]),
            "action_sum_norm": float(action_norm_np[i]),
        }
        for j in range(direction_ids_np.shape[1]):
            row[f"direction_id_{j}"] = int(direction_ids_np[i, j])
        sample_rows.append(row)
    _write_csv(args.out_dir / f"hyper_radius_samples_k{args.offset}_{valid_tag}_n{total}.csv", sample_rows)

    summary = {
        "offset": int(args.offset),
        "samples": total,
        "radius_id_mean": float(radius_ids_np.mean()),
        "radius_id_std": float(radius_ids_np.std()),
        "radius_value_mean": float(radius_values_np.mean()),
        "radius_value_std": float(radius_values_np.std()),
        "radius_value_p10": float(np.percentile(radius_values_np, 10)),
        "radius_value_p50": float(np.percentile(radius_values_np, 50)),
        "radius_value_p90": float(np.percentile(radius_values_np, 90)),
        "radius_value_min": float(radius_values_np.min()),
        "radius_value_max": float(radius_values_np.max()),
        "radius_action_sum_norm_pearson": _pearson(radius_values_np, action_norm_np),
        "radius_action_sum_norm_spearman": _spearman(radius_values_np, action_norm_np),
    }
    _write_csv(args.out_dir / f"hyper_radius_summary_k{args.offset}_{valid_tag}_n{total}.csv", [summary])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)
    axes[0].bar(np.arange(len(radii)), counts, color="#4C78A8")
    axes[0].set_xlabel("radius code id")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"Hyper factorized radius ids at k={args.offset}")
    axes[0].set_xticks(np.arange(len(radii)))

    axes[1].scatter(radius_values_np, action_norm_np, s=9, alpha=0.35, color="#F58518", linewidths=0)
    axes[1].set_xlabel("selected radius value")
    axes[1].set_ylabel("||sum action sequence||")
    axes[1].set_title(
        "within-k action magnitude\n"
        f"Pearson={summary['radius_action_sum_norm_pearson']:.3f}, "
        f"Spearman={summary['radius_action_sum_norm_spearman']:.3f}"
    )

    png = args.out_dir / f"hyper_radius_distribution_k{args.offset}_{valid_tag}_n{total}.png"
    pdf = args.out_dir / f"hyper_radius_distribution_k{args.offset}_{valid_tag}_n{total}.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")

    print(f"wrote {args.out_dir / f'hyper_radius_distribution_k{args.offset}_{valid_tag}_n{total}.csv'}")
    print(f"wrote {args.out_dir / f'hyper_radius_samples_k{args.offset}_{valid_tag}_n{total}.csv'}")
    print(f"wrote {args.out_dir / f'hyper_radius_summary_k{args.offset}_{valid_tag}_n{total}.csv'}")
    print(f"wrote {png}")
    print(f"summary={summary}", flush=True)


if __name__ == "__main__":
    main()
