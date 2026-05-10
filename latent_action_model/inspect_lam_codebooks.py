#!/usr/bin/env python3
"""Dump and summarize VQ codebooks from LAM Lightning checkpoints."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_CKPTS = {
    "hyper_factorized_30k": "/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong/outputs/lam_bridge/logs/visual_vq_lam_bridge_hyperbolic_factorized_rad16_0to4_dir16_rad1_dir4tokens_prelift_hmax9_50k_workers2/epoch=0-step=30000.ckpt",
    "euclid_visual_vq_30k": "/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong/outputs/lam_bridge/logs/visual_vq_lam_bridge_euclidean_control_hmax9_50k_t2mid_t2future_metricgroups/epoch=0-step=30000.ckpt",
    "univla_stage2_60k": "/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong/outputs/lam_bridge/logs/task_centric_lam_stage2_bridge/epoch=0-step=60000.ckpt",
}


@dataclass
class Codebook:
    name: str
    kind: str
    weight: np.ndarray
    source_key: str
    radius_values: np.ndarray | None = None
    direction_values: np.ndarray | None = None


def safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().float().numpy()


def offdiag_cosine_stats(weight: np.ndarray) -> dict[str, float]:
    if weight.shape[0] < 2:
        return {
            "cos_mean": float("nan"),
            "cos_std": float("nan"),
            "cos_min": float("nan"),
            "cos_max": float("nan"),
            "cos_abs_mean": float("nan"),
        }
    normed = weight / np.maximum(np.linalg.norm(weight, axis=1, keepdims=True), 1e-12)
    cosine = normed @ normed.T
    mask = ~np.eye(cosine.shape[0], dtype=bool)
    vals = cosine[mask]
    return {
        "cos_mean": float(vals.mean()),
        "cos_std": float(vals.std()),
        "cos_min": float(vals.min()),
        "cos_max": float(vals.max()),
        "cos_abs_mean": float(np.abs(vals).mean()),
    }


def summarize_weight(weight: np.ndarray) -> dict[str, float]:
    norms = np.linalg.norm(weight, axis=1)
    stats = {
        "n_codes": int(weight.shape[0]),
        "dim": int(weight.shape[1]) if weight.ndim == 2 else int(np.prod(weight.shape[1:])),
        "norm_min": float(norms.min()),
        "norm_p10": float(np.quantile(norms, 0.10)),
        "norm_p50": float(np.quantile(norms, 0.50)),
        "norm_p90": float(np.quantile(norms, 0.90)),
        "norm_max": float(norms.max()),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "coord_min": float(weight.min()),
        "coord_max": float(weight.max()),
        "coord_mean": float(weight.mean()),
        "coord_std": float(weight.std()),
    }
    stats.update(offdiag_cosine_stats(weight))
    return stats


def factorized_radius_values(delta_unconstrained: torch.Tensor, floor: float) -> torch.Tensor:
    deltas = F.softplus(delta_unconstrained.detach().cpu().float()) + floor
    # Current factorized checkpoints use radius[0] == 0, which stores only R-1 deltas.
    return torch.cat([torch.zeros(1), torch.cumsum(deltas, dim=0)], dim=0)


def extract_codebooks(state_dict: dict[str, torch.Tensor], radius_delta_floor: float) -> list[Codebook]:
    codebooks: list[Codebook] = []

    # Factorized VQ: radius_delta_unconstrained + direction_codebook.weight.
    for dkey, direction in sorted(state_dict.items()):
        if not dkey.endswith("direction_codebook.weight"):
            continue
        prefix = dkey[: -len("direction_codebook.weight")]
        rkey = prefix + "radius_delta_unconstrained"
        if rkey not in state_dict:
            continue
        radii = factorized_radius_values(state_dict[rkey], radius_delta_floor)
        directions = F.normalize(direction.detach().cpu().float(), dim=-1, eps=1e-6)
        full = (radii[:, None, None] * directions[None, :, :]).reshape(-1, directions.shape[-1])
        codebooks.append(
            Codebook(
                name=prefix.rstrip("."),
                kind="factorized_full_radius_x_direction",
                weight=to_numpy(full),
                source_key=f"{rkey}+{dkey}",
                radius_values=to_numpy(radii),
                direction_values=to_numpy(directions),
            )
        )
        codebooks.append(
            Codebook(
                name=prefix.rstrip(".") + ".direction",
                kind="factorized_direction_unit",
                weight=to_numpy(directions),
                source_key=dkey,
                radius_values=to_numpy(radii),
                direction_values=to_numpy(directions),
            )
        )

    # Plain VQ: embedding table.
    factorized_direction_keys = {cb.source_key.split("+")[-1] for cb in codebooks if "+" in cb.source_key}
    for key, weight in sorted(state_dict.items()):
        if not key.endswith("codebook.weight"):
            continue
        if ".vq" not in key:
            continue
        if key in factorized_direction_keys or key.endswith("direction_codebook.weight"):
            continue
        if weight.ndim != 2:
            continue
        codebooks.append(
            Codebook(
                name=key[: -len(".weight")],
                kind="plain_embedding",
                weight=to_numpy(weight),
                source_key=key,
            )
        )

    return codebooks


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_weight_csv(path: Path, weight: np.ndarray) -> None:
    rows = []
    for code_id, vector in enumerate(weight):
        row = {"code_id": int(code_id)}
        row.update({f"d{i}": float(value) for i, value in enumerate(vector)})
        rows.append(row)
    write_csv(path, rows)


def inspect_one(label: str, ckpt_path: Path, out_dir: Path, radius_delta_floor: float) -> list[dict[str, object]]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    codebooks = extract_codebooks(state_dict, radius_delta_floor)
    label_dir = out_dir / safe_name(label)
    label_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for codebook in codebooks:
        stem = safe_name(codebook.name)
        npz_kwargs = {"weight": codebook.weight}
        if codebook.radius_values is not None:
            npz_kwargs["radius_values"] = codebook.radius_values
        if codebook.direction_values is not None:
            npz_kwargs["direction_values"] = codebook.direction_values
        np.savez_compressed(label_dir / f"{stem}.npz", **npz_kwargs)
        write_weight_csv(label_dir / f"{stem}_weight.csv", codebook.weight)

        norms = np.linalg.norm(codebook.weight, axis=1)
        write_csv(
            label_dir / f"{stem}_norms.csv",
            [{"code_id": i, "norm": float(v)} for i, v in enumerate(norms)],
        )
        if codebook.radius_values is not None:
            write_csv(
                label_dir / f"{stem}_radius_values.csv",
                [{"radius_id": i, "radius": float(v)} for i, v in enumerate(codebook.radius_values)],
            )

        stats = summarize_weight(codebook.weight)
        rows.append(
            {
                "checkpoint": label,
                "global_step": int(ckpt.get("global_step", -1)),
                "codebook": codebook.name,
                "kind": codebook.kind,
                "source_key": codebook.source_key,
                "npz": str(label_dir / f"{stem}.npz"),
                **stats,
            }
        )

    write_csv(label_dir / "summary.csv", rows)
    return rows


def parse_ckpt_args(items: list[str]) -> dict[str, str]:
    if not items:
        return DEFAULT_CKPTS
    parsed: dict[str, str] = {}
    for item in items:
        if "=" in item:
            label, path = item.split("=", 1)
        else:
            path = item
            label = Path(path).parent.name + "_" + Path(path).stem
        parsed[label] = path
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="*", help="Either /path/to.ckpt or label=/path/to.ckpt")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/analysis/codebooks"))
    parser.add_argument("--radius-delta-floor", type=float, default=1e-5)
    args = parser.parse_args()

    all_rows: list[dict[str, object]] = []
    for label, path_str in parse_ckpt_args(args.checkpoints).items():
        ckpt_path = Path(path_str)
        if not ckpt_path.exists():
            print(f"[skip missing] {label}: {ckpt_path}")
            continue
        print(f"[inspect] {label}: {ckpt_path}")
        rows = inspect_one(label, ckpt_path, args.out_dir, args.radius_delta_floor)
        if not rows:
            print(f"  no codebook tensors found")
        for row in rows:
            print(
                "  "
                f"{row['codebook']} {row['kind']} "
                f"shape=({row['n_codes']},{row['dim']}) "
                f"norm_mean={row['norm_mean']:.4f} "
                f"norm_p50={row['norm_p50']:.4f} "
                f"norm_max={row['norm_max']:.4f} "
                f"cos_mean={row['cos_mean']:.4f} "
                f"cos_max={row['cos_max']:.4f}"
            )
        all_rows.extend(rows)

    write_csv(args.out_dir / "summary.csv", all_rows)
    print(f"[saved] {args.out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
