#!/usr/bin/env python3
"""Evaluate Bridge fixed-offset action/latent neighborhood alignment for LAMs."""

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict

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
from eval_lerobot_offset_geometry import _metrics  # noqa: E402
from eval_offset_norms import _collect_offset_batches, _to_device  # noqa: E402


def _parse_model(spec: str) -> tuple[str, str, str, str]:
    parts = spec.split("|")
    if len(parts) != 4:
        raise ValueError("model spec must be name|kind|config|ckpt")
    return parts[0], parts[1], parts[2], parts[3]


def _future_zq(outputs: Dict, batch_size: int) -> torch.Tensor:
    z = outputs["z_q"].detach().float()
    if z.shape[0] == batch_size and z.ndim >= 4:
        z = z[:, -1]
    elif z.shape[0] != batch_size:
        z = z.reshape(batch_size, -1, *z.shape[1:])[:, -1]
    return z.reshape(batch_size, -1)


def _masked_action_sum(batch: Dict, batch_size: int) -> torch.Tensor:
    if "radprog_action_sequence" not in batch:
        return batch["action"].detach().float().reshape(batch_size, -1)

    seq = batch["radprog_action_sequence"].detach().float()
    if seq.shape[0] != batch_size:
        seq = seq.reshape(batch_size, -1, seq.shape[-1])
    else:
        seq = seq.reshape(batch_size, -1, seq.shape[-1])

    mask = batch.get("radprog_action_sequence_mask")
    if mask is None:
        return seq.sum(dim=1)

    mask = mask.detach().float()
    if mask.shape[0] != batch_size:
        mask = mask.reshape(batch_size, -1)
    else:
        mask = mask.reshape(batch_size, -1)

    steps = min(seq.shape[1], mask.shape[1])
    seq = seq[:, :steps]
    mask = mask[:, :steps]
    return (seq * mask.unsqueeze(-1)).sum(dim=1)


def _encode_batches(
    model: torch.nn.Module,
    batches: list[Dict],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    zs, actions = [], []
    with torch.no_grad():
        for idx, batch_cpu in enumerate(batches, start=1):
            batch = _to_device(batch_cpu, device)
            batch_size = int(batch["videos"].shape[0])
            outputs = model.lam.vq_encode(batch["videos"])
            zs.append(_future_zq(outputs, batch_size).cpu().numpy())
            actions.append(_masked_action_sum(batch, batch_size).cpu().numpy())
            print(f"encode batch={idx}/{len(batches)} samples={sum(x.shape[0] for x in zs)}", flush=True)
    return np.concatenate(zs, axis=0), np.concatenate(actions, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, help="name|kind|config|ckpt")
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--offsets", default="1,5,9")
    parser.add_argument("--samples-per-offset", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latent-preprocess", default="raw,zscore")
    parser.add_argument("--triplets-per-query", type=int, default=8)
    parser.add_argument("--feature-cache-dir", default="outputs/analysis/cache/bridge_offset_geometry")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    offsets = [int(x) for x in args.offsets.split(",") if x.strip()]
    variants = [x for x in args.latent_preprocess.split(",") if x.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} offsets={offsets}", flush=True)
    print(f"data_config={args.data_config}", flush=True)

    cached = _collect_offset_batches(
        args.data_config,
        offsets=offsets,
        samples_per_offset=args.samples_per_offset,
        batch_size=args.batch_size,
    )
    batches_by_offset: dict[int, list[Dict]] = defaultdict(list)
    for offset, batch in cached:
        batches_by_offset[int(offset)].append(batch)
    print(
        "cached "
        + ", ".join(
            f"k={offset}:n={sum(int(b['videos'].shape[0]) for b in batches)}"
            for offset, batches in sorted(batches_by_offset.items())
        ),
        flush=True,
    )

    rows = []
    for spec in args.model:
        model_name, kind, config, ckpt = _parse_model(spec)
        print(f"[model] loading {model_name}", flush=True)
        model = _load_model(kind, config, ckpt, device)
        for offset in offsets:
            cache_path = (
                Path(args.feature_cache_dir)
                / f"{model_name}_bridge_k{offset}_n{args.samples_per_offset}.npz"
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            if cache_path.exists():
                payload = np.load(cache_path)
                z = payload["z"]
                action = payload["action"]
                print(f"[cache] {cache_path}", flush=True)
            else:
                z, action = _encode_batches(model, batches_by_offset[offset], device)
                np.savez_compressed(cache_path, z=z, action=action)
                print(f"[write] {cache_path} z={z.shape} action={action.shape}", flush=True)

            for variant in variants:
                row = {
                    "dataset": "bridge",
                    "model": model_name,
                    "kind": kind,
                    "offset_k": int(offset),
                    "samples": int(z.shape[0]),
                    "latent_dim": int(z.shape[1]),
                    "action_dim": int(action.shape[1]),
                    **_metrics(z, action, variant, args.seed + offset, args.triplets_per_query),
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
