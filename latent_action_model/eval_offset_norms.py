import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch


os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = Path(__file__).resolve().parents[1]
LATENT_ROOT = Path(__file__).resolve().parent
for path in (str(REPO_ROOT), str(LATENT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from eval_ksg_mi import _class_kwargs, _load_datamodule, _load_model, _load_yaml  # noqa: E402


def _slice_batch(batch: Dict, mask: torch.Tensor, limit: int) -> Dict:
    indices = torch.nonzero(mask, as_tuple=False).flatten()[:limit]
    sliced = {}
    batch_size = int(mask.shape[0])
    for key, value in batch.items():
        if torch.is_tensor(value) and value.shape[:1] == (batch_size,):
            sliced[key] = value.index_select(0, indices).cpu()
        elif isinstance(value, np.ndarray) and value.shape[:1] == (batch_size,):
            sliced[key] = value[indices.cpu().numpy()]
        elif isinstance(value, list) and len(value) == batch_size:
            idx = indices.cpu().tolist()
            sliced[key] = [value[i] for i in idx]
        elif isinstance(value, tuple) and len(value) == batch_size:
            idx = indices.cpu().tolist()
            sliced[key] = tuple(value[i] for i in idx)
        else:
            sliced[key] = value
    return sliced


def _to_device(batch: Dict, device: torch.device) -> Dict:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def _collect_offset_batches(
    data_config: str,
    offsets: Iterable[int],
    samples_per_offset: int,
    batch_size: int,
    respect_valid: bool = True,
) -> list[tuple[int, Dict]]:
    cached: list[tuple[int, Dict]] = []
    previous_force = os.environ.get("UNIVLA_LAM_FORCE_H2_OFFSET")
    try:
        for offset in offsets:
            os.environ["UNIVLA_LAM_FORCE_H2_OFFSET"] = str(offset)
            dm = _load_datamodule(data_config, batch_size)
            seen = 0
            for batch in dm.test_dataloader():
                valid = batch.get("radprog_valid")
                if valid is None or not respect_valid:
                    valid_mask = torch.ones(batch["videos"].shape[0], dtype=torch.bool)
                else:
                    valid_mask = valid.reshape(-1).float() > 0.5
                remaining = samples_per_offset - seen
                if remaining <= 0:
                    break
                if int(valid_mask.sum()) == 0:
                    continue
                sliced = _slice_batch(batch, valid_mask, remaining)
                n = int(sliced["videos"].shape[0])
                cached.append((offset, sliced))
                seen += n
                print(f"cached offset={offset} samples={seen}/{samples_per_offset}", flush=True)
                if seen >= samples_per_offset:
                    break
            if seen < samples_per_offset:
                print(f"[warn] offset={offset} only cached {seen} samples", flush=True)
    finally:
        if previous_force is None:
            os.environ.pop("UNIVLA_LAM_FORCE_H2_OFFSET", None)
        else:
            os.environ["UNIVLA_LAM_FORCE_H2_OFFSET"] = previous_force
    return cached


def _future_tensor(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    tensor = tensor.detach().float()
    if tensor.shape[0] == batch_size and tensor.ndim >= 4:
        return tensor[:, -1]
    if tensor.shape[0] != batch_size:
        return tensor.reshape(batch_size, -1, *tensor.shape[1:])[:, -1]
    return tensor


def _sample_norm(tensor: torch.Tensor, batch_size: int) -> np.ndarray:
    future = _future_tensor(tensor, batch_size)
    norms = torch.linalg.vector_norm(future, dim=-1)
    if norms.ndim > 1:
        norms = norms.reshape(norms.shape[0], -1).mean(dim=1)
    return norms.detach().cpu().numpy()


def _summarize(values: list[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.shape[0]),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def _pearson(xs: list[float], ys: list[float]) -> float:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if x.size < 2 or x.std() <= 1e-12 or y.std() <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _evaluate_model(
    name: str,
    kind: str,
    config: str,
    ckpt: str,
    cached_batches: list[tuple[int, Dict]],
    device: torch.device,
) -> tuple[list[Dict], list[Dict]]:
    model = _load_model(kind, config, ckpt, device)
    grouped: dict[tuple[int, str], list[float]] = defaultdict(list)
    all_offsets: dict[str, list[float]] = defaultdict(list)
    all_norms: dict[str, list[float]] = defaultdict(list)

    with torch.no_grad():
        for batch_idx, (offset, batch_cpu) in enumerate(cached_batches, start=1):
            batch = _to_device(batch_cpu, device)
            outputs = model.lam.vq_encode(batch["videos"])
            batch_size = int(batch["videos"].shape[0])
            for key in ("emb", "z_q", "z"):
                if key not in outputs:
                    continue
                values = _sample_norm(outputs[key], batch_size)
                grouped[(offset, key)].extend(values.tolist())
                all_offsets[key].extend([float(offset)] * values.shape[0])
                all_norms[key].extend(values.tolist())
            print(f"eval name={name} batch={batch_idx}/{len(cached_batches)}", flush=True)

    rows = []
    pearson_rows = []
    for (offset, key), values in sorted(grouped.items()):
        row = {
            "name": name,
            "kind": kind,
            "offset": offset,
            "tensor": key,
        }
        row.update(_summarize(values))
        rows.append(row)

    for key in sorted(all_norms):
        pearson_rows.append(
            {
                "name": name,
                "kind": kind,
                "tensor": key,
                "offset_norm_pearson": _pearson(all_offsets[key], all_norms[key]),
                "samples": len(all_norms[key]),
            }
        )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows, pearson_rows


def _parse_model_spec(spec: str) -> tuple[str, str, str, str]:
    parts = spec.split("|")
    if len(parts) != 4:
        raise ValueError("model spec must be name|kind|config|ckpt")
    return parts[0], parts[1], parts[2], parts[3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, help="name|kind|config|ckpt")
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--offsets", default="2,3,4,5,6,7,8,9")
    parser.add_argument("--samples-per-offset", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    _ = _class_kwargs, _load_yaml
    offsets = [int(x) for x in args.offsets.split(",") if x.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} offsets={offsets}", flush=True)
    print(f"data_config={args.data_config}", flush=True)

    cached_batches = _collect_offset_batches(
        args.data_config,
        offsets=offsets,
        samples_per_offset=args.samples_per_offset,
        batch_size=args.batch_size,
    )
    print(f"cached_batches={len(cached_batches)}", flush=True)

    all_rows = []
    all_pearson_rows = []
    for spec in args.model:
        name, kind, config, ckpt = _parse_model_spec(spec)
        rows, pearson_rows = _evaluate_model(name, kind, config, ckpt, cached_batches, device)
        all_rows.extend(rows)
        all_pearson_rows.extend(pearson_rows)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["name", "kind", "offset", "tensor", "count", "mean", "std", "p10", "p50", "p90"],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    pearson_output = output.with_name(output.stem + "_pearson.csv")
    with open(pearson_output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["name", "kind", "tensor", "offset_norm_pearson", "samples"],
        )
        writer.writeheader()
        writer.writerows(all_pearson_rows)

    print(f"wrote {output}", flush=True)
    print(f"wrote {pearson_output}", flush=True)


if __name__ == "__main__":
    main()
