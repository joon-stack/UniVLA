import argparse
import csv
import math
import os
import sys
from collections import Counter, defaultdict
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
from eval_offset_norms import _collect_offset_batches, _to_device  # noqa: E402


def _rechunk_batch(batch: Dict, chunk_size: int) -> list[Dict]:
    if chunk_size <= 0:
        return [batch]
    batch_size = int(batch["videos"].shape[0])
    if batch_size <= chunk_size:
        return [batch]
    chunks = []
    for start in range(0, batch_size, chunk_size):
        end = min(start + chunk_size, batch_size)
        indices = torch.arange(start, end)
        chunk = {}
        for key, value in batch.items():
            if torch.is_tensor(value) and value.shape[:1] == (batch_size,):
                chunk[key] = value.index_select(0, indices)
            elif isinstance(value, np.ndarray) and value.shape[:1] == (batch_size,):
                chunk[key] = value[start:end]
            elif isinstance(value, list) and len(value) == batch_size:
                chunk[key] = value[start:end]
            elif isinstance(value, tuple) and len(value) == batch_size:
                chunk[key] = tuple(value[start:end])
            else:
                chunk[key] = value
        chunks.append(chunk)
    return chunks


def _rechunk_offset_batches(cached: list[tuple[int, Dict]], chunk_size: int) -> list[tuple[int, Dict]]:
    rechunked: list[tuple[int, Dict]] = []
    for offset, batch in cached:
        for chunk in _rechunk_batch(batch, chunk_size):
            rechunked.append((offset, chunk))
    return rechunked


def _load_or_collect_batches(
    data_config: str,
    offset: int,
    samples: int,
    eval_batch_size: int,
    collect_batch_size: int,
    batch_cache: str,
    respect_valid: bool,
) -> list[tuple[int, Dict]]:
    if batch_cache:
        cache_path = Path(batch_cache)
        if cache_path.exists():
            payload = torch.load(cache_path, map_location="cpu")
            cached = payload["batches"] if isinstance(payload, dict) and "batches" in payload else payload
            cached = _rechunk_offset_batches(cached, eval_batch_size)
            print(f"loaded batch_cache={cache_path} cached_batches={len(cached)}", flush=True)
            return cached

    cached = _collect_offset_batches(
        data_config,
        offsets=[offset],
        samples_per_offset=samples,
        batch_size=collect_batch_size,
        respect_valid=respect_valid,
    )
    cached = _rechunk_offset_batches(cached, eval_batch_size)
    if batch_cache:
        cache_path = Path(batch_cache)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "data_config": data_config,
                "offset": int(offset),
                "samples": int(samples),
                "eval_batch_size": int(eval_batch_size),
                "collect_batch_size": int(collect_batch_size),
                "respect_valid": bool(respect_valid),
                "batches": cached,
            },
            cache_path,
        )
        print(f"wrote batch_cache={cache_path} cached_batches={len(cached)}", flush=True)
    return cached


def _entropy(labels: np.ndarray, minlength: int | None = None) -> float:
    labels = labels.reshape(-1).astype(np.int64)
    if labels.size == 0:
        return 0.0
    if minlength is None:
        counts = np.asarray(list(Counter(labels.tolist()).values()), dtype=np.float64)
    else:
        counts = np.bincount(labels, minlength=minlength).astype(np.float64)
    counts = counts[counts > 0]
    probs = counts / counts.sum()
    return float(-(probs * np.log(probs)).sum())


def _mi_nmi(x: np.ndarray, y: np.ndarray, num_x: int, num_y: int) -> tuple[float, float]:
    x = x.reshape(-1).astype(np.int64)
    y = y.reshape(-1).astype(np.int64)
    if x.size == 0 or x.size != y.size or num_x <= 1 or num_y <= 1:
        return 0.0, 0.0
    x = np.clip(x, 0, num_x - 1)
    y = np.clip(y, 0, num_y - 1)
    joint = np.bincount(x * num_y + y, minlength=num_x * num_y).reshape(num_x, num_y).astype(np.float64)
    total = joint.sum()
    if total <= 0:
        return 0.0, 0.0
    pxy = joint / total
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    nz = pxy > 0
    expected = np.maximum(px[:, None] * py[None, :], 1e-12)
    mi = float((pxy[nz] * (np.log(pxy[nz]) - np.log(expected[nz]))).sum())
    hx = _entropy(x, minlength=num_x)
    hy = _entropy(y, minlength=num_y)
    denom = math.sqrt(max(hx * hy, 1e-12))
    return mi, float(mi / denom) if hx > 0 and hy > 0 else 0.0


def _rank_bins(values: np.ndarray, bins: int) -> tuple[np.ndarray, int]:
    values = values.reshape(-1).astype(np.float64)
    if values.size <= 1 or float(values.var()) <= 1e-12:
        return np.zeros(values.shape[0], dtype=np.int64), 1
    bins = max(1, min(int(bins), int(values.size)))
    order = np.argsort(values)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(values.size)
    return np.minimum((ranks * bins) // values.size, bins - 1).astype(np.int64), bins


def _action_targets(action: np.ndarray, num_bins: int) -> Dict[str, np.ndarray | int]:
    action_norm = np.linalg.norm(action, axis=1)
    norm_bins, num_norm_bins = _rank_bins(action_norm, num_bins)
    dominant_dim = np.argmax(np.abs(action), axis=1)
    dominant_value = action[np.arange(action.shape[0]), dominant_dim]
    direction_bins = dominant_dim * 2 + (dominant_value >= 0).astype(np.int64)
    num_direction_bins = max(1, action.shape[1] * 2)
    action_bins = norm_bins * num_direction_bins + direction_bins
    return {
        "action_norm_bins": norm_bins,
        "num_action_norm_bins": num_norm_bins,
        "action_direction_bins": direction_bins,
        "num_action_direction_bins": num_direction_bins,
        "action_bins": action_bins.astype(np.int64),
        "num_action_bins": num_norm_bins * num_direction_bins,
    }


def _state_targets(flat: np.ndarray, num_bins: int) -> Dict[str, np.ndarray | int]:
    norm = np.linalg.norm(flat, axis=1)
    bins, num = _rank_bins(norm, num_bins)
    return {"dino_norm_bins": bins, "num_dino_norm_bins": num}


def _future_indices(outputs: Dict, batch_size: int, factorized_direction_only: bool) -> torch.Tensor:
    key = "direction_indices" if factorized_direction_only else "indices"
    indices = outputs[key].detach().long()
    if indices.shape[0] < batch_size:
        raise ValueError(f"{key} has too few rows: {tuple(indices.shape)} for batch_size={batch_size}")
    return indices.reshape(indices.shape[0], -1)[-batch_size:]


def _tuple_id(tokens: np.ndarray, base: int = 16) -> np.ndarray:
    tokens = tokens.astype(np.int64)
    ids = np.zeros(tokens.shape[0], dtype=np.int64)
    for slot in range(tokens.shape[1]):
        ids = ids * base + tokens[:, slot]
    return ids


def _within_variance(labels: np.ndarray, target: np.ndarray) -> tuple[float, float, int]:
    groups: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels.reshape(-1).tolist()):
        groups[int(label)].append(idx)
    weighted = 0.0
    unweighted = []
    used = 0
    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        values = target[idxs]
        var = float(np.mean(np.var(values, axis=0)))
        weighted += var * len(idxs)
        unweighted.append(var)
        used += len(idxs)
    if used == 0:
        return float("nan"), float("nan"), 0
    return weighted / used, float(np.mean(unweighted)), used


def _evaluate_model(
    name: str,
    kind: str,
    config: str,
    ckpt: str,
    cached_batches: list[tuple[int, Dict]],
    device: torch.device,
    num_bins: int,
    factorized_direction_only: bool,
) -> Dict:
    model = _load_model(kind, config, ckpt, device)
    token_chunks = []
    action_target_chunks = []
    dino_target_chunks = []

    with torch.no_grad():
        for batch_idx, (_, batch_cpu) in enumerate(cached_batches, start=1):
            batch = _to_device(batch_cpu, device)
            outputs = model.lam.vq_encode(batch["videos"])
            batch_size = int(batch["videos"].shape[0])
            tokens = _future_indices(outputs, batch_size, factorized_direction_only)
            if tokens.shape[1] != 4:
                raise ValueError(f"{name}: expected 4 tokens, got shape {tuple(tokens.shape)}")
            token_chunks.append(tokens.cpu().numpy())
            action_target_chunks.append(
                batch["radprog_action_sequence"].detach().float().cpu().numpy().reshape(batch_size, -1)
            )
            patches = outputs["patches"].detach().float()
            change = patches[:, -1] - patches[:, 0]
            dino_target_chunks.append(change.reshape(batch_size, -1).cpu().numpy())
            print(f"eval name={name} batch={batch_idx}/{len(cached_batches)}", flush=True)

    tokens = np.concatenate(token_chunks, axis=0)
    tuple_ids = _tuple_id(tokens, base=16)
    flat_tokens = tokens.reshape(-1)
    num_tuple = 16 ** tokens.shape[1]
    action_flat = np.concatenate(action_target_chunks, axis=0)
    dino_flat = np.concatenate(dino_target_chunks, axis=0)
    action_info = _action_targets(action_flat, num_bins)
    state_info = _state_targets(dino_flat, num_bins)
    action_bins = action_info["action_bins"].astype(np.int64)
    action_norm_bins = action_info["action_norm_bins"].astype(np.int64)
    action_direction_bins = action_info["action_direction_bins"].astype(np.int64)
    dino_norm_bins = state_info["dino_norm_bins"].astype(np.int64)

    num_action_bins = max(int(action_info["num_action_bins"]), int(action_bins.max()) + 1)
    num_action_norm_bins = max(int(action_info["num_action_norm_bins"]), int(action_norm_bins.max()) + 1)
    num_action_direction_bins = max(int(action_info["num_action_direction_bins"]), int(action_direction_bins.max()) + 1)
    num_dino_norm_bins = max(int(state_info["num_dino_norm_bins"]), int(dino_norm_bins.max()) + 1)

    tuple_action_mi, tuple_action_nmi = _mi_nmi(tuple_ids, action_bins, num_tuple, num_action_bins)
    tuple_action_norm_mi, tuple_action_norm_nmi = _mi_nmi(
        tuple_ids,
        action_norm_bins,
        num_tuple,
        num_action_norm_bins,
    )
    tuple_action_direction_mi, tuple_action_direction_nmi = _mi_nmi(
        tuple_ids,
        action_direction_bins,
        num_tuple,
        num_action_direction_bins,
    )
    tuple_dino_mi, tuple_dino_nmi = _mi_nmi(tuple_ids, dino_norm_bins, num_tuple, num_dino_norm_bins)

    flat_action_mi, flat_action_nmi = _mi_nmi(
        flat_tokens,
        np.repeat(action_bins, tokens.shape[1]),
        16,
        num_action_bins,
    )
    flat_action_norm_mi, flat_action_norm_nmi = _mi_nmi(
        flat_tokens,
        np.repeat(action_norm_bins, tokens.shape[1]),
        16,
        num_action_norm_bins,
    )
    flat_action_direction_mi, flat_action_direction_nmi = _mi_nmi(
        flat_tokens,
        np.repeat(action_direction_bins, tokens.shape[1]),
        16,
        num_action_direction_bins,
    )
    flat_dino_mi, flat_dino_nmi = _mi_nmi(
        flat_tokens,
        np.repeat(dino_norm_bins, tokens.shape[1]),
        16,
        num_dino_norm_bins,
    )

    within_action_weighted, within_action_unweighted, within_action_used = _within_variance(tuple_ids, action_flat)
    within_dino_weighted, within_dino_unweighted, within_dino_used = _within_variance(tuple_ids, dino_flat)

    result = {
        "name": name,
        "kind": kind,
        "token_view": "direction4" if factorized_direction_only else "code4",
        "samples": int(tokens.shape[0]),
        "possible_tuple": int(num_tuple),
        "active_tuple": int(np.unique(tuple_ids).size),
        "tuple_entropy": _entropy(tuple_ids, minlength=num_tuple),
        "tuple_perplexity": math.exp(_entropy(tuple_ids, minlength=num_tuple)),
        "flat_token_entropy": _entropy(flat_tokens, minlength=16),
        "flat_token_perplexity": math.exp(_entropy(flat_tokens, minlength=16)),
        "mi_tuple_action_bin": tuple_action_mi,
        "nmi_tuple_action_bin": tuple_action_nmi,
        "mi_tuple_action_norm": tuple_action_norm_mi,
        "nmi_tuple_action_norm": tuple_action_norm_nmi,
        "mi_tuple_action_direction": tuple_action_direction_mi,
        "nmi_tuple_action_direction": tuple_action_direction_nmi,
        "mi_tuple_dino_norm": tuple_dino_mi,
        "nmi_tuple_dino_norm": tuple_dino_nmi,
        "mi_flat_token_action_bin": flat_action_mi,
        "nmi_flat_token_action_bin": flat_action_nmi,
        "mi_flat_token_action_norm": flat_action_norm_mi,
        "nmi_flat_token_action_norm": flat_action_norm_nmi,
        "mi_flat_token_action_direction": flat_action_direction_mi,
        "nmi_flat_token_action_direction": flat_action_direction_nmi,
        "flat_token_direction_minus_norm_nmi": flat_action_direction_nmi - flat_action_norm_nmi,
        "mi_flat_token_dino_norm": flat_dino_mi,
        "nmi_flat_token_dino_norm": flat_dino_nmi,
        "tuple_direction_minus_norm_nmi": tuple_action_direction_nmi - tuple_action_norm_nmi,
        "within_tuple_action_var_weighted": within_action_weighted,
        "within_tuple_action_var_unweighted": within_action_unweighted,
        "within_tuple_action_samples_used": within_action_used,
        "within_tuple_dino_var_weighted": within_dino_weighted,
        "within_tuple_dino_var_unweighted": within_dino_unweighted,
        "within_tuple_dino_samples_used": within_dino_used,
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _parse_model_spec(spec: str) -> tuple[str, str, str, str, bool]:
    parts = spec.split("|")
    if len(parts) != 5:
        raise ValueError("model spec must be name|kind|config|ckpt|direction_only")
    return parts[0], parts[1], parts[2], parts[3], parts[4].lower() in {"1", "true", "yes"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", default=[], help="name|kind|config|ckpt|direction_only")
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-bins", type=int, default=16)
    parser.add_argument("--offset", type=int, default=9)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-cache", default="")
    parser.add_argument("--cache-collect-batch-size", type=int, default=0)
    parser.add_argument("--ignore-valid", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} offset={args.offset} samples={args.samples}", flush=True)
    cached_batches = _load_or_collect_batches(
        args.data_config,
        args.offset,
        args.samples,
        args.batch_size,
        args.cache_collect_batch_size or args.batch_size,
        args.batch_cache,
        not args.ignore_valid,
    )
    print(f"cached_batches={len(cached_batches)}", flush=True)
    if args.cache_only:
        return
    if not args.model:
        raise ValueError("--model is required unless --cache-only is set")

    rows = []
    for spec in args.model:
        name, kind, config, ckpt, direction_only = _parse_model_spec(spec)
        rows.append(
            _evaluate_model(
                name,
                kind,
                config,
                ckpt,
                cached_batches,
                device,
                args.num_bins,
                direction_only,
            )
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output}", flush=True)
    for row in rows:
        print(row, flush=True)


if __name__ == "__main__":
    main()
