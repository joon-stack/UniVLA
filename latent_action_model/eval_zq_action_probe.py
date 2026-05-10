import argparse
import csv
import os
import sys
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

from eval_ksg_mi import _load_datamodule, _load_model  # noqa: E402
from eval_offset_norms import _slice_batch, _to_device  # noqa: E402


def _collect_batches(data_config: str, samples: int, batch_size: int, respect_valid: bool) -> list[Dict]:
    dm = _load_datamodule(data_config, batch_size)
    cached: list[Dict] = []
    seen = 0
    for batch in dm.test_dataloader():
        remaining = samples - seen
        if remaining <= 0:
            break
        mask = torch.ones(batch["videos"].shape[0], dtype=torch.bool)
        if respect_valid and "radprog_valid" in batch:
            mask = batch["radprog_valid"].reshape(-1).float() > 0.5
        if int(mask.sum()) == 0:
            continue
        sliced = _slice_batch(batch, mask, remaining)
        n = int(sliced["videos"].shape[0])
        cached.append(sliced)
        seen += n
        print(f"cached samples={seen}/{samples}", flush=True)
        if seen >= samples:
            break
    if seen < samples:
        print(f"[warn] only cached {seen} samples", flush=True)
    return cached


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


def _rechunk_batches(cached: list[Dict], chunk_size: int) -> list[Dict]:
    rechunked: list[Dict] = []
    for batch in cached:
        rechunked.extend(_rechunk_batch(batch, chunk_size))
    return rechunked


def _load_or_collect_batches(
    data_config: str,
    samples: int,
    eval_batch_size: int,
    collect_batch_size: int,
    respect_valid: bool,
    batch_cache: str,
) -> list[Dict]:
    if batch_cache:
        cache_path = Path(batch_cache)
        if cache_path.exists():
            payload = torch.load(cache_path, map_location="cpu")
            cached = payload["batches"] if isinstance(payload, dict) and "batches" in payload else payload
            cached = _rechunk_batches(cached, eval_batch_size)
            print(f"loaded batch_cache={cache_path} cached_batches={len(cached)}", flush=True)
            return cached

    cached = _collect_batches(data_config, samples, collect_batch_size, respect_valid)
    cached = _rechunk_batches(cached, eval_batch_size)
    if batch_cache:
        cache_path = Path(batch_cache)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "data_config": data_config,
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


def _future_zq(outputs: Dict, batch_size: int) -> torch.Tensor:
    z = outputs["z_q"].detach().float()
    if z.shape[0] == batch_size and z.ndim >= 4:
        z = z[:, -1]
    elif z.shape[0] != batch_size:
        z = z.reshape(batch_size, -1, *z.shape[1:])[:, -1]
    return z.reshape(batch_size, -1)


def _target(batch: Dict, name: str, batch_size: int) -> torch.Tensor:
    if name == "action_sequence" and "radprog_action_sequence" in batch:
        return batch["radprog_action_sequence"].detach().float().reshape(batch_size, -1)
    if name == "future_action" and "radprog_future_action" in batch:
        return batch["radprog_future_action"].detach().float().reshape(batch_size, -1)
    if name == "action":
        return batch["action"].detach().float().reshape(batch_size, -1)
    raise ValueError(f"target={name} is not available in this batch")


def _collect_xy(
    name: str,
    kind: str,
    config: str,
    ckpt: str,
    batches: list[Dict],
    device: torch.device,
    target_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    model = _load_model(kind, config, ckpt, device)
    xs = []
    ys = []
    with torch.no_grad():
        for batch_idx, batch_cpu in enumerate(batches, start=1):
            batch = _to_device(batch_cpu, device)
            outputs = model.lam.vq_encode(batch["videos"])
            batch_size = int(batch["videos"].shape[0])
            xs.append(_future_zq(outputs, batch_size).cpu())
            ys.append(_target(batch, target_name, batch_size).cpu())
            print(f"eval name={name} target={target_name} batch={batch_idx}/{len(batches)}", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.cat(xs, dim=0).numpy(), torch.cat(ys, dim=0).numpy()


def _split_indices(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    return idx[:train_end], idx[train_end:val_end], idx[val_end:]


def _prepare_features(
    x_train: np.ndarray,
    x_eval: np.ndarray,
    preprocess: str,
) -> tuple[np.ndarray, np.ndarray]:
    x_train = x_train.astype(np.float64, copy=False)
    x_eval = x_eval.astype(np.float64, copy=False)
    if preprocess == "raw":
        return x_train, x_eval
    if preprocess == "zscore":
        mean = x_train.mean(axis=0, keepdims=True)
        std = x_train.std(axis=0, keepdims=True) + 1e-8
        return (x_train - mean) / std, (x_eval - mean) / std
    raise ValueError(f"unknown preprocess: {preprocess}")


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    xtx = x_aug.T @ x_aug
    reg = np.eye(xtx.shape[0], dtype=x.dtype) * alpha
    reg[-1, -1] = 0.0
    rhs = x_aug.T @ y
    try:
        return np.linalg.solve(xtx + reg, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(xtx + reg) @ rhs


def _predict(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    return x_aug @ weights


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    y_true = y_true.astype(np.float64, copy=False)
    y_pred = y_pred.astype(np.float64, copy=False)
    sse = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2))
    total = 1.0 - sse / max(sst, 1e-12)
    per_dim_sse = np.sum((y_true - y_pred) ** 2, axis=0)
    per_dim_sst = np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2, axis=0)
    per_dim = 1.0 - per_dim_sse / np.maximum(per_dim_sst, 1e-12)
    return float(total), float(np.mean(per_dim))


def _cosine(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.linalg.norm(y_true, axis=1) * np.linalg.norm(y_pred, axis=1)
    ok = denom > 1e-12
    if int(ok.sum()) == 0:
        return float("nan")
    return float(np.mean(np.sum(y_true[ok] * y_pred[ok], axis=1) / denom[ok]))


def _probe_target(
    x: np.ndarray,
    y: np.ndarray,
    indices: tuple[np.ndarray, np.ndarray, np.ndarray],
    preprocess: str,
    alphas: list[float],
) -> Dict:
    train_idx, val_idx, test_idx = indices
    x_train_raw = x[train_idx]
    y_train = y[train_idx].astype(np.float64, copy=False)

    best_alpha = alphas[0]
    best_val_r2 = -float("inf")
    best_weights = None
    for alpha in alphas:
        x_train, x_val = _prepare_features(x_train_raw, x[val_idx], preprocess)
        weights = _fit_ridge(x_train, y_train, alpha)
        pred_val = _predict(x_val, weights)
        val_r2, _ = _r2(y[val_idx], pred_val)
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_alpha = alpha
            best_weights = weights

    x_train, x_test = _prepare_features(x_train_raw, x[test_idx], preprocess)
    if best_weights is None:
        best_weights = _fit_ridge(x_train, y_train, best_alpha)
    pred = _predict(x_test, best_weights)
    test_r2, test_r2_mean_dim = _r2(y[test_idx], pred)
    return {
        "preprocess": preprocess,
        "best_alpha": float(best_alpha),
        "val_r2": float(best_val_r2),
        "test_r2": test_r2,
        "test_r2_mean_dim": test_r2_mean_dim,
        "test_mse": float(np.mean((y[test_idx] - pred) ** 2)),
        "test_cosine": _cosine(y[test_idx], pred),
    }


def _evaluate_probe(
    name: str,
    kind: str,
    x: np.ndarray,
    y: np.ndarray,
    target_name: str,
    seed: int,
    alphas: list[float],
) -> list[Dict]:
    indices = _split_indices(x.shape[0], seed)
    y_norm = np.linalg.norm(y, axis=1, keepdims=True)
    z_norm = np.linalg.norm(x, axis=1)
    target_norm = y_norm.reshape(-1)
    z_target_norm_pearson = float(np.corrcoef(z_norm, target_norm)[0, 1]) if np.std(z_norm) > 1e-12 else float("nan")

    rows = []
    for preprocess in ("raw", "zscore"):
        full = _probe_target(x, y, indices, preprocess, alphas)
        full.update(
            name=name,
            kind=kind,
            target=target_name,
            probe_target="full",
            samples=int(x.shape[0]),
            latent_dim=int(x.shape[1]),
            target_dim=int(y.shape[1]),
            z_norm_target_norm_pearson=z_target_norm_pearson,
        )
        rows.append(full)

        norm = _probe_target(x, y_norm, indices, preprocess, alphas)
        norm.update(
            name=name,
            kind=kind,
            target=target_name,
            probe_target="norm",
            samples=int(x.shape[0]),
            latent_dim=int(x.shape[1]),
            target_dim=1,
            z_norm_target_norm_pearson=z_target_norm_pearson,
        )
        rows.append(norm)
    return rows


def _parse_model_spec(spec: str) -> tuple[str, str, str, str]:
    parts = spec.split("|")
    if len(parts) != 4:
        raise ValueError("model spec must be name|kind|config|ckpt")
    return parts[0], parts[1], parts[2], parts[3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", default=[], help="name|kind|config|ckpt")
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--target", choices=["action_sequence", "future_action", "action"], default="action_sequence")
    parser.add_argument("--respect-valid", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-cache", default="")
    parser.add_argument("--cache-collect-batch-size", type=int, default=0)
    parser.add_argument("--cache-only", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} samples={args.samples} target={args.target}", flush=True)
    batches = _load_or_collect_batches(
        args.data_config,
        args.samples,
        args.batch_size,
        args.cache_collect_batch_size or args.batch_size,
        args.respect_valid,
        args.batch_cache,
    )
    if args.cache_only:
        return
    if not args.model:
        raise ValueError("--model is required unless --cache-only is set")

    rows = []
    for spec in args.model:
        name, kind, config, ckpt = _parse_model_spec(spec)
        x, y = _collect_xy(name, kind, config, ckpt, batches, device, args.target)
        model_rows = _evaluate_probe(name, kind, x, y, args.target, args.seed, alphas)
        rows.extend(model_rows)
        for row in model_rows:
            print(row, flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
