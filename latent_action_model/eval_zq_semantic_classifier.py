import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

try:
    import wandb

    if not hasattr(wandb, "init"):
        wandb.init = lambda *args, **kwargs: None
except Exception:
    pass


os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = Path(__file__).resolve().parents[1]
LATENT_ROOT = Path(__file__).resolve().parent
for path in (str(REPO_ROOT), str(LATENT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from eval_ksg_mi import _load_model  # noqa: E402
from eval_offset_norms import _to_device  # noqa: E402


def _load_cache(path: str) -> list[Dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    batches = payload["batches"] if isinstance(payload, dict) and "batches" in payload else payload
    if batches and isinstance(batches[0], tuple):
        batches = [batch for _, batch in batches]
    print(f"loaded cache={path} batches={len(batches)}", flush=True)
    return batches


def _all_zq(outputs: Dict, batch_size: int) -> torch.Tensor:
    z = outputs["z_q"].detach().float()
    if z.shape[0] == batch_size:
        return z.reshape(batch_size, -1)
    return z.reshape(batch_size, -1, *z.shape[1:]).reshape(batch_size, -1)


def _token_unit_zq(outputs: Dict, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    z = outputs["z_q"].detach().float()
    if z.shape[0] != batch_size:
        z = z.reshape(batch_size, -1, *z.shape[1:])
    tokens = z.reshape(batch_size, -1, z.shape[-1])
    norms = torch.linalg.vector_norm(tokens, dim=-1, keepdim=True).clamp_min(1e-8)
    return (tokens / norms).reshape(batch_size, -1), norms.squeeze(-1)


def _targets(batch: Dict, batch_size: int, num_bins: int) -> Dict[str, np.ndarray]:
    seq = batch["radprog_action_sequence"].detach().float().reshape(batch_size, -1, batch["radprog_action_sequence"].shape[-1])
    seq_sum = seq.sum(dim=1).cpu().numpy()
    seq_flat = seq.reshape(batch_size, -1).cpu().numpy()
    future = batch.get("radprog_future_action")
    if future is None:
        future_np = seq[:, -1].cpu().numpy()
    else:
        future_np = future.detach().float().reshape(batch_size, -1).cpu().numpy()

    def direction(x: np.ndarray) -> np.ndarray:
        dim = np.argmax(np.abs(x), axis=1)
        sign = (x[np.arange(x.shape[0]), dim] >= 0).astype(np.int64)
        return (dim * 2 + sign).astype(np.int64)

    def rank_bins(values: np.ndarray) -> np.ndarray:
        values = values.reshape(-1)
        order = np.argsort(values)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(values.shape[0])
        return np.minimum((ranks * num_bins) // values.shape[0], num_bins - 1).astype(np.int64)

    seq_dir = direction(seq_sum)
    future_dir = direction(future_np)
    seq_norm = rank_bins(np.linalg.norm(seq_sum, axis=1))
    flat_norm = rank_bins(np.linalg.norm(seq_flat, axis=1))
    return {
        "seq_sum_direction": seq_dir,
        "future_direction": future_dir,
        "seq_sum_action_bin": seq_norm * int(seq_sum.shape[1] * 2) + seq_dir,
        "seq_flat_action_bin": flat_norm * int(seq_sum.shape[1] * 2) + seq_dir,
        "seq_sum_norm_bin": seq_norm,
        "seq_flat_norm_bin": flat_norm,
    }


def _collect_xy(
    name: str,
    kind: str,
    config: str,
    ckpt: str,
    batches: list[Dict],
    device: torch.device,
    num_bins: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    model = _load_model(kind, config, ckpt, device)
    feats = {"raw": [], "sample_unit": [], "token_unit": [], "unit_plus_token_norm": []}
    labels: dict[str, list[np.ndarray]] = {}
    with torch.no_grad():
        for idx, batch_cpu in enumerate(batches, start=1):
            batch = _to_device(batch_cpu, device)
            outputs = model.lam.vq_encode(batch["videos"])
            batch_size = int(batch["videos"].shape[0])
            raw = _all_zq(outputs, batch_size)
            sample_norm = torch.linalg.vector_norm(raw, dim=1, keepdim=True).clamp_min(1e-8)
            token_unit, token_norm = _token_unit_zq(outputs, batch_size)
            feats["raw"].append(raw.cpu().numpy())
            feats["sample_unit"].append((raw / sample_norm).cpu().numpy())
            feats["token_unit"].append(token_unit.cpu().numpy())
            feats["unit_plus_token_norm"].append(
                torch.cat([token_unit, token_norm], dim=1).cpu().numpy()
            )
            for key, value in _targets(batch, batch_size, num_bins).items():
                labels.setdefault(key, []).append(value)
            print(f"eval name={name} batch={idx}/{len(batches)}", flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (
        {key: np.concatenate(value, axis=0).astype(np.float64) for key, value in feats.items()},
        {key: np.concatenate(value, axis=0).astype(np.int64) for key, value in labels.items()},
    )


def _split(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    a = int(n * 0.70)
    b = int(n * 0.85)
    return idx[:a], idx[a:b], idx[b:]


def _standardize(train: np.ndarray, eval_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + 1e-8
    return (train - mean) / std, (eval_x - mean) / std


def _ridge_weights(x: np.ndarray, y: np.ndarray, classes: int, alpha: float) -> np.ndarray:
    yy = np.zeros((x.shape[0], classes), dtype=np.float64)
    yy[np.arange(x.shape[0]), y] = 1.0
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    xtx = x_aug.T @ x_aug
    reg = np.eye(xtx.shape[0], dtype=x.dtype) * alpha
    reg[-1, -1] = 0.0
    return np.linalg.solve(xtx + reg, x_aug.T @ yy)


def _predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    return x_aug @ w


def _entropy(labels: np.ndarray, classes: int) -> float:
    counts = np.bincount(labels, minlength=classes).astype(np.float64)
    probs = counts[counts > 0] / labels.shape[0]
    return float(-(probs * np.log(probs + 1e-12)).sum())


def _ce_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    z = logits - logits.max(axis=1, keepdims=True)
    log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True) + 1e-12)
    return float(-log_probs[np.arange(labels.shape[0]), labels].mean())


def _balanced_acc(pred: np.ndarray, labels: np.ndarray, classes: int) -> float:
    vals = []
    for cls in range(classes):
        mask = labels == cls
        if mask.any():
            vals.append(float((pred[mask] == labels[mask]).mean()))
    return float(np.mean(vals)) if vals else 0.0


def _probe(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    standardize: bool,
    alphas: list[float],
) -> Dict[str, float]:
    classes = int(y.max()) + 1
    train_idx, val_idx, test_idx = _split(x.shape[0], seed)
    x_train_raw = x[train_idx]
    best_alpha = alphas[0]
    best_acc = -1.0
    best_w = None
    for alpha in alphas:
        x_train, x_val = _standardize(x_train_raw, x[val_idx]) if standardize else (x_train_raw, x[val_idx])
        w = _ridge_weights(x_train, y[train_idx], classes, alpha)
        val_logits = _predict(x_val, w)
        acc = float((val_logits.argmax(axis=1) == y[val_idx]).mean())
        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha
            best_w = w
    x_train, x_test = _standardize(x_train_raw, x[test_idx]) if standardize else (x_train_raw, x[test_idx])
    if best_w is None:
        best_w = _ridge_weights(x_train, y[train_idx], classes, best_alpha)
    logits = _predict(x_test, best_w)
    pred = logits.argmax(axis=1)
    ce = _ce_from_logits(logits, y[test_idx])
    h = _entropy(y[test_idx], classes)
    return {
        "classes": int(classes),
        "best_alpha": float(best_alpha),
        "val_acc": float(best_acc),
        "test_acc": float((pred == y[test_idx]).mean()),
        "test_balanced_acc": _balanced_acc(pred, y[test_idx], classes),
        "test_ce": ce,
        "test_entropy": h,
        "ce_info_gain": float(h - ce),
        "ce_info_gain_frac": float((h - ce) / max(h, 1e-12)),
    }


def _parse_model(spec: str) -> tuple[str, str, str, str]:
    parts = spec.split("|")
    if len(parts) != 4:
        raise ValueError("model spec: name|kind|config|ckpt")
    return parts[0], parts[1], parts[2], parts[3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-cache", required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--num-bins", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    parser.add_argument("--features", default="raw,sample_unit,token_unit,unit_plus_token_norm")
    parser.add_argument(
        "--labels",
        default="seq_sum_direction,future_direction,seq_sum_action_bin,seq_flat_action_bin,seq_sum_norm_bin,seq_flat_norm_bin",
    )
    parser.add_argument("--standardize", choices=["both", "false", "true"], default="both")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batches = _load_cache(args.batch_cache)
    alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
    rows = []
    feature_filter = {x for x in args.features.split(",") if x}
    label_filter = {x for x in args.labels.split(",") if x}
    std_values = [False, True] if args.standardize == "both" else [args.standardize == "true"]
    for spec in args.model:
        name, kind, config, ckpt = _parse_model(spec)
        feat_map, label_map = _collect_xy(name, kind, config, ckpt, batches, device, args.num_bins)
        for feat_name, x in feat_map.items():
            if feat_name not in feature_filter:
                continue
            for label_name, y in label_map.items():
                if label_name not in label_filter:
                    continue
                for std in std_values:
                    row = {
                        "name": name,
                        "kind": kind,
                        "feature": feat_name,
                        "label": label_name,
                        "standardize": bool(std),
                        "samples": int(x.shape[0]),
                        "feature_dim": int(x.shape[1]),
                    }
                    row.update(_probe(x, y, args.seed, std, alphas))
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
