#!/usr/bin/env python3
"""One-layer attentive probe for cached z_q tokens.

The probe is intentionally small:

    token features -> Linear -> 1 TransformerEncoderLayer -> mean pool -> Linear

It uses continuous z_q-derived token features, not token ids.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


MODELS = [
    "factorized_hyper_30k",
    "euclidean_visual_vq_30k",
    "univla_stage2_60k",
]


def split_indices(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    a = int(0.70 * n)
    b = int(0.85 * n)
    return idx[:a], idx[a:b], idx[b:]


def unit(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def target_dict(action_seq_flat: np.ndarray) -> dict[str, np.ndarray]:
    seq = action_seq_flat.reshape(action_seq_flat.shape[0], 10, 7).astype(np.float32)
    action_sum = seq.sum(axis=1)
    return {
        "action_sum": action_sum,
        "translation_sum": action_sum[:, :3],
        "rotation_sum": action_sum[:, 3:6],
        "full_seq": seq.reshape(seq.shape[0], 70),
    }


def token_features(name: str, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    z_raw = data["z_raw"].astype(np.float32).reshape(-1, 4, 128)
    z_dir = data["z_token_unit"].astype(np.float32).reshape(-1, 4, 128)
    z_radius = data["z_token_norms"].astype(np.float32)[..., None]

    if name == "factorized_hyper_30k":
        # Match how the factorized VLA sees the latent action:
        # one explicit radius token followed by four direction tokens.
        # The radius token stores the four radius scalars in the first four
        # coordinates and pads the rest with zeros so the token dimension
        # matches the 128-D direction tokens.
        radius_token = np.zeros((z_raw.shape[0], 1, 128), dtype=np.float32)
        radius_token[:, 0, :4] = data["z_token_norms"].astype(np.float32)
        radius_token_z = np.zeros((z_raw.shape[0], 1, 128), dtype=np.float32)
        radius_values = data["z_token_norms"].astype(np.float32)
        radius_token_z[:, 0, :4] = (radius_values - radius_values.mean()) / (radius_values.std() + 1e-6)
        return {
            "hyper_5tok_radius_dir": np.concatenate([radius_token, z_dir], axis=1),
            "hyper_5tok_zradius_dir": np.concatenate([radius_token_z, z_dir], axis=1),
            "raw_zq_tokens": z_raw,
        }
    return {
        "raw_zq_tokens": z_raw,
    }


def standardize_tokens(
    x: np.ndarray, train_idx: np.ndarray, eps: float = 1e-6
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x[train_idx].mean(axis=(0, 1), keepdims=True)
    std = x[train_idx].std(axis=(0, 1), keepdims=True) + eps
    return (x - mean) / std, mean, std


def standardize_target(
    y: np.ndarray, train_idx: np.ndarray, eps: float = 1e-6
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = y[train_idx].mean(axis=0, keepdims=True)
    std = y[train_idx].std(axis=0, keepdims=True) + eps
    return (y - mean) / std, mean, std


class OneLayerAttentiveProbe(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_tokens: int, d_model: int, heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, num_tokens, d_model))
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x) + self.pos
        h = self.layer(h)
        h = self.norm(h.mean(dim=1))
        return self.head(h)


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - y
    mse = float(np.mean(err * err))
    sse = float(np.sum(err * err))
    sst = float(np.sum((y - y.mean(axis=0, keepdims=True)) ** 2))
    denom = np.linalg.norm(y, axis=1) * np.linalg.norm(pred, axis=1)
    ok = denom > 1e-9
    cos = float(np.mean(np.sum(y[ok] * pred[ok], axis=1) / denom[ok])) if int(ok.sum()) else float("nan")
    return {"mse": mse, "r2": 1.0 - sse / max(sst, 1e-12), "cosine": cos}


def fit_probe(
    x: np.ndarray,
    y: np.ndarray,
    split: tuple[np.ndarray, np.ndarray, np.ndarray],
    seed: int,
    args: argparse.Namespace,
) -> dict[str, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr, va, te = split
    xz, _, _ = standardize_tokens(x, tr)
    yz, y_mean, y_std = standardize_target(y, tr)

    device = torch.device(args.device)
    x_t = torch.tensor(xz, dtype=torch.float32)
    y_t = torch.tensor(yz, dtype=torch.float32)
    train_ds = TensorDataset(x_t[tr], y_t[tr])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=generator)

    model = OneLayerAttentiveProbe(
        in_dim=x.shape[-1],
        out_dim=y.shape[-1],
        num_tokens=x.shape[1],
        d_model=args.d_model,
        heads=args.heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    best_val = math.inf
    best_state = None
    step = 0
    while step < args.steps:
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            step += 1
            if step % args.eval_every == 0 or step == args.steps:
                model.eval()
                with torch.no_grad():
                    val_pred = model(x_t[va].to(device)).cpu().numpy()
                    val_loss = float(np.mean((val_pred - yz[va]) ** 2))
                model.train()
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if step >= args.steps:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_z = model(x_t[te].to(device)).cpu().numpy()
    pred = pred_z * y_std + y_mean
    out = metrics(y[te], pred)
    out["val_mse_z"] = best_val
    return out


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["name"]), str(row["feature"]), str(row["target"]))
        grouped.setdefault(key, []).append(row)
    out = []
    for (name, feature, target), vals in grouped.items():
        item = {"name": name, "feature": feature, "target": target, "seeds": len(vals)}
        for metric in ("mse", "r2", "cosine", "val_mse_z"):
            arr = np.asarray([float(v[metric]) for v in vals], dtype=np.float64)
            item[f"{metric}_mean"] = float(arr.mean())
            item[f"{metric}_std"] = float(arr.std(ddof=0))
        out.append(item)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="outputs/analysis/cache/zq_full_action_k9_all_2048")
    parser.add_argument("--output", default="outputs/analysis/zq_attentive_probe_k9_2048.csv")
    parser.add_argument("--summary-output", default="outputs/analysis/zq_attentive_probe_k9_2048_summary.csv")
    parser.add_argument("--targets", default="action_sum,full_seq,translation_sum,rotation_sum")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    targets = [x.strip() for x in args.targets.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    cache = Path(args.cache_dir)
    rows = []

    for name in MODELS:
        data = dict(np.load(cache / f"{name}.npz"))
        all_targets = target_dict(data["action_seq"])
        features = token_features(name, data)
        for seed in seeds:
            split = split_indices(data["action_seq"].shape[0], seed)
            for target_name in targets:
                y = all_targets[target_name]
                for feature_name, x in features.items():
                    result = fit_probe(x, y, split, seed, args)
                    row = {
                        "name": name,
                        "feature": feature_name,
                        "target": target_name,
                        "seed": seed,
                        "samples": int(x.shape[0]),
                        "tokens": int(x.shape[1]),
                        "feature_dim": int(x.shape[2]),
                        "target_dim": int(y.shape[1]),
                        **result,
                    }
                    print(row, flush=True)
                    rows.append(row)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    sout = Path(args.summary_output)
    with sout.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print(f"wrote {out}")
    print(f"wrote {sout}")
    print("\nBest by target/metric:")
    for target in targets:
        cand = [r for r in summary if r["target"] == target]
        for metric in ("r2_mean", "cosine_mean"):
            best = max(cand, key=lambda r: float(r[metric]))
            print(target, metric, best)
        best_mse = min(cand, key=lambda r: float(r["mse_mean"]))
        print(target, "mse_mean", best_mse)


if __name__ == "__main__":
    main()
