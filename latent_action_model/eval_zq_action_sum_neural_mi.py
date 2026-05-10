#!/usr/bin/env python3
"""Neural MI diagnostics between cached z_q features and summed action sequence.

This is intentionally a diagnostic tool, not a paper-grade estimator. It uses a
train/test split and reports held-out DV-MINE, NWJ, and InfoNCE-style bounds so
the numbers are less dominated by critic overfitting.
"""

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _global_scale(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (x - float(x.mean())) / (float(x.std()) + eps)


def _zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + eps)


def _random_project(x: np.ndarray, dim: int, seed: int) -> np.ndarray:
    if x.shape[1] <= dim:
        return x
    rng = np.random.default_rng(seed)
    proj = rng.normal(size=(x.shape[1], dim)).astype(np.float64) / math.sqrt(float(dim))
    return x @ proj


def _feature(data: Dict[str, np.ndarray], name: str) -> np.ndarray:
    z_raw = data["z_raw"].astype(np.float64)
    if name == "raw_global":
        return _global_scale(z_raw)
    if name == "zscore":
        return _zscore(z_raw)
    if name == "token_unit":
        return data["z_token_unit"].astype(np.float64)
    if name == "sample_unit":
        return z_raw / np.maximum(np.linalg.norm(z_raw, axis=1, keepdims=True), 1e-8)
    raise ValueError(name)


def _action_sum(data: Dict[str, np.ndarray]) -> np.ndarray:
    action_seq = data["action_seq"].astype(np.float64)
    if action_seq.shape[1] % 7 != 0:
        raise ValueError(f"Expected flattened action_seq dim divisible by 7, got {action_seq.shape}")
    horizon = action_seq.shape[1] // 7
    return action_seq.reshape(action_seq.shape[0], horizon, 7).sum(axis=1)


class Critic(nn.Module):
    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, y], dim=-1)).squeeze(-1)


def _bounds_bits(critic: Critic, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    critic.eval()
    with torch.no_grad():
        joint = critic(x, y)
        y_perm = y[torch.randperm(y.shape[0], device=y.device)]
        marg = critic(x, y_perm)
        mine = joint.mean() - torch.logsumexp(marg, dim=0) + math.log(float(marg.shape[0]))
        nwj = joint.mean() - torch.exp(marg - 1.0).mean()

        n = x.shape[0]
        x_rep = x[:, None, :].expand(n, n, x.shape[1]).reshape(n * n, x.shape[1])
        y_rep = y[None, :, :].expand(n, n, y.shape[1]).reshape(n * n, y.shape[1])
        scores = critic(x_rep, y_rep).reshape(n, n)
        labels = torch.arange(n, device=x.device)
        infonce = math.log(float(n)) - nn.functional.cross_entropy(scores, labels)

    inv_log2 = 1.0 / math.log(2.0)
    return {
        "mine_dv_bits": float(mine.item() * inv_log2),
        "nwj_bits": float(nwj.item() * inv_log2),
        "infonce_bits": float(infonce.item() * inv_log2),
    }


def _train_one(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    steps: int,
    batch_size: int,
    hidden_dim: int,
    lr: float,
    device: torch.device,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(x.shape[0])
    split = int(0.8 * x.shape[0])
    train_idx, test_idx = idx[:split], idx[split:]
    x_train = torch.as_tensor(x[train_idx], dtype=torch.float32)
    y_train = torch.as_tensor(y[train_idx], dtype=torch.float32)
    x_test = torch.as_tensor(x[test_idx], dtype=torch.float32, device=device)
    y_test = torch.as_tensor(y[test_idx], dtype=torch.float32, device=device)

    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True)
    critic = Critic(x.shape[1], y.shape[1], hidden_dim).to(device)
    opt = torch.optim.AdamW(critic.parameters(), lr=lr, weight_decay=1e-4)

    step = 0
    while step < steps:
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            y_perm = yb[torch.randperm(yb.shape[0], device=device)]
            joint = critic(xb, yb)
            marg = critic(xb, y_perm)
            dv = joint.mean() - torch.logsumexp(marg, dim=0) + math.log(float(marg.shape[0]))
            loss = -dv
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 5.0)
            opt.step()
            step += 1
            if step >= steps:
                break

    return _bounds_bits(critic, x_test, y_test)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument("--name", action="append", required=True)
    parser.add_argument("--features", default="raw_global,zscore")
    parser.add_argument("--dims", default="64,128,256")
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = args.name
    features = [x for x in args.features.split(",") if x]
    dims = [int(x) for x in args.dims.split(",") if x]
    seeds = [int(x) for x in args.seeds.split(",") if x]
    rows = []

    for name in names:
        data = dict(np.load(Path(args.feature_cache_dir) / f"{name}.npz"))
        y_all = _zscore(_action_sum(data))
        n = min(args.samples, y_all.shape[0])
        base_rng = np.random.default_rng(42)
        sample_idx = base_rng.choice(y_all.shape[0], size=n, replace=False)
        y = y_all[sample_idx]

        for feature in features:
            x_all = _feature(data, feature)[sample_idx]
            for dim in dims:
                x = _random_project(x_all, dim, seed=42)
                for seed in seeds:
                    torch.manual_seed(seed)
                    metrics = _train_one(
                        x=x,
                        y=y,
                        seed=seed,
                        steps=args.steps,
                        batch_size=args.batch_size,
                        hidden_dim=args.hidden_dim,
                        lr=args.lr,
                        device=device,
                    )
                    row = {
                        "name": name,
                        "feature": feature,
                        "x_dim": int(x.shape[1]),
                        "y_dim": int(y.shape[1]),
                        "samples": int(n),
                        "seed": int(seed),
                        "steps": int(args.steps),
                        "target": "action_seq_sum",
                        **metrics,
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
