import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr


def _global_scale(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (x - float(x.mean())) / (float(x.std()) + eps)


def _zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + eps)


def _sample_unit(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def _project(x: np.ndarray, dim: int, seed: int) -> np.ndarray:
    if dim <= 0 or x.shape[1] <= dim:
        return x
    rng = np.random.default_rng(seed)
    proj = rng.normal(size=(x.shape[1], dim)).astype(np.float64) / math.sqrt(float(dim))
    return x @ proj


def _feature(data: dict[str, np.ndarray], name: str) -> np.ndarray:
    z_raw = data["z_raw"].astype(np.float64)
    if name == "raw":
        return z_raw
    if name == "raw_global":
        return _global_scale(z_raw)
    if name == "zscore":
        return _zscore(z_raw)
    if name == "sample_unit":
        return _sample_unit(z_raw)
    if name == "token_unit":
        return data["z_token_unit"].astype(np.float64)
    if name == "token_norms":
        return _global_scale(data["z_token_norms"].astype(np.float64))
    if name == "unit_plus_norms":
        return np.concatenate(
            [
                data["z_token_unit"].astype(np.float64),
                _global_scale(data["z_token_norms"].astype(np.float64)),
            ],
            axis=1,
        )
    raise ValueError(f"unknown feature: {name}")


def _preprocess_y(y: np.ndarray, mode: str) -> np.ndarray:
    y = y.astype(np.float64)
    if mode == "raw":
        return y
    if mode == "global":
        return _global_scale(y)
    if mode == "zscore":
        return _zscore(y)
    raise ValueError(mode)


def _sample_pairs(dist_x: np.ndarray, dist_y: np.ndarray, pair_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if pair_samples <= 0 or pair_samples >= dist_x.shape[0]:
        return dist_x, dist_y
    rng = np.random.default_rng(seed)
    idx = rng.choice(dist_x.shape[0], size=pair_samples, replace=False)
    return dist_x[idx], dist_y[idx]


def distance_alignment_rows(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    feature: str,
    pair_samples: int,
    seed: int,
) -> list[dict]:
    rows = []
    for x_metric in ("euclidean", "cosine"):
        if x.shape[1] == 1 and x_metric == "cosine":
            continue
        dx = pdist(x, metric=x_metric)
        for y_metric in ("euclidean", "cosine"):
            dy = pdist(y, metric=y_metric)
            dx_s, dy_s = _sample_pairs(dx, dy, pair_samples, seed)
            sp = spearmanr(dx_s, dy_s).statistic
            pr = pearsonr(dx_s, dy_s).statistic
            rows.append(
                {
                    "metric_type": "distance_alignment",
                    "name": name,
                    "feature": feature,
                    "x_dim": int(x.shape[1]),
                    "y_dim": int(y.shape[1]),
                    "x_metric": x_metric,
                    "y_metric": y_metric,
                    "pair_samples": int(dx_s.shape[0]),
                    "spearman": float(sp),
                    "pearson": float(pr),
                }
            )
    return rows


class MineCritic(torch.nn.Module):
    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(x_dim + y_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, y], dim=-1)).squeeze(-1)


@torch.no_grad()
def _mine_eval_bits(
    critic: MineCritic,
    x: torch.Tensor,
    y: torch.Tensor,
    shuffles: int,
    generator: torch.Generator,
) -> float:
    joint = critic(x, y).mean()
    marginal_terms = []
    for _ in range(shuffles):
        perm = torch.randperm(y.shape[0], device=y.device, generator=generator)
        marginal_terms.append(torch.logsumexp(critic(x, y[perm]), dim=0) - math.log(y.shape[0]))
    marginal = torch.stack(marginal_terms).mean()
    return float(((joint - marginal) / math.log(2.0)).detach().cpu())


def mine_row(
    name: str,
    x_np: np.ndarray,
    y_np: np.ndarray,
    feature: str,
    seed: int,
    train_frac: float,
    steps: int,
    batch_size: int,
    hidden_dim: int,
    lr: float,
    eval_shuffles: int,
    device: torch.device,
) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(x_np.shape[0])
    n_train = max(2, int(round(x_np.shape[0] * train_frac)))
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]
    if val_idx.shape[0] < 2:
        val_idx = idx[-max(2, x_np.shape[0] // 5) :]
        train_idx = idx[: -val_idx.shape[0]]

    x_train = torch.as_tensor(x_np[train_idx], dtype=torch.float32, device=device)
    y_train = torch.as_tensor(y_np[train_idx], dtype=torch.float32, device=device)
    x_val = torch.as_tensor(x_np[val_idx], dtype=torch.float32, device=device)
    y_val = torch.as_tensor(y_np[val_idx], dtype=torch.float32, device=device)

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    critic = MineCritic(x_train.shape[1], y_train.shape[1], hidden_dim).to(device)
    opt = torch.optim.AdamW(critic.parameters(), lr=lr, weight_decay=1e-4)

    best_val = -float("inf")
    last_train = float("nan")
    for step in range(steps):
        batch_idx = torch.randint(0, x_train.shape[0], (batch_size,), device=device, generator=generator)
        xb = x_train[batch_idx]
        yb = y_train[batch_idx]
        perm = torch.randperm(batch_size, device=device, generator=generator)
        t_joint = critic(xb, yb).mean()
        t_marginal = torch.logsumexp(critic(xb, yb[perm]), dim=0) - math.log(batch_size)
        loss = -(t_joint - t_marginal)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 5.0)
        opt.step()
        last_train = float(((-loss) / math.log(2.0)).detach().cpu())
        if (step + 1) % max(1, steps // 10) == 0 or step == steps - 1:
            val_bits = _mine_eval_bits(critic, x_val, y_val, eval_shuffles, generator)
            best_val = max(best_val, val_bits)

    val_bits = _mine_eval_bits(critic, x_val, y_val, eval_shuffles, generator)
    return {
        "metric_type": "mine_dv",
        "name": name,
        "feature": feature,
        "x_dim": int(x_np.shape[1]),
        "y_dim": int(y_np.shape[1]),
        "samples": int(x_np.shape[0]),
        "train_samples": int(train_idx.shape[0]),
        "val_samples": int(val_idx.shape[0]),
        "seed": int(seed),
        "mine_steps": int(steps),
        "mine_batch_size": int(batch_size),
        "mine_hidden_dim": int(hidden_dim),
        "mine_lr": float(lr),
        "mine_train_last_bits": float(last_train),
        "mine_val_bits": float(val_bits),
        "mine_val_best_bits": float(best_val),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument("--name", action="append", required=True)
    parser.add_argument("--features", default="raw_global,zscore,token_unit,token_norms,unit_plus_norms")
    parser.add_argument("--metrics", default="distance,mine")
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--x-project-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preprocess-y", choices=["raw", "global", "zscore"], default="zscore")
    parser.add_argument("--pair-samples", type=int, default=500000)
    parser.add_argument("--mine-steps", type=int, default=800)
    parser.add_argument("--mine-batch-size", type=int, default=256)
    parser.add_argument("--mine-hidden-dim", type=int, default=256)
    parser.add_argument("--mine-lr", type=float, default=3e-4)
    parser.add_argument("--mine-eval-shuffles", type=int, default=8)
    parser.add_argument("--mine-device", default="auto")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    metrics = {x.strip() for x in args.metrics.split(",") if x.strip()}
    features = [x.strip() for x in args.features.split(",") if x.strip()]
    rows: list[dict] = []
    device = torch.device(
        "cuda"
        if args.mine_device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.mine_device == "auto" else args.mine_device)
    )

    rng = np.random.default_rng(args.seed)
    for name in args.name:
        data = dict(np.load(Path(args.feature_cache_dir) / f"{name}.npz"))
        y_all = _preprocess_y(data["action_seq"], args.preprocess_y)
        n = min(args.samples, y_all.shape[0])
        idx = rng.choice(y_all.shape[0], size=n, replace=False)
        y = y_all[idx]
        for feature in features:
            x_all = _feature(data, feature)
            x_base = x_all[idx].astype(np.float64)
            x = _project(x_base, args.x_project_dim, args.seed)
            if "distance" in metrics:
                rows.extend(distance_alignment_rows(name, x, y, feature, args.pair_samples, args.seed))
            if "mine" in metrics:
                row = mine_row(
                    name,
                    x,
                    y,
                    feature,
                    args.seed,
                    train_frac=0.8,
                    steps=args.mine_steps,
                    batch_size=args.mine_batch_size,
                    hidden_dim=args.mine_hidden_dim,
                    lr=args.mine_lr,
                    eval_shuffles=args.mine_eval_shuffles,
                    device=device,
                )
                rows.append(row)
                print(row, flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
