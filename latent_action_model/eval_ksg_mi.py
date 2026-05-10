import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
import yaml
from scipy.special import digamma
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[1]
LATENT_ROOT = Path(__file__).resolve().parent
for path in (str(REPO_ROOT), str(LATENT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import tensorflow as tf  # noqa: E402

tf.config.set_visible_devices([], "GPU")

from genie.dataset import LightningOpenX  # noqa: E402
from genie.model import DINO_LAM  # noqa: E402
from genie.model_visual_vq import VisualVQ_DINO_LAM  # noqa: E402


def _load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _class_kwargs(cls, cfg: Dict) -> Dict:
    import inspect

    params = inspect.signature(cls.__init__).parameters
    return {k: v for k, v in cfg.items() if k in params}


def _load_model(kind: str, config_path: str, ckpt_path: str, device: torch.device) -> torch.nn.Module:
    cfg = _load_yaml(config_path)
    model_cfg = cfg.get("model", {})
    if kind == "visual_vq":
        model = VisualVQ_DINO_LAM(**_class_kwargs(VisualVQ_DINO_LAM, model_cfg))
    elif kind == "stage2":
        model = DINO_LAM(**_class_kwargs(DINO_LAM, model_cfg))
    else:
        raise ValueError(f"unknown kind: {kind}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)}", flush=True)
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)}", flush=True)
    return model.to(device).eval()


def _load_datamodule(config_path: str, batch_size: int) -> LightningOpenX:
    cfg = _load_yaml(config_path)
    data_cfg = dict(cfg.get("data", {}))
    data_cfg["batch_size"] = batch_size
    data_cfg["val_batch_size"] = batch_size
    dm = LightningOpenX(**_class_kwargs(LightningOpenX, data_cfg))
    dm.batch_size = batch_size
    dm.val_batch_size = batch_size
    dm.setup("test")
    return dm


def _to_device(batch: Dict, device: torch.device) -> Dict:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def _future_zq(outputs: Dict, batch_size: int) -> torch.Tensor:
    z = outputs["z_q"]
    if z.shape[0] == batch_size and z.ndim >= 4:
        z = z[:, -1]
    elif z.shape[0] != batch_size:
        z = z.reshape(batch_size, -1, *z.shape[1:])[:, -1]
    return z.detach().float().reshape(batch_size, -1)


def _target_action(batch: Dict, target: str, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if target == "action":
        action = batch["action"].detach().float().reshape(batch_size, -1)
        mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        return action, mask
    if target == "future_action":
        if "radprog_future_action" not in batch:
            raise ValueError("target=future_action requires batch['radprog_future_action'].")
        action = batch["radprog_future_action"].detach().float().reshape(batch_size, -1)
        if "radprog_valid" in batch:
            mask = batch["radprog_valid"].detach().to(device=device).reshape(batch_size) > 0.5
        else:
            mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        return action, mask
    if target == "action_sequence":
        if "radprog_action_sequence" not in batch:
            raise ValueError("target=action_sequence requires batch['radprog_action_sequence'].")
        action = batch["radprog_action_sequence"].detach().float().reshape(batch_size, -1)
        if "radprog_valid" in batch:
            mask = batch["radprog_valid"].detach().to(device=device).reshape(batch_size) > 0.5
        else:
            mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        return action, mask
    raise ValueError(f"unknown target: {target}")


def _collect_pairs(
    model: torch.nn.Module,
    dataloader: Iterable,
    device: torch.device,
    max_samples: int,
    target: str,
) -> tuple[np.ndarray, np.ndarray]:
    zs = []
    actions = []
    seen = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = _to_device(batch, device)
            outputs = model.lam(batch) if hasattr(model, "lam") else model(batch)
            batch_size = batch["action"].shape[0]
            z = _future_zq(outputs, batch_size)
            action, mask = _target_action(batch, target, batch_size, device)
            z = z[mask]
            action = action[mask]
            if z.shape[0] == 0:
                print(f"collect batch={batch_idx + 1} samples={seen}", flush=True)
                continue
            zs.append(z.cpu())
            actions.append(action.cpu())
            seen += z.shape[0]
            print(f"collect batch={batch_idx + 1} samples={seen}", flush=True)
            if seen >= max_samples:
                break

    z_np = torch.cat(zs, dim=0)[:max_samples].numpy()
    a_np = torch.cat(actions, dim=0)[:max_samples].numpy()
    return z_np, a_np


def _standardize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    return (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + eps)


def _global_scale(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    return (x - float(x.mean())) / (float(x.std()) + eps)


def _random_project(x: np.ndarray, out_dim: int, seed: int) -> np.ndarray:
    if x.shape[1] <= out_dim:
        return x
    rng = np.random.default_rng(seed)
    projection = rng.normal(size=(x.shape[1], out_dim)).astype(np.float64)
    projection /= math.sqrt(float(out_dim))
    return x @ projection


def _prepare_latent(
    z_raw: np.ndarray,
    preprocess: str,
    project_dim: int,
    seed: int,
) -> tuple[np.ndarray, Dict]:
    z_raw = z_raw.astype(np.float64, copy=False)
    raw_norm = np.linalg.norm(z_raw, axis=1, keepdims=True)
    info = {
        "latent_dim_raw": int(z_raw.shape[1]),
        "latent_raw_norm_mean": float(raw_norm.mean()),
        "latent_raw_norm_std": float(raw_norm.std()),
    }

    if preprocess == "raw":
        z = z_raw
        info.update(
            latent_preprocess="raw",
            latent_projected=False,
            latent_dim_after_preprocess=int(z.shape[1]),
        )
        return z, info

    if preprocess == "global_scale":
        z = _global_scale(z_raw)
        info.update(
            latent_preprocess="global_scale",
            latent_projected=False,
            latent_dim_after_preprocess=int(z.shape[1]),
        )
        return z, info

    if preprocess == "norm_augmented":
        z = _standardize(z_raw)
        z = _random_project(z, project_dim, seed)
        z = _standardize(z)
        norm_feature = _global_scale(raw_norm)
        z = np.concatenate([z, norm_feature], axis=1)
        info.update(
            latent_preprocess="norm_augmented",
            latent_projected=z_raw.shape[1] > project_dim,
            latent_dim_after_preprocess=int(z.shape[1]),
        )
        return z, info

    if preprocess == "standard":
        z = _standardize(z_raw)
        z = _random_project(z, project_dim, seed)
        z = _standardize(z)
        info.update(
            latent_preprocess="standard",
            latent_projected=z_raw.shape[1] > project_dim,
            latent_dim_after_preprocess=int(z.shape[1]),
        )
        return z, info

    raise ValueError(f"unknown latent preprocess: {preprocess}")


def ksg_mi_bits(x: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"x/y sample mismatch: {x.shape[0]} != {y.shape[0]}")
    n = x.shape[0]
    if n <= k:
        raise ValueError(f"need n > k, got n={n}, k={k}")

    xy = np.concatenate([x, y], axis=1)
    tree_xy = cKDTree(xy)
    tree_x = cKDTree(x)
    tree_y = cKDTree(y)
    distances, _ = tree_xy.query(xy, k=k + 1, p=np.inf, workers=-1)
    eps = np.nextafter(distances[:, k], 0.0)

    nx = np.empty(n, dtype=np.int64)
    ny = np.empty(n, dtype=np.int64)
    for i in range(n):
        nx[i] = len(tree_x.query_ball_point(x[i], eps[i], p=np.inf)) - 1
        ny[i] = len(tree_y.query_ball_point(y[i], eps[i], p=np.inf)) - 1

    mi_nats = digamma(k) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1))
    return float(mi_nats / math.log(2.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--kind", choices=["visual_vq", "stage2"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-config", default="")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ksg-k", type=int, default=5)
    parser.add_argument("--project-dim", type=int, default=256)
    parser.add_argument("--target", choices=["action", "future_action", "action_sequence"], default="action")
    parser.add_argument(
        "--latent-preprocess",
        choices=["standard", "raw", "global_scale", "norm_augmented", "all_norm_checks"],
        default="standard",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"name={args.name} device={device}", flush=True)

    model = _load_model(args.kind, args.config, args.ckpt, device)
    dm = _load_datamodule(args.data_config or args.config, args.batch_size)
    z, action = _collect_pairs(model, dm.test_dataloader(), device, args.samples, args.target)

    action = _standardize(action)
    preprocesses = (
        ["raw", "global_scale", "norm_augmented"]
        if args.latent_preprocess == "all_norm_checks"
        else [args.latent_preprocess]
    )

    results = []
    for preprocess in preprocesses:
        z_prepared, latent_info = _prepare_latent(z, preprocess, args.project_dim, args.seed)
        mi = ksg_mi_bits(z_prepared, action, k=args.ksg_k)
        result = {
            "name": args.name,
            "samples": int(z_prepared.shape[0]),
            "latent_dim_after_projection": int(z_prepared.shape[1]),
            "action_dim": int(action.shape[1]),
            "ksg_k": int(args.ksg_k),
            "target": args.target,
            "forced_h2_offset": os.environ.get("UNIVLA_LAM_FORCE_H2_OFFSET"),
            "ksg_mi_bits": mi,
        }
        result.update(latent_info)
        results.append(result)
        print(result, flush=True)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "a", encoding="utf-8") as f:
            f.write(yaml.safe_dump(results, sort_keys=False))


if __name__ == "__main__":
    main()
