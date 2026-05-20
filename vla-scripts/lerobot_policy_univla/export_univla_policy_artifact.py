#!/usr/bin/env python

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
from pathlib import Path
from typing import Any

import torch

import lerobot_policy_univla  # noqa: F401
try:
    from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
except ImportError:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE


def _load_stats(path: Path, dataset_name: str | None) -> tuple[str, dict[str, Any]]:
    with open(path, "r") as f:
        stats = json.load(f)
    if dataset_name is None:
        if len(stats) != 1:
            raise ValueError(
                f"dataset_statistics.json contains multiple datasets {list(stats)}; pass --dataset-name."
            )
        dataset_name = next(iter(stats))
    if dataset_name not in stats:
        raise KeyError(f"Dataset {dataset_name!r} not found in {path}; available={list(stats)}")
    return dataset_name, stats[dataset_name]


def _tensor_stats(entry: dict[str, Any], keys: tuple[str, ...]) -> dict[str, torch.Tensor]:
    result = {}
    for key in keys:
        if key in entry:
            result[key] = torch.tensor(entry[key], dtype=torch.float32)
    return result


def _read_json_from_vla_path(vla_path: str, filename: str) -> dict[str, Any] | None:
    local_path = Path(vla_path) / filename
    if local_path.exists():
        with open(local_path, "r") as f:
            return json.load(f)

    if "/" not in vla_path:
        return None
    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(repo_id=vla_path, filename=filename, repo_type="model")
        with open(downloaded, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _resolve_latent_action_token_len(vla_path: str, requested: int | None) -> int:
    config = _read_json_from_vla_path(vla_path, "config.json") or {}
    meta = _read_json_from_vla_path(vla_path, "eval_meta.json") or {}
    inferred = config.get("latent_action_token_len") or meta.get("latent_action_token_len")

    if requested is not None:
        if inferred is not None and int(inferred) != int(requested):
            raise ValueError(
                "latent action token length mismatch: "
                f"--latent-action-token-len={requested}, but {vla_path}/config.json or eval_meta.json says {inferred}. "
                "Use the token length that matches the VLA/LAM checkpoint."
            )
        return int(requested)

    if inferred is None:
        raise ValueError(
            "Could not infer latent action token length from vla_path/config.json or eval_meta.json. "
            "Pass --latent-action-token-len explicitly. hfrad/euclidean-5token use 5; UniVLA stage2 uses 4."
        )
    return int(inferred)


def _copy_file_if_needed(src: Path, dst: Path) -> None:
    if src.resolve() == dst.resolve():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_vla_snapshot(vla_path: str, dst: Path) -> None:
    src = Path(vla_path)
    if not src.is_dir():
        from huggingface_hub import snapshot_download

        src = Path(snapshot_download(repo_id=vla_path, repo_type="model"))
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".git"))


def _config_init_fields(cfg_cls: type) -> set[str]:
    if not dataclasses.is_dataclass(cfg_cls):
        return set()
    return {field.name for field in dataclasses.fields(cfg_cls) if field.init}


def _patch_saved_config(output_dir: Path, values: dict[str, Any]) -> None:
    config_path = output_dir / "config.json"
    if not values or not config_path.exists():
        return
    with open(config_path, "r") as f:
        config = json.load(f)
    config.update(values)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _normalize_wrist_fusion(value: str | None) -> str:
    mode = str(value or "none").strip().lower()
    aliases = {"late": "decoder_residual", "decoder_late": "decoder_residual"}
    mode = aliases.get(mode, mode)
    if mode not in {"none", "decoder_residual"}:
        raise ValueError(f"Unsupported wrist_fusion={value!r}. Use 'none' or 'decoder_residual'.")
    return mode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--vla-path", required=True)
    parser.add_argument("--action-decoder-path", required=True)
    parser.add_argument("--dataset-statistics-path", required=True, type=Path)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--univla-repo-root", default=None)
    parser.add_argument("--image-key", default="observation.images.primary")
    parser.add_argument("--wrist-fusion", default="none")
    parser.add_argument("--wrist-image-key", default="observation.images.wrist")
    parser.add_argument("--state-key", default=OBS_STATE)
    parser.add_argument("--task-key", default="task")
    parser.add_argument("--image-size", default=224, type=int)
    parser.add_argument("--state-dim", default=7, type=int)
    parser.add_argument("--action-dim", default=7, type=int)
    parser.add_argument("--window-size", default=10, type=int)
    parser.add_argument("--n-action-steps", default=10, type=int)
    parser.add_argument("--latent-action-token-len", default=None, type=int)
    parser.add_argument("--action-vocab-size", default=None, type=int)
    parser.add_argument("--action-token-mask", default="none", choices=("none", "flat", "all", "factorized_rad_dir"))
    parser.add_argument("--action-token-id-offset", default=32001, type=int)
    parser.add_argument("--radius-action-vocab-size", default=16, type=int)
    parser.add_argument("--direction-action-vocab-size", default=16, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--decoder-output-tanh", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-history-action", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--embed-vla",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Copy the VLA snapshot into output-dir/vla and store relative paths for HF policy repos.",
    )
    args = parser.parse_args()
    args.wrist_fusion = _normalize_wrist_fusion(args.wrist_fusion)

    latent_action_token_len = _resolve_latent_action_token_len(args.vla_path, args.latent_action_token_len)
    dataset_name, dataset_stats = _load_stats(args.dataset_statistics_path, args.dataset_name)
    action_stats = dataset_stats["action"]
    if len(action_stats["q01"]) != args.action_dim:
        raise ValueError(f"action_dim={args.action_dim} but stats action dim={len(action_stats['q01'])}")

    lerobot_stats = {
        ACTION: _tensor_stats(action_stats, ("mean", "std", "min", "max", "q01", "q99")),
    }
    if "proprio" in dataset_stats:
        lerobot_stats[args.state_key] = _tensor_stats(
            dataset_stats["proprio"], ("mean", "std", "min", "max", "q01", "q99")
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_action_decoder = args.output_dir / Path(args.action_decoder_path).name
    artifact_stats = args.output_dir / "dataset_statistics.json"
    _copy_file_if_needed(Path(args.action_decoder_path), artifact_action_decoder)
    _copy_file_if_needed(args.dataset_statistics_path, artifact_stats)

    config_vla_path = args.vla_path
    if args.embed_vla:
        _copy_vla_snapshot(args.vla_path, args.output_dir / "vla")
        config_vla_path = "vla"

    cfg_cls = PreTrainedConfig.get_choice_class("univla")
    input_features = {
        args.image_key: PolicyFeature(FeatureType.VISUAL, (3, args.image_size, args.image_size)),
        args.state_key: PolicyFeature(FeatureType.STATE, (args.state_dim,)),
    }
    wrist_config = {}
    if args.wrist_fusion != "none":
        input_features[args.wrist_image_key] = PolicyFeature(FeatureType.VISUAL, (3, args.image_size, args.image_size))
        wrist_config = {
            "wrist_fusion": args.wrist_fusion,
            "wrist_image_key": args.wrist_image_key,
        }

    cfg_kwargs = {
        "input_features": input_features,
        "output_features": {ACTION: PolicyFeature(FeatureType.ACTION, (args.action_dim,))},
        "device": args.device,
        "dummy": False,
        "vla_path": config_vla_path,
        "action_decoder_path": artifact_action_decoder.name,
        "dataset_statistics_path": artifact_stats.name,
        "dataset_name": dataset_name,
        "univla_repo_root": args.univla_repo_root,
        "image_key": args.image_key,
        "state_key": args.state_key,
        "task_key": args.task_key,
        "window_size": args.window_size,
        "n_action_steps": args.n_action_steps,
        "action_dim": args.action_dim,
        "state_dim": args.state_dim,
        "latent_action_token_len": latent_action_token_len,
        "action_token_mask": args.action_token_mask,
        "action_token_id_offset": args.action_token_id_offset,
        "radius_action_vocab_size": args.radius_action_vocab_size,
        "direction_action_vocab_size": args.direction_action_vocab_size,
        "torch_dtype": args.torch_dtype,
        "attn_implementation": args.attn_implementation,
        "decoder_output_tanh": args.decoder_output_tanh,
        "use_history_action": args.use_history_action,
    }
    if args.action_vocab_size is not None:
        cfg_kwargs["action_vocab_size"] = args.action_vocab_size

    cfg_fields = _config_init_fields(cfg_cls)
    cfg_kwargs.update({key: value for key, value in wrist_config.items() if key in cfg_fields})
    cfg = cfg_cls(**cfg_kwargs)

    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=lerobot_stats)
    cfg.save_pretrained(args.output_dir)
    _patch_saved_config(args.output_dir, {key: value for key, value in wrist_config.items() if key not in cfg_fields})
    preprocessor.save_pretrained(args.output_dir)
    postprocessor.save_pretrained(args.output_dir)
    (args.output_dir / "train_config.json").write_text(
        json.dumps(
            {
                "policy": "univla",
                "dataset_name": dataset_name,
                "vla_path": config_vla_path,
                "action_decoder_path": artifact_action_decoder.name,
                "dataset_statistics_path": artifact_stats.name,
                "latent_action_token_len": latent_action_token_len,
                "action_vocab_size": args.action_vocab_size,
                "action_token_mask": args.action_token_mask,
                "action_token_id_offset": args.action_token_id_offset,
                "radius_action_vocab_size": args.radius_action_vocab_size,
                "direction_action_vocab_size": args.direction_action_vocab_size,
                **wrist_config,
                "normalization_mapping": {
                    "VISUAL": "IDENTITY",
                    "STATE": "IDENTITY",
                    "ACTION": "QUANTILES",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
