#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

import lerobot_policy_univla  # noqa: F401
from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--vla-path", required=True)
    parser.add_argument("--action-decoder-path", required=True)
    parser.add_argument("--dataset-statistics-path", required=True, type=Path)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--univla-repo-root", default=None)
    parser.add_argument("--image-key", default="observation.images.primary")
    parser.add_argument("--state-key", default=OBS_STATE)
    parser.add_argument("--task-key", default="task")
    parser.add_argument("--image-size", default=224, type=int)
    parser.add_argument("--state-dim", default=7, type=int)
    parser.add_argument("--action-dim", default=7, type=int)
    parser.add_argument("--window-size", default=10, type=int)
    parser.add_argument("--n-action-steps", default=10, type=int)
    parser.add_argument("--latent-action-token-len", default=4, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--decoder-output-tanh", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-history-action", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

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

    cfg_cls = PreTrainedConfig.get_choice_class("univla")
    cfg = cfg_cls(
        input_features={
            args.image_key: PolicyFeature(FeatureType.VISUAL, (3, args.image_size, args.image_size)),
            args.state_key: PolicyFeature(FeatureType.STATE, (args.state_dim,)),
        },
        output_features={ACTION: PolicyFeature(FeatureType.ACTION, (args.action_dim,))},
        device=args.device,
        dummy=False,
        vla_path=args.vla_path,
        action_decoder_path=args.action_decoder_path,
        dataset_statistics_path=str(args.dataset_statistics_path),
        dataset_name=dataset_name,
        univla_repo_root=args.univla_repo_root,
        image_key=args.image_key,
        state_key=args.state_key,
        task_key=args.task_key,
        window_size=args.window_size,
        n_action_steps=args.n_action_steps,
        action_dim=args.action_dim,
        state_dim=args.state_dim,
        latent_action_token_len=args.latent_action_token_len,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        decoder_output_tanh=args.decoder_output_tanh,
        use_history_action=args.use_history_action,
    )

    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=lerobot_stats)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.save_pretrained(args.output_dir)
    preprocessor.save_pretrained(args.output_dir)
    postprocessor.save_pretrained(args.output_dir)
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
