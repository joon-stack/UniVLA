#!/usr/bin/env python3
"""Diagnose the real LeRobot UniVLA runtime path.

This script is meant to be run inside the robot/server Python environment.
It checks which UniVLA package is imported, which artifact files are resolved,
and optionally runs one saved inference dump through the loaded policy.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        default="joon-stack/univla-policy-lerobot-fixedstats-fps-ppnb-hfrad-fps5-step10000",
        help="HF repo id or local exported policy artifact directory.",
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dump-dir", type=Path, default=None, help="Saved server dump dir, e.g. test_gaeSsibal/obs_000016_0003.")
    parser.add_argument("--device", default=None, help="Override config.device before loading the policy.")
    parser.add_argument("--load-policy", action="store_true", help="Load the full policy and VLA weights.")
    parser.add_argument("--run-forward", action="store_true", help="Run predict_action_chunk on --dump-dir. Implies --load-policy.")
    parser.add_argument("--state-ablation", action="store_true", help="Run current/zero/mean state forwards on --dump-dir.")
    parser.add_argument("--debug", action="store_true", help="Enable UNIVLA_DEBUG for policy forward.")
    parser.add_argument("--max-rows", type=int, default=5, help="Rows of action chunk to print.")
    return parser.parse_args()


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def short(value: Any, limit: int = 240) -> str:
    text = repr(value)
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def tensor_summary(tensor: torch.Tensor) -> str:
    data = tensor.detach().float().cpu()
    return (
        f"shape={tuple(data.shape)} dtype={tensor.dtype} "
        f"min={data.min().item():.6g} max={data.max().item():.6g} mean={data.mean().item():.6g}"
    )


def as_list(values: Any, precision: int = 6) -> list[float]:
    if isinstance(values, torch.Tensor):
        array = values.detach().float().cpu().numpy().reshape(-1)
    else:
        array = np.asarray(values, dtype=np.float32).reshape(-1)
    return [round(float(x), precision) for x in array.tolist()]


def import_lerobot_classes() -> tuple[type[Any] | None, type[Any] | None]:
    section("Imports")
    print("python:", sys.executable)
    print("cwd:", os.getcwd())
    print("PYTHONPATH:", os.environ.get("PYTHONPATH", ""))
    print("UNIVLA_REPO_ROOT:", os.environ.get("UNIVLA_REPO_ROOT", ""))

    for package in ("torch", "transformers", "tokenizers", "huggingface_hub"):
        try:
            module = __import__(package)
            print(f"{package}:", getattr(module, "__version__", None), getattr(module, "__file__", None))
        except Exception as exc:
            print(f"ERROR importing {package}:", repr(exc))

    try:
        import lerobot

        print("lerobot:", getattr(lerobot, "__file__", None))
    except Exception as exc:
        print("ERROR importing lerobot:", repr(exc))

    try:
        import lerobot_policy_univla.configuration_univla as cfg_mod
        import lerobot_policy_univla.modeling_univla as pol_mod
        import lerobot_policy_univla.processor_univla as proc_mod

        print("lerobot_policy_univla.configuration:", inspect.getfile(cfg_mod))
        print("lerobot_policy_univla.modeling:", inspect.getfile(pol_mod))
        print("lerobot_policy_univla.processor:", inspect.getfile(proc_mod))
    except Exception as exc:
        print("ERROR importing lerobot_policy_univla:", repr(exc))

    try:
        from lerobot.utils.import_utils import register_third_party_plugins

        register_third_party_plugins()
        print("register_third_party_plugins: ok")
    except Exception as exc:
        print("register_third_party_plugins: failed:", repr(exc))

    cfg_cls = None
    pol_cls = None
    try:
        try:
            from lerobot.configs import PreTrainedConfig
        except ImportError:
            from lerobot.configs.policies import PreTrainedConfig

        cfg_cls = PreTrainedConfig.get_choice_class("univla")
        print("registered config class:", cfg_cls)
        print("registered config file:", inspect.getfile(cfg_cls))
    except Exception as exc:
        print("registered config lookup failed:", repr(exc))

    try:
        from lerobot.policies.factory import get_policy_class

        pol_cls = get_policy_class("univla")
        print("registered policy class:", pol_cls)
        print("registered policy file:", inspect.getfile(pol_cls))
    except Exception as exc:
        print("registered policy lookup failed:", repr(exc))
        try:
            from lerobot_policy_univla.modeling_univla import UniVLAPolicy

            pol_cls = UniVLAPolicy
            print("fallback policy class:", pol_cls)
            print("fallback policy file:", inspect.getfile(pol_cls))
        except Exception as fallback_exc:
            print("fallback policy import failed:", repr(fallback_exc))

    try:
        import prismatic.extern.hf.modeling_prismatic as hf_mod

        path = Path(inspect.getfile(hf_mod))
        text = path.read_text()
        print("prismatic hf modeling:", path)
        print("has LatentActionTokenMaskLogitsProcessor:", "LatentActionTokenMaskLogitsProcessor" in text)
        print("has suffix attention_mask patch:", 'kwargs["attention_mask"] = torch.cat' in text)
    except Exception as exc:
        print("ERROR importing prismatic hf modeling:", repr(exc))

    return cfg_cls, pol_cls


def snapshot_dir(policy: str, revision: str | None, cache_dir: str | None, local_files_only: bool) -> Path | None:
    path = Path(policy)
    if path.is_dir():
        return path.resolve()
    try:
        from huggingface_hub import snapshot_download

        return Path(
            snapshot_download(
                repo_id=policy,
                revision=revision,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
        )
    except Exception as exc:
        print("snapshot_download failed:", repr(exc))
        return None


def load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def inspect_artifact(args: argparse.Namespace) -> tuple[Path | None, dict[str, Any] | None, dict[str, Any] | None]:
    section("Artifact")
    artifact_dir = snapshot_dir(args.policy, args.revision, args.cache_dir, args.local_files_only)
    print("policy:", args.policy)
    print("artifact_dir:", artifact_dir)
    if artifact_dir is None:
        return None, None, None

    config_path = artifact_dir / "config.json"
    if not config_path.exists():
        print("ERROR missing config.json")
        return artifact_dir, None, None

    config = load_json(config_path)
    keys = [
        "type",
        "policy_type",
        "dummy",
        "vla_path",
        "action_decoder_path",
        "dataset_statistics_path",
        "dataset_name",
        "image_key",
        "state_key",
        "latent_action_token_len",
        "action_vocab_size",
        "action_token_mask",
        "action_token_id_offset",
        "radius_action_vocab_size",
        "direction_action_vocab_size",
        "use_proprio",
        "use_history_action",
        "do_sample",
        "n_action_steps",
        "window_size",
        "decoder_output_tanh",
        "normalization_mapping",
    ]
    for key in keys:
        if key in config:
            print(f"{key}: {short(config[key])}")

    for field in ("vla_path", "action_decoder_path", "dataset_statistics_path"):
        value = config.get(field)
        if not value:
            print(f"{field}: empty")
            continue
        value_path = Path(value)
        artifact_candidate = artifact_dir / value
        cwd_candidate = Path.cwd() / value
        print(
            f"{field}: value={value!r} absolute={value_path.is_absolute()} "
            f"artifact_exists={artifact_candidate.exists()} cwd_exists={cwd_candidate.exists()}"
        )
        if not value_path.is_absolute() and cwd_candidate.exists() and artifact_candidate.exists():
            print(f"WARNING {field}: cwd path can shadow artifact path in stale loaders: {cwd_candidate}")

    stats = None
    stats_path_value = config.get("dataset_statistics_path") or "dataset_statistics.json"
    stats_path = Path(stats_path_value)
    if not stats_path.is_absolute():
        stats_path = artifact_dir / stats_path_value
    if stats_path.exists():
        stats = load_json(stats_path)
        print("stats_path:", stats_path)
        print("stats_datasets:", list(stats.keys()))
    else:
        print("stats_path missing:", stats_path)

    return artifact_dir, config, stats


def inspect_dump(dump_dir: Path, stats: dict[str, Any] | None, dataset_name: str | None) -> dict[str, Any] | None:
    section("Dump")
    print("dump_dir:", dump_dir)
    metadata_path = dump_dir / "metadata.json"
    tensors_path = dump_dir / "tensors.pt"
    if not metadata_path.exists() or not tensors_path.exists():
        print("ERROR dump needs metadata.json and tensors.pt")
        return None

    metadata = load_json(metadata_path)
    print("metadata.pretrained_name_or_path:", metadata.get("pretrained_name_or_path"))
    print("metadata.timestep:", metadata.get("timestep"), "actions_per_chunk:", metadata.get("actions_per_chunk"))
    print("metadata.task:", metadata.get("task"))

    tensors = torch.load(tensors_path, map_location="cpu")
    for obs_key in ("prepared_observation", "preprocessed_observation"):
        obs = tensors.get(obs_key, {})
        print(f"{obs_key}.keys:", list(obs.keys()))
        state = obs.get("observation.state")
        if isinstance(state, torch.Tensor):
            print(f"{obs_key}.state:", tensor_summary(state), as_list(state))
        image = obs.get("observation.images.top")
        if isinstance(image, torch.Tensor):
            print(f"{obs_key}.image.top:", tensor_summary(image))

    state = tensors["preprocessed_observation"].get("observation.state")
    if isinstance(state, torch.Tensor) and stats:
        chosen = dataset_name
        if not chosen:
            chosen = next(iter(stats)) if len(stats) == 1 else None
        if chosen and chosen in stats and "proprio" in stats[chosen]:
            prop = stats[chosen]["proprio"]
            mean = torch.tensor(prop["mean"], dtype=torch.float32)
            std = torch.clamp(torch.tensor(prop["std"], dtype=torch.float32), min=1e-2)
            z = (state.squeeze(0).float() - mean) / std
            print("proprio_stats_dataset:", chosen)
            print("state_zscore:", as_list(z))

    for key in ("full_raw_policy_action_chunk", "full_postprocessed_action_chunk"):
        value = tensors.get(key)
        if isinstance(value, torch.Tensor):
            print(f"{key}:", tensor_summary(value))
            rows = value[0] if value.ndim == 3 else value
            for i in range(min(3, rows.shape[0])):
                print(f"{key}[{i}]:", as_list(rows[i]))
    return tensors


def unnormalize_action(normed: torch.Tensor, stats: dict[str, Any], dataset_name: str | None) -> torch.Tensor:
    chosen = dataset_name or (next(iter(stats)) if len(stats) == 1 else None)
    if chosen is None or chosen not in stats:
        raise KeyError(f"Cannot choose action stats dataset from {list(stats)}")
    action_stats = stats[chosen]["action"]
    low = torch.tensor(action_stats["q01"], dtype=torch.float32, device=normed.device)
    high = torch.tensor(action_stats["q99"], dtype=torch.float32, device=normed.device)
    return 0.5 * (normed.float() + 1.0) * (high - low) + low


def make_batch(tensors: dict[str, Any]) -> dict[str, Any]:
    batch = copy.deepcopy(tensors["preprocessed_observation"])
    for key, value in list(batch.items()):
        if isinstance(value, torch.Tensor):
            batch[key] = value.clone()
    return batch


def print_chunk(label: str, normed: torch.Tensor, post: torch.Tensor, rows: int) -> None:
    print(f"\n{label} normed:", tensor_summary(normed))
    normed_rows = normed[0] if normed.ndim == 3 else normed
    post_rows = post[0] if post.ndim == 3 else post
    for i in range(min(rows, normed_rows.shape[0])):
        print(f"{label} normed[{i}]:", as_list(normed_rows[i]))
        print(f"{label} post[{i}]:", as_list(post_rows[i]))
    print(f"{label} post[-1]:", as_list(post_rows[-1]))


def load_policy_and_forward(
    args: argparse.Namespace,
    pol_cls: type[Any] | None,
    config: dict[str, Any] | None,
    stats: dict[str, Any] | None,
    tensors: dict[str, Any] | None,
) -> None:
    if not args.load_policy and not args.run_forward and not args.state_ablation:
        return
    if pol_cls is None:
        print("ERROR no policy class available")
        return
    if args.debug or args.run_forward:
        os.environ.setdefault("UNIVLA_DEBUG", "1")
        os.environ.setdefault("UNIVLA_DEBUG_MAX_CALLS", "20")

    section("Policy Load")
    kwargs: dict[str, Any] = {
        "local_files_only": args.local_files_only,
    }
    if args.revision:
        kwargs["revision"] = args.revision
    if args.cache_dir:
        kwargs["cache_dir"] = args.cache_dir

    policy = pol_cls.from_pretrained(args.policy, **kwargs)
    if args.device is not None:
        policy.to(args.device)
        if hasattr(policy, "config"):
            policy.config.device = args.device
    policy.eval()

    for attr in (
        "_resolved_vla_path",
        "_resolved_action_decoder_path",
        "_resolved_dataset_statistics_path",
        "_dataset_name",
        "_action_decoder_uses_proprio",
        "_action_decoder_uses_wrist",
    ):
        print(f"{attr}:", getattr(policy, attr, None))
    if getattr(policy, "_proprio_mean", None) is not None:
        print("_proprio_mean:", as_list(policy._proprio_mean))
        print("_proprio_std:", as_list(policy._proprio_std))

    if not args.run_forward and not args.state_ablation:
        return
    if tensors is None or stats is None:
        print("ERROR --run-forward needs --dump-dir and dataset stats")
        return

    section("Policy Forward")
    dataset_name = getattr(policy, "_dataset_name", None)

    def run(label: str, batch: dict[str, Any]) -> None:
        with torch.no_grad():
            normed = policy.predict_action_chunk(batch)
        post = unnormalize_action(normed.detach(), stats, dataset_name)
        print_chunk(label, normed.detach().cpu(), post.detach().cpu(), args.max_rows)

    batch = make_batch(tensors)
    run("current_state", batch)

    if args.state_ablation:
        state = batch.get("observation.state")
        if not isinstance(state, torch.Tensor):
            print("state ablation skipped: no observation.state tensor")
            return
        zero_batch = make_batch(tensors)
        zero_batch["observation.state"] = torch.zeros_like(state)
        run("zero_raw_state", zero_batch)

        if dataset_name and dataset_name in stats and "proprio" in stats[dataset_name]:
            mean_batch = make_batch(tensors)
            mean_state = torch.tensor(stats[dataset_name]["proprio"]["mean"], dtype=state.dtype).view_as(state)
            mean_batch["observation.state"] = mean_state
            run("mean_raw_state", mean_batch)


def main() -> None:
    args = parse_args()
    if args.run_forward or args.state_ablation:
        args.load_policy = True

    cfg_cls, pol_cls = import_lerobot_classes()
    del cfg_cls
    artifact_dir, config, stats = inspect_artifact(args)
    del artifact_dir

    tensors = None
    if args.dump_dir is not None:
        tensors = inspect_dump(args.dump_dir, stats, (config or {}).get("dataset_name"))

    load_policy_and_forward(args, pol_cls, config, stats, tensors)


if __name__ == "__main__":
    main()
