#!/usr/bin/env python

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _replace(text: str, old: str, new: str) -> str:
    return text.replace(old, new, 1) if old in text else text


def remove_patch_from_lerobot(lerobot_root: Path) -> None:
    policies_root = lerobot_root / "policies"
    async_root = lerobot_root / "async_inference"
    if not policies_root.is_dir() or not async_root.is_dir():
        raise FileNotFoundError(f"{lerobot_root} does not look like a LeRobot package root.")

    factory_path = policies_root / "factory.py"
    text = factory_path.read_text()
    text = _replace(text, "from .univla.configuration_univla import UniVLAConfig\n", "")
    text = _replace(
        text,
        '    elif name == "univla":\n'
        "        from .univla.modeling_univla import UniVLAPolicy\n\n"
        "        return UniVLAPolicy\n",
        "",
    )
    text = _replace(
        text,
        '    elif policy_type == "univla":\n'
        "        return UniVLAConfig(**kwargs)\n",
        "",
    )
    text = _replace(
        text,
        "    elif isinstance(policy_cfg, UniVLAConfig):\n"
        "        from .univla.processor_univla import make_univla_pre_post_processors\n\n"
        "        processors = make_univla_pre_post_processors(\n"
        "            config=policy_cfg,\n"
        '            dataset_stats=kwargs.get("dataset_stats"),\n'
        "        )\n\n",
        "",
    )
    factory_path.write_text(text)

    constants_path = async_root / "constants.py"
    text = constants_path.read_text()
    text = text.replace(', "univla"', "")
    constants_path.write_text(text)

    shutil.rmtree(policies_root / "univla", ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lerobot-root", type=Path, required=True)
    args = parser.parse_args()
    remove_patch_from_lerobot(args.lerobot_root.resolve())
    print(f"Removed built-in UniVLA patch from {args.lerobot_root.resolve()}")


if __name__ == "__main__":
    main()
