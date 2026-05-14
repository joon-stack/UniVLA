#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path


def _replace_once(text: str, old: str, new: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise RuntimeError(f"Could not find patch anchor:\n{old}")
    return text.replace(old, new, 1)


def patch_constants(constants_path: Path) -> None:
    text = constants_path.read_text()
    if '"univla"' in text:
        constants_path.write_text(text)
        return

    old = 'SUPPORTED_POLICIES = ["act", "smolvla", "diffusion", "tdmpc", "vqbet", "pi0", "pi05", "groot"]'
    new = (
        'SUPPORTED_POLICIES = ["act", "smolvla", "diffusion", "tdmpc", "vqbet", '
        '"pi0", "pi05", "groot", "univla"]'
    )
    if old not in text:
        raise RuntimeError(f"Could not patch {constants_path}; SUPPORTED_POLICIES anchor changed.")
    constants_path.write_text(text.replace(old, new, 1))


def patch_policy_server(policy_server_path: Path) -> None:
    text = policy_server_path.read_text()
    text = _replace_once(
        text,
        "from lerobot.types import PolicyAction\n",
        "from lerobot.types import PolicyAction\n"
        "from lerobot.utils.import_utils import register_third_party_plugins\n",
    )
    text = _replace_once(
        text,
        "        if policy_specs.policy_type not in SUPPORTED_POLICIES:\n",
        "        register_third_party_plugins()\n\n"
        "        if policy_specs.policy_type not in SUPPORTED_POLICIES:\n",
    )
    policy_server_path.write_text(text)


def patch_lerobot(lerobot_root: Path) -> None:
    async_root = lerobot_root / "async_inference"
    if not async_root.is_dir():
        raise FileNotFoundError(f"{lerobot_root} does not look like a LeRobot package root.")
    patch_constants(async_root / "constants.py")
    patch_policy_server(async_root / "policy_server.py")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lerobot-root", type=Path, required=True)
    args = parser.parse_args()
    patch_lerobot(args.lerobot_root.resolve())
    print(f"Patched LeRobot async policy_server for lerobot_policy_univla at {args.lerobot_root.resolve()}")


if __name__ == "__main__":
    main()
