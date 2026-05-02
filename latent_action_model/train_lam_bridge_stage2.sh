#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export HF_HOME="${HF_HOME:-/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong/data/hf_cache}"
export WANDB_PROJECT="${WANDB_PROJECT:-univla_lam_bridge}"
export PYTHONPATH="${REPO_ROOT}:${SCRIPT_DIR}:${PYTHONPATH:-}"

cd "${SCRIPT_DIR}"

"${REPO_ROOT}/.venv/bin/torchrun" --standalone --nnodes 1 --nproc-per-node 8 main.py fit \
  --config config/lam-stage-2-bridge.yaml \
  "$@"
