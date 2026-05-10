#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export HF_HOME="${HF_HOME:-/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong/data/hf_cache}"
export WANDB_PROJECT="${WANDB_PROJECT:-univla_lam_bridge}"
export PYTHONPATH="${REPO_ROOT}:${SCRIPT_DIR}:${PYTHONPATH:-}"
# Keep Lightning from cycling the RLDS IterableDataset at the natural Bridge epoch
# boundary (~33k steps with batch_size=64), where TFDS can hang while rebuilding.
export UNIVLA_RLDS_LEN_OVERRIDE="${UNIVLA_RLDS_LEN_OVERRIDE:-20000000}"
TORCHRUN="${TORCHRUN:-${REPO_ROOT}/.venv/bin/torchrun}"
if [[ ! -x "${TORCHRUN}" ]]; then
  TORCHRUN="/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong/UniVLA/.venv/bin/torchrun"
fi
echo "Using torchrun: ${TORCHRUN}"

cd "${SCRIPT_DIR}"

"${TORCHRUN}" --standalone --nnodes 1 --nproc-per-node "${LAM_NPROC_PER_NODE:-8}" main_visual_vq.py fit \
  --config config/lam-visual-vq-bridge-hyperbolic-factorized-warmup-50k.yaml \
  "$@"
