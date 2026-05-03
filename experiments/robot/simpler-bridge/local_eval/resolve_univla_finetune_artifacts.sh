#!/usr/bin/env bash
# Source this file after Bridge finetuning finishes:
#   source ./experiments/robot/simpler-bridge/local_eval/resolve_univla_finetune_artifacts.sh

ROOT="${ROOT:-/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong}"
FINETUNE_RUN_ROOT="${FINETUNE_RUN_ROOT:-${ROOT}/outputs/finetune_bridge}"
FINETUNE_RUN_NOTE="${FINETUNE_RUN_NOTE:-bridge_lam_stage2_b200_tf21_vla50k_cleanenv}"

CKPT_PATH="$(find "${FINETUNE_RUN_ROOT}" -maxdepth 1 -type d -name "*${FINETUNE_RUN_NOTE}*" -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk 'NR == 1 { print $2 }')"
if [[ -z "${CKPT_PATH}" ]]; then
  echo "[missing] No finetune run dir under ${FINETUNE_RUN_ROOT} matching ${FINETUNE_RUN_NOTE}" >&2
  return 1 2>/dev/null || exit 1
fi

ACTION_DECODER_PATH="$(find "${CKPT_PATH}" -maxdepth 1 -type f -name 'action_decoder-*.pt' -printf '%T@ %p\n' | sort -nr | awk 'NR == 1 { print $2 }')"
if [[ -z "${ACTION_DECODER_PATH}" ]]; then
  echo "[missing] No action_decoder-*.pt under ${CKPT_PATH}" >&2
  return 1 2>/dev/null || exit 1
fi

export CKPT_PATH
export ACTION_DECODER_PATH

echo "CKPT_PATH=${CKPT_PATH}"
echo "ACTION_DECODER_PATH=${ACTION_DECODER_PATH}"

