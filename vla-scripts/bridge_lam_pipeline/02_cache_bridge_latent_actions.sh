#!/usr/bin/env bash
set -euo pipefail

ROOT="/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong"
REPO="${ROOT}/UniVLA"
DATA_ROOT="${ROOT}/data/rlds_bridge_orig"
LAM_CKPT="${ROOT}/outputs/lam_bridge/logs/task_centric_lam_stage2_bridge/last.ckpt"
CACHE_DIR="${ROOT}/outputs/univla_bridge_lam_local/cache"
CACHE_DB="${CACHE_DB:-${CACHE_DIR}/bridge_latent_actions_stage2.sqlite}"
LOG_DIR="${ROOT}/outputs/univla_bridge_lam_local/run_logs"
LOG="${LOG_DIR}/cache_bridge_latent_actions_stage2.log"

MAX_SAMPLES="${MAX_SAMPLES:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "${CACHE_DIR}" "${LOG_DIR}"

test -f "${LAM_CKPT}"
test -e "${DATA_ROOT}/bridge_orig/1.0.0/dataset_info.json"

export HF_HOME="${HF_HOME:-${ROOT}/data/hf_cache}"
export PYTHONPATH="${REPO}:${REPO}/latent_action_model:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES

cd "${REPO}"

.venv/bin/python vla-scripts/cache_bridge_latent_actions.py \
  --data_root "${DATA_ROOT}" \
  --data_mix bridge \
  --lam_path "${LAM_CKPT}" \
  --output "${CACHE_DB}" \
  --max_samples "${MAX_SAMPLES}" \
  2>&1 | tee "${LOG}"

test -f "${CACHE_DB}"
echo "[cache] done: ${CACHE_DB}"
