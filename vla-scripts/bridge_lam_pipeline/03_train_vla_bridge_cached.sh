#!/usr/bin/env bash
set -euo pipefail

ROOT="/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong"
REPO="${ROOT}/UniVLA"
DATA_ROOT="${ROOT}/data/rlds_bridge_orig"
BASE_VLM="${ROOT}/data/univla_checkpoints/prismatic-vlms/prism-dinosiglip-224px+7b"
CACHE_DB="${CACHE_DB:-${ROOT}/outputs/univla_bridge_lam_local/cache/bridge_latent_actions_stage2.sqlite}"
RUN_ROOT="${ROOT}/outputs/univla_bridge_lam_local"
LOG_DIR="${RUN_ROOT}/run_logs"
LOG="${LOG_DIR}/vla_bridge_cached_8xb200.log"

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-28671}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MAX_STEPS="${MAX_STEPS:-20000}"
WANDB_ENTITY="${WANDB_ENTITY:-joonstack}"
WANDB_PROJECT="${WANDB_PROJECT:-univla_bridge_lam_local}"
RUN_ID_NOTE="${RUN_ID_NOTE:-bridge_orig_cached_lam_stage2_b200}"
IMAGE_AUG="${IMAGE_AUG:-false}"

mkdir -p "${LOG_DIR}"

test -f "${BASE_VLM}/checkpoints/latest-checkpoint.pt"
test -f "${CACHE_DB}"
test -e "${DATA_ROOT}/bridge_orig/1.0.0/dataset_info.json"

export HF_HOME="${HF_HOME:-${ROOT}/data/hf_cache}"
export PYTHONPATH="${REPO}:${REPO}/latent_action_model:${PYTHONPATH:-}"

cd "${REPO}"

exec .venv/bin/torchrun \
  --standalone \
  --nproc_per_node "${GPUS_PER_NODE}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  vla-scripts/train.py \
  --vla.type prism-dinosiglip-224px+mx-bridge \
  --vla.max_steps "${MAX_STEPS}" \
  --pretrain_vlm "${BASE_VLM}" \
  --latent_action_cache_path "${CACHE_DB}" \
  --data_root_dir "${DATA_ROOT}" \
  --run_root_dir "${RUN_ROOT}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_entity "${WANDB_ENTITY}" \
  --run_id_note "${RUN_ID_NOTE}" \
  --image_aug "${IMAGE_AUG}" \
  2>&1 | tee "${LOG}"
