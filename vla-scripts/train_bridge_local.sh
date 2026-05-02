#!/usr/bin/env bash
set -euo pipefail

ROOT="/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong"
REPO="${ROOT}/UniVLA"
DATA_ROOT="${ROOT}/data/rlds_bridge_orig"
BASE_VLM="${ROOT}/data/univla_checkpoints/prismatic-vlms/prism-dinosiglip-224px+7b"
LAM_CKPT="${ROOT}/data/univla_checkpoints/univla-latent-action-model/lam-stage-2.ckpt"
RUN_ROOT="${ROOT}/outputs/univla_bridge"

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-28596}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MAX_STEPS="${MAX_STEPS:-}"
WANDB_ENTITY="${WANDB_ENTITY:-joonstack}"
WANDB_PROJECT="${WANDB_PROJECT:-univla_bridge}"

test -f "${BASE_VLM}/checkpoints/latest-checkpoint.pt"
test -f "${LAM_CKPT}"
test -e "${DATA_ROOT}/bridge_orig/1.0.0/dataset_info.json"

cd "${REPO}"

ARGS=(
  --vla.type prism-dinosiglip-224px+mx-bridge
  --pretrain_vlm "${BASE_VLM}"
  --lam_path "${LAM_CKPT}"
  --data_root_dir "${DATA_ROOT}"
  --run_root_dir "${RUN_ROOT}"
  --wandb_project "${WANDB_PROJECT}"
  --wandb_entity "${WANDB_ENTITY}"
  --run_id_note "bridge_orig_b200"
)

if [[ -n "${MAX_STEPS}" ]]; then
  ARGS+=(--vla.max_steps "${MAX_STEPS}")
fi

exec .venv/bin/torchrun \
  --standalone \
  --nproc_per_node "${GPUS_PER_NODE}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  vla-scripts/train.py \
  "${ARGS[@]}"
