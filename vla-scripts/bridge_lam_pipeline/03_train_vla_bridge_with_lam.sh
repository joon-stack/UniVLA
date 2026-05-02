#!/usr/bin/env bash
set -euo pipefail

ROOT="/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong"
REPO="${ROOT}/UniVLA"
DATA_ROOT="${ROOT}/data/rlds_bridge_orig"
BASE_VLM="${ROOT}/data/univla_checkpoints/prismatic-vlms/prism-dinosiglip-224px+7b"
LAM_CKPT="${LAM_CKPT:-${ROOT}/outputs/lam_bridge/logs/task_centric_lam_stage2_bridge/last.ckpt}"
RUN_ROOT="${ROOT}/outputs/univla_bridge_lam_local"
LOG_DIR="${ROOT}/outputs/univla_bridge_lam_local/run_logs"
LOG_NAME="${LOG_NAME:-vla_bridge_with_local_lam_8xb200.log}"
LOG="${LOG_DIR}/${LOG_NAME}"

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-28611}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MAX_STEPS="${MAX_STEPS:-20000}"
SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-20000}"
IMAGE_AUG="${IMAGE_AUG:-true}"
WANDB_ENTITY="${WANDB_ENTITY:-joonstack}"
WANDB_PROJECT="${WANDB_PROJECT:-univla_bridge_lam_local}"
RUN_ID_NOTE="${RUN_ID_NOTE:-bridge_dataset_local_lam_stage2_b200}"

mkdir -p "${LOG_DIR}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export UNIVLA_TRAJ_THREADS="${UNIVLA_TRAJ_THREADS:-4}"
export UNIVLA_TRAJ_READ_THREADS="${UNIVLA_TRAJ_READ_THREADS:-4}"
export UNIVLA_FRAME_THREADS="${UNIVLA_FRAME_THREADS:-8}"
export UNIVLA_TF_RAM_BUDGET_MB="${UNIVLA_TF_RAM_BUDGET_MB:-512}"
export UNIVLA_PROFILE_STEPS="${UNIVLA_PROFILE_STEPS:-0}"
export UNIVLA_DUMMY_LATENT_ACTIONS="${UNIVLA_DUMMY_LATENT_ACTIONS:-0}"

test -f "${BASE_VLM}/checkpoints/latest-checkpoint.pt"
test -f "${LAM_CKPT}"
test -e "${DATA_ROOT}/bridge_dataset/1.0.0/dataset_info.json"

cd "${REPO}"

ARGS=(
  --vla.type prism-dinosiglip-224px+mx-bridge
  --vla.max_steps "${MAX_STEPS}"
  --vla.shuffle_buffer_size "${SHUFFLE_BUFFER_SIZE}"
  --image_aug "${IMAGE_AUG}"
  --pretrain_vlm "${BASE_VLM}"
  --lam_path "${LAM_CKPT}"
  --data_root_dir "${DATA_ROOT}"
  --run_root_dir "${RUN_ROOT}"
  --wandb_project "${WANDB_PROJECT}"
  --wandb_entity "${WANDB_ENTITY}"
  --run_id_note "${RUN_ID_NOTE}"
)

exec .venv/bin/torchrun \
  --standalone \
  --nproc_per_node "${GPUS_PER_NODE}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  vla-scripts/train.py \
  "${ARGS[@]}" \
  2>&1 | tee "${LOG}"
