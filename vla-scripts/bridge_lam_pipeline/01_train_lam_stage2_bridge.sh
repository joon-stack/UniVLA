#!/usr/bin/env bash
set -euo pipefail

ROOT="/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong"
REPO="${ROOT}/UniVLA"
STAGE1_CKPT="${ROOT}/outputs/lam_bridge/logs/task_centric_lam_stage1_bridge/epoch=1-step=60000.ckpt"
STAGE2_DIR="${ROOT}/outputs/lam_bridge/logs/task_centric_lam_stage2_bridge"
STAGE2_LAST="${STAGE2_DIR}/last.ckpt"
LOG_DIR="${ROOT}/outputs/lam_bridge/run_logs"
LOG="${LOG_DIR}/lam_stage2_bridge_8xb200.log"

mkdir -p "${LOG_DIR}"

test -f "${STAGE1_CKPT}"
test -e "${ROOT}/data/rlds_bridge_orig/bridge_dataset/1.0.0/dataset_info.json"

export HF_HOME="${HF_HOME:-${ROOT}/data/hf_cache}"
export WANDB_PROJECT="${WANDB_PROJECT:-univla_lam_bridge}"
export WANDB_NAME="${WANDB_NAME:-lam_stage2_bridge_8xb200}"
export PYTHONPATH="${REPO}:${REPO}/latent_action_model:${PYTHONPATH:-}"
export UNIVLA_TRAJ_THREADS="${UNIVLA_TRAJ_THREADS:-4}"
export UNIVLA_TRAJ_READ_THREADS="${UNIVLA_TRAJ_READ_THREADS:-4}"
export UNIVLA_FRAME_THREADS="${UNIVLA_FRAME_THREADS:-8}"

cd "${REPO}/latent_action_model"

./train_lam_bridge_stage2.sh 2>&1 | tee "${LOG}"

test -f "${STAGE2_LAST}"
echo "[stage2] done: ${STAGE2_LAST}"
