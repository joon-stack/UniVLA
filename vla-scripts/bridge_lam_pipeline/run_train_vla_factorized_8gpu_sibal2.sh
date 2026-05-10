#!/usr/bin/env bash
set -euo pipefail

ROOT=/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong
REPO=${ROOT}/UniVLA-hyper
PYTHON_BIN=${ROOT}/UniVLA/.venvs/sibal2/bin/python

cd "${REPO}"

env \
  ROOT="${ROOT}" \
  REPO="${REPO}" \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  NPROC_PER_NODE=8 \
  UNIVLA_PIPELINE_PYTHON="${PYTHON_BIN}" \
  PYTHON="${PYTHON_BIN}" \
  WANDB_MODE=online \
  TOKENIZERS_PARALLELISM=false \
  OMP_NUM_THREADS=8 \
  UNIVLA_PROFILE_STEPS=20 \
  UNIVLA_VLA_DATALOADER_WORKERS=0 \
  UNIVLA_TRAJ_THREADS=4 \
  UNIVLA_TRAJ_READ_THREADS=4 \
  UNIVLA_FRAME_THREADS=8 \
  UNIVLA_TF_RAM_BUDGET_MB=512 \
  VLA_MAX_STEPS=50000 \
  VLA_SHUFFLE_BUFFER_SIZE=20000 \
  VLA_WANDB_PROJECT=univla_bridge_lam_local \
  LOG_ROOT="${ROOT}/outputs/bridge_pipeline_logs/factorized_vla_lam30k_50k_sibal2_8gpu" \
  ./vla-scripts/bridge_lam_pipeline/bridge_factorized_pipeline.sh train-vla

env \
  ROOT="${ROOT}" \
  REPO="${REPO}" \
  UNIVLA_PIPELINE_PYTHON="${PYTHON_BIN}" \
  PYTHON="${PYTHON_BIN}" \
  WANDB_MODE=online \
  TOKENIZERS_PARALLELISM=false \
  VLA_CKPT_STEP=050000 \
  LOG_ROOT="${ROOT}/outputs/bridge_pipeline_logs/factorized_vla_lam30k_50k_sibal2_8gpu" \
  ./vla-scripts/bridge_lam_pipeline/bridge_factorized_pipeline.sh convert-hf
