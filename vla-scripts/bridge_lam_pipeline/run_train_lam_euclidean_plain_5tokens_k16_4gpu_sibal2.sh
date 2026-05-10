#!/usr/bin/env bash
set -euo pipefail

ROOT=/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong
REPO=${ROOT}/UniVLA-hyper
PYTHON_BIN=${ROOT}/UniVLA/.venvs/sibal2/bin/python

RUN_NAME=visual_vq_lam_bridge_euclidean_plain_5tokens_k16_50k_4gpu_b64_acc2_global512

cd "${REPO}"

env \
  ROOT="${ROOT}" \
  REPO="${REPO}" \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  NPROC_PER_NODE=4 \
  LAM_NPROC_PER_NODE=4 \
  UNIVLA_PIPELINE_PYTHON="${PYTHON_BIN}" \
  PYTHON="${PYTHON_BIN}" \
  LAM_CONFIG=config/lam-visual-vq-bridge-euclidean-plain-5tokens-50k-4gpu.yaml \
  LAM_WANDB_NAME="${RUN_NAME}" \
  UNIVLA_RLDS_LEN_OVERRIDE=20000000 \
  WANDB_MODE=online \
  TOKENIZERS_PARALLELISM=false \
  OMP_NUM_THREADS=8 \
  UNIVLA_PROFILE_STEPS=20 \
  UNIVLA_LAM_PROFILE_STEPS=20 \
  UNIVLA_LAM_MODULE_PROFILE_STEPS=20 \
  UNIVLA_LAM_NUM_WORKERS=2 \
  UNIVLA_TRAJ_THREADS=4 \
  UNIVLA_TRAJ_READ_THREADS=4 \
  UNIVLA_FRAME_THREADS=8 \
  UNIVLA_TF_RAM_BUDGET_MB=512 \
  LOG_ROOT="${ROOT}/outputs/bridge_pipeline_logs/euclidean_plain_5tokens_k16_lam_50k_sibal2_4gpu_b64_acc2" \
  ./vla-scripts/bridge_lam_pipeline/bridge_factorized_pipeline.sh train-lam
