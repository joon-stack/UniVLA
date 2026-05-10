#!/usr/bin/env bash
set -euo pipefail

ROOT=/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong
REPO=${ROOT}/UniVLA-hyper
PYTHON_BIN=${ROOT}/UniVLA/.venvs/sibal2/bin/python

cd "${REPO}"

COMMON_ENV=(
  ROOT="${ROOT}"
  REPO="${REPO}"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  NPROC_PER_NODE=8
  UNIVLA_PIPELINE_PYTHON="${PYTHON_BIN}"
  PYTHON="${PYTHON_BIN}"
  WANDB_MODE=online
  TOKENIZERS_PARALLELISM=false
  OMP_NUM_THREADS=8
  UNIVLA_PROFILE_STEPS=20
  UNIVLA_VLA_DATALOADER_WORKERS=0
  UNIVLA_TRAJ_THREADS=4
  UNIVLA_TRAJ_READ_THREADS=4
  UNIVLA_FRAME_THREADS=8
  UNIVLA_TF_RAM_BUDGET_MB=512
  LAM_KIND=visual_vq_factorized
  LAM_CONFIG_PATH="${REPO}/latent_action_model/config/lam-visual-vq-bridge-euclidean-plain-5tokens-k16-30k-8gpu.yaml"
  LAM_CKPT="${ROOT}/outputs/lam_bridge/logs/visual_vq_lam_bridge_euclidean_plain_5tokens_k16_30k_8gpu_b64_global512_workers2/epoch=0-step=30000.ckpt"
  LAM_CODEBOOK_SIZE=16
  LAM_LATENT_ACTION_TOKEN_LEN=5
  VLA_MAX_STEPS=50000
  VLA_SHUFFLE_BUFFER_SIZE=20000
  VLA_RUN_ROOT="${ROOT}/outputs/univla_bridge_euclidean_plain_5tokens_k16_lam30k_8gpu_g256"
  VLA_RUN_NOTE=bridge_dataset_euclidean_plain_5tokens_k16_lam30k_50k_8gpu_g256
  VLA_WANDB_PROJECT=univla_bridge_lam_local
  VLA_CKPT_STEP=050000
  HF_VLA_DIR="${ROOT}/outputs/hf_univla_bridge_euclidean_plain_5tokens_k16_lam30k_8gpu_g256_step050000"
  LOG_ROOT="${ROOT}/outputs/bridge_pipeline_logs/euclidean_plain_5tokens_k16_vla_lam30k_50k_sibal2_8gpu_g256"
)

env "${COMMON_ENV[@]}" \
  ./vla-scripts/bridge_lam_pipeline/bridge_factorized_pipeline.sh train-vla \
    --vla.expected_world_size 8 \
    --vla.per_device_batch_size 32 \
    --vla.global_batch_size 256

env "${COMMON_ENV[@]}" \
  ./vla-scripts/bridge_lam_pipeline/bridge_factorized_pipeline.sh convert-hf
