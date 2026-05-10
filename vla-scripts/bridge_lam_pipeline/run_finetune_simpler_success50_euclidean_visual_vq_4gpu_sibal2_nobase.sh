#!/usr/bin/env bash
set -euo pipefail

cd /NHNHOME/WORKSPACE/0526040036_A/user/01/youngjoonjeong/UniVLA-hyper




ROOT=/NHNHOME/WORKSPACE/0526040036_A/user/01/youngjoonjeong
REPO="${ROOT}/UniVLA-hyper"
PYTHON_BIN="${ROOT}/UniVLA/.venvs/sibal2/bin/python"
HF_VLA_DIR="${ROOT}/outputs/hf_univla_bridge_euclidean_visual_vq_lam30k_4gpu_b64_g256_step050000"
LAM_CKPT="${ROOT}/outputs/lam_bridge/logs/visual_vq_lam_bridge_euclidean_control_hmax9_50k_t2mid_t2future_metricgroups/epoch=0-step=30000.ckpt"
LAM_CONFIG_PATH="${REPO}/latent_action_model/config/lam-visual-vq-bridge-euclidean-control.yaml"
SIMPLER_DATA_ROOT=/NHNHOME/WORKSPACE/0526040036_A/user/01/youngjoonjeong/data/simpler_mvplam_noeval_success50_rlds

FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-32}"
FINETUNE_MAX_STEPS="${FINETUNE_MAX_STEPS:-10000}"
FINETUNE_SAVE_STEPS="${FINETUNE_SAVE_STEPS:-5000}"
FINETUNE_GRAD_ACCUMULATION_STEPS="${FINETUNE_GRAD_ACCUMULATION_STEPS:-1}"
FINETUNE_RUN_ID_NOTE="${FINETUNE_RUN_ID_NOTE:-simpler_success50_euclidean_visual_vq_vla50k_4gpu_b32_acc1_sibal2_workers2_trainloop_lam}"
FINETUNE_MASTER_PORT="${FINETUNE_MASTER_PORT:-28761}"
FINETUNE_RUN_ROOT="${FINETUNE_RUN_ROOT:-${ROOT}/outputs/finetune_simpler_success50_euclidean_visual_vq_vla50k_sibal2_workers2_4gpu}"
FINETUNE_ADAPTER_TMP_DIR="${FINETUNE_ADAPTER_TMP_DIR:-${ROOT}/outputs/finetune_simpler_success50_euclidean_visual_vq_vla50k_sibal2_workers2_4gpu_adapter_tmp}"
FINETUNE_SHUFFLE_BUFFER_SIZE="${FINETUNE_SHUFFLE_BUFFER_SIZE:-512}"
FINETUNE_IMAGE_AUG="${FINETUNE_IMAGE_AUG:-true}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/outputs/bridge_pipeline_logs/euclidean_visual_vq_simpler_success50_sibal2_workers2_4gpu}"

env \
  ROOT="${ROOT}" \
  REPO="${REPO}" \
  SIMPLER_DATA_ROOT="${SIMPLER_DATA_ROOT}" \
  HF_VLA_DIR="${HF_VLA_DIR}" \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  NPROC_PER_NODE=4 \
  UNIVLA_PIPELINE_PYTHON="${PYTHON_BIN}" \
  PYTHON="${PYTHON_BIN}" \
  LAM_KIND=visual_vq_factorized \
  LAM_CONFIG_PATH="${LAM_CONFIG_PATH}" \
  LAM_CKPT="${LAM_CKPT}" \
  LAM_CODEBOOK_SIZE=16 \
  LAM_LATENT_ACTION_TOKEN_LEN=4 \
  FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE}" \
  FINETUNE_MAX_STEPS="${FINETUNE_MAX_STEPS}" \
  FINETUNE_SAVE_STEPS="${FINETUNE_SAVE_STEPS}" \
  FINETUNE_GRAD_ACCUMULATION_STEPS="${FINETUNE_GRAD_ACCUMULATION_STEPS}" \
  FINETUNE_RUN_ID_NOTE="${FINETUNE_RUN_ID_NOTE}" \
  FINETUNE_MASTER_PORT="${FINETUNE_MASTER_PORT}" \
  FINETUNE_RUN_ROOT="${FINETUNE_RUN_ROOT}" \
  FINETUNE_ADAPTER_TMP_DIR="${FINETUNE_ADAPTER_TMP_DIR}" \
  LOG_ROOT="${LOG_ROOT}" \
  WANDB_MODE=online \
  TF_CPP_MIN_LOG_LEVEL=2 \
  UNIVLA_PROFILE_STEPS=20 \
  UNIVLA_FINETUNE_PROFILE_STEPS=20 \
  UNIVLA_TRAJ_THREADS=4 \
  UNIVLA_TRAJ_READ_THREADS=4 \
  UNIVLA_FRAME_THREADS=4 \
  UNIVLA_TF_RAM_BUDGET_MB=512 \
  UNIVLA_LAM_IN_TRAIN_LOOP=1 \
  UNIVLA_BATCH_LAM_IN_COLLATOR=1 \
  UNIVLA_VLA_DATALOADER_WORKERS=2 \
  UNIVLA_VLA_DATALOADER_PREFETCH_FACTOR=2 \
  FINETUNE_SHUFFLE_BUFFER_SIZE="${FINETUNE_SHUFFLE_BUFFER_SIZE}" \
  FINETUNE_IMAGE_AUG="${FINETUNE_IMAGE_AUG}" \
  HF_UPLOAD_ENABLED="${HF_UPLOAD_ENABLED:-0}" \
  HF_UPLOAD_APPROVED="${HF_UPLOAD_APPROVED:-0}" \
  HF_UPLOAD_REPO_ID="${HF_UPLOAD_REPO_ID:-joon-stack/univla-simpler-success50-euclidean-visual-vq-vla50k-nobase-step10000}" \
  HF_UPLOAD_PRIVATE="${HF_UPLOAD_PRIVATE:-0}" \
  HF_UPLOAD_DRY_RUN="${HF_UPLOAD_DRY_RUN:-0}" \
  HF_UPLOAD_LAM_FAMILY="${HF_UPLOAD_LAM_FAMILY:-euclidean_visual_vq}" \
  HF_UPLOAD_NOTES="${HF_UPLOAD_NOTES:-UniVLA Simpler success50 euclidean visual VQ nobase finetune.}" \
  ./vla-scripts/bridge_lam_pipeline/bridge_factorized_pipeline.sh finetune-simpler
