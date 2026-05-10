#!/usr/bin/env bash
set -euo pipefail

ROOT=/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong
REPO=${ROOT}/UniVLA-hyper
PYTHON_BIN=${ROOT}/UniVLA/.venvs/sibal2/bin/python

cd "${REPO}"

env \
  ROOT="${ROOT}" \
  REPO="${REPO}" \
  SIMPLER_DATA_ROOT=/tmp/youngjoon_univla_data/simpler_mvplam_noeval_success50_rlds \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  NPROC_PER_NODE=4 \
  UNIVLA_PIPELINE_PYTHON="${PYTHON_BIN}" \
  PYTHON="${PYTHON_BIN}" \
  LAM_CKPT="${ROOT}/outputs/lam_bridge/logs/visual_vq_lam_bridge_hyperbolic_factorized_rad16_0to4_dir16_rad1_dir4tokens_prelift_hmax9_50k_workers2/epoch=0-step=30000.ckpt" \
  LAM_KIND=visual_vq_factorized \
  LAM_CONFIG_PATH="${REPO}/latent_action_model/config/lam-visual-vq-bridge-hyperbolic-factorized-50k.yaml" \
  LAM_TOKEN_VIEW=factorized_direction \
  LAM_CODEBOOK_SIZE=16 \
  LAM_LATENT_ACTION_TOKEN_LEN=4 \
  HF_VLA_DIR="${ROOT}/outputs/hf_univla_bridge_factorized_dironly_lam30k_step050000" \
  FINETUNE_BATCH_SIZE=32 \
  FINETUNE_MAX_STEPS=20000 \
  FINETUNE_SAVE_STEPS=5000 \
  FINETUNE_GRAD_ACCUMULATION_STEPS=1 \
  FINETUNE_RUN_ID_NOTE=simpler_success50_factorized_dironly_vla50k_4gpu_b32_acc1_sibal2_workers2_trainloop_lam \
  FINETUNE_MASTER_PORT=28762 \
  FINETUNE_RUN_ROOT="${ROOT}/outputs/finetune_simpler_success50_factorized_dironly_vla50k_sibal2_workers2_4gpu" \
  FINETUNE_ADAPTER_TMP_DIR="${ROOT}/outputs/finetune_simpler_success50_factorized_dironly_vla50k_sibal2_workers2_4gpu_adapter_tmp" \
  LOG_ROOT="${ROOT}/outputs/bridge_pipeline_logs/factorized_dironly_simpler_success50_sibal2_workers2_4gpu" \
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
  FINETUNE_SHUFFLE_BUFFER_SIZE=512 \
  FINETUNE_IMAGE_AUG=true \
  HF_UPLOAD_ENABLED="${HF_UPLOAD_ENABLED:-0}" \
  HF_UPLOAD_APPROVED="${HF_UPLOAD_APPROVED:-0}" \
  HF_UPLOAD_REPO_ID="${HF_UPLOAD_REPO_ID:-joon-stack/univla-simpler-success50-factorized-dironly-vla50k-step5000}" \
  HF_UPLOAD_PRIVATE="${HF_UPLOAD_PRIVATE:-0}" \
  HF_UPLOAD_DRY_RUN="${HF_UPLOAD_DRY_RUN:-0}" \
  HF_UPLOAD_LAM_FAMILY="${HF_UPLOAD_LAM_FAMILY:-factorized_dironly}" \
  HF_UPLOAD_NOTES="${HF_UPLOAD_NOTES:-UniVLA Simpler success50 factorized direction-only finetune.}" \
  ./vla-scripts/bridge_lam_pipeline/bridge_factorized_pipeline.sh finetune-simpler
