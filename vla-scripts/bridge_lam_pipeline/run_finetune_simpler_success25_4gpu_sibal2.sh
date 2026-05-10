#!/usr/bin/env bash
set -euo pipefail

cd /NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong/UniVLA-hyper

SIMPLER_DATA_ROOT=/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong/data/simpler_mvplam_noeval_success25_rlds

for task in carrot eggplant spoon stack; do
  if [[ ! -f "${SIMPLER_DATA_ROOT}/${task}/1.0.0/dataset_info.json" ]]; then
    echo "[error] Missing RLDS dataset_info.json for ${task}: ${SIMPLER_DATA_ROOT}/${task}/1.0.0" >&2
    exit 1
  fi
done

env \
  ROOT=/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong \
  SIMPLER_DATA_ROOT="${SIMPLER_DATA_ROOT}" \
  CUDA_VISIBLE_DEVICES=4,5,6,7 \
  NPROC_PER_NODE=4 \
  UNIVLA_PIPELINE_PYTHON=/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong/UniVLA/.venvs/sibal2/bin/python \
  PYTHON=/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong/UniVLA/.venvs/sibal2/bin/python \
  FINETUNE_BATCH_SIZE=32 \
  FINETUNE_MAX_STEPS=20000 \
  FINETUNE_SAVE_STEPS=5000 \
  FINETUNE_GRAD_ACCUMULATION_STEPS=1 \
  FINETUNE_RUN_ID_NOTE=simpler_success25_factorized_vla50k_4gpu_b32_acc1_sibal2_workers2_trainloop_lam \
  FINETUNE_MASTER_PORT=28762 \
  FINETUNE_RUN_ROOT=/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong/outputs/finetune_simpler_success25_factorized_vla50k_sibal2_workers2_4gpu \
  FINETUNE_ADAPTER_TMP_DIR=/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong/outputs/finetune_simpler_success25_factorized_vla50k_sibal2_workers2_4gpu_adapter_tmp \
  LOG_ROOT=/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong/outputs/bridge_pipeline_logs/factorized_simpler_success25_sibal2_workers2_4gpu \
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
  HF_UPLOAD_REPO_ID="${HF_UPLOAD_REPO_ID:-joon-stack/univla-simpler-success25-factorized-vla50k-step5000}" \
  HF_UPLOAD_PRIVATE="${HF_UPLOAD_PRIVATE:-0}" \
  HF_UPLOAD_DRY_RUN="${HF_UPLOAD_DRY_RUN:-0}" \
  HF_UPLOAD_LAM_FAMILY="${HF_UPLOAD_LAM_FAMILY:-factorized}" \
  HF_UPLOAD_NOTES="${HF_UPLOAD_NOTES:-UniVLA Simpler success25 factorized finetune.}" \
  ./vla-scripts/bridge_lam_pipeline/bridge_factorized_pipeline.sh finetune-simpler
