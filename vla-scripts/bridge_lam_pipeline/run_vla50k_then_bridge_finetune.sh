#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong}"
REPO="${REPO:-${ROOT}/UniVLA}"
NEW_VENV="${NEW_VENV:-${ROOT}/.venvs/univla-tfcheck-py310}"

STAGE2_CKPT="${STAGE2_CKPT:-${ROOT}/outputs/lam_bridge/logs/task_centric_lam_stage2_bridge/last.ckpt}"
BRIDGE_DATA="${BRIDGE_DATA:-${ROOT}/data/rlds_bridge_orig}"
BASE_VLM="${BASE_VLM:-${ROOT}/data/univla_checkpoints/prismatic-vlms/prism-dinosiglip-224px+7b}"

RUN_ROOT="${RUN_ROOT:-${ROOT}/outputs/univla_bridge_lam_local}"
RUN_NOTE="${RUN_NOTE:-bridge_dataset_local_lam_stage2_b200_tf21_vla50k_cleanenv}"
HF_VLA_DIR="${HF_VLA_DIR:-${ROOT}/outputs/hf_univla_bridge_lam_stage2_b200_tf21_vla50k_cleanenv}"
FINETUNE_RUN_ROOT="${FINETUNE_RUN_ROOT:-${ROOT}/outputs/finetune_bridge}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/outputs/bridge_pipeline_logs/vla50k_then_finetune_20260503_2030}"

WANDB_ENTITY="${WANDB_ENTITY:-joonstack}"
WANDB_MODE="${WANDB_MODE:-online}"
VLA_MASTER_PORT="${VLA_MASTER_PORT:-28611}"
FINETUNE_MASTER_PORT="${FINETUNE_MASTER_PORT:-28621}"

NVIDIA_SITE="${NEW_VENV}/lib/python3.10/site-packages/nvidia"
NEW_TF_LD_PATH="${NVIDIA_SITE}/cublas/lib:${NVIDIA_SITE}/cuda_cupti/lib:${NVIDIA_SITE}/cuda_nvrtc/lib:${NVIDIA_SITE}/cuda_runtime/lib:${NVIDIA_SITE}/cudnn/lib:${NVIDIA_SITE}/cufft/lib:${NVIDIA_SITE}/curand/lib:${NVIDIA_SITE}/cusolver/lib:${NVIDIA_SITE}/cusparse/lib:${NVIDIA_SITE}/nccl/lib:${NVIDIA_SITE}/nvjitlink/lib"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[missing file] ${path}" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "[missing dir] ${path}" >&2
    exit 1
  fi
}

require_file "${NEW_VENV}/bin/python"
require_file "${BASE_VLM}/checkpoints/latest-checkpoint.pt"
require_file "${STAGE2_CKPT}"
require_file "${BRIDGE_DATA}/bridge_dataset/1.0.0/dataset_info.json"

mkdir -p "${RUN_ROOT}/run_logs" "${LOG_ROOT}"
cd "${REPO}"

unset UNIVLA_RLDS_LEN_OVERRIDE
export HF_HOME="${HF_HOME:-${ROOT}/data/hf_cache}"
export PYTHONPATH="${REPO}:${REPO}/latent_action_model:${PYTHONPATH:-}"
export WANDB_MODE
export UNIVLA_DUMMY_LATENT_ACTIONS=0
export UNIVLA_TRAJ_THREADS="${UNIVLA_TRAJ_THREADS:-4}"
export UNIVLA_TRAJ_READ_THREADS="${UNIVLA_TRAJ_READ_THREADS:-4}"
export UNIVLA_FRAME_THREADS="${UNIVLA_FRAME_THREADS:-8}"
export UNIVLA_TF_RAM_BUDGET_MB="${UNIVLA_TF_RAM_BUDGET_MB:-512}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TF_FORCE_GPU_ALLOW_GROWTH=true

echo "[chain] VLA 50k start: ${RUN_NOTE}"
LD_LIBRARY_PATH="${NEW_TF_LD_PATH}:${LD_LIBRARY_PATH:-}" \
"${NEW_VENV}/bin/python" -m torch.distributed.run \
  --standalone \
  --nproc_per_node 8 \
  --master_port "${VLA_MASTER_PORT}" \
  vla-scripts/train.py \
  --vla.type prism-dinosiglip-224px+mx-bridge \
  --vla.max_steps 50000 \
  --vla.shuffle_buffer_size 20000 \
  --image_aug true \
  --pretrain_vlm "${BASE_VLM}" \
  --lam_path "${STAGE2_CKPT}" \
  --data_root_dir "${BRIDGE_DATA}" \
  --run_root_dir "${RUN_ROOT}" \
  --wandb_project univla_bridge_lam_local \
  --wandb_entity "${WANDB_ENTITY}" \
  --run_id_note "${RUN_NOTE}" \
  2>&1 | tee "${RUN_ROOT}/run_logs/vla_${RUN_NOTE}.log"

VLA_RUN_DIR="${RUN_ROOT}/prism-dinosiglip-224px+mx-bridge+n1+b32+x42--${RUN_NOTE}--image_aug-Latent-Action-Pretraining"
require_dir "${VLA_RUN_DIR}/checkpoints"
VLA_CKPT_NAME="$(find "${VLA_RUN_DIR}/checkpoints" -maxdepth 1 -type f -name '*.pt' -printf '%T@ %f\n' | sort -nr | awk 'NR == 1 { print $2 }')"
if [[ -z "${VLA_CKPT_NAME}" ]]; then
  echo "[missing checkpoint] ${VLA_RUN_DIR}/checkpoints/*.pt" >&2
  exit 1
fi

echo "[chain] convert VLA checkpoint: ${VLA_CKPT_NAME}"
"${NEW_VENV}/bin/python" vla-scripts/extern/convert_univla_weights_to_hf.py \
  --openvla_model_path_or_id "${VLA_RUN_DIR}" \
  --ckpt_name "${VLA_CKPT_NAME}" \
  --output_hf_model_local_path "${HF_VLA_DIR}" \
  2>&1 | tee "${LOG_ROOT}/convert_to_hf.log"

echo "[chain] Bridge finetune start"
LD_LIBRARY_PATH="${NEW_TF_LD_PATH}:${LD_LIBRARY_PATH:-}" \
"${NEW_VENV}/bin/python" -m torch.distributed.run \
  --standalone \
  --nproc_per_node 8 \
  --master_port "${FINETUNE_MASTER_PORT}" \
  vla-scripts/finetune_bridge.py \
  --vla_path "${HF_VLA_DIR}" \
  --lam_path "${STAGE2_CKPT}" \
  --data_root_dir "${BRIDGE_DATA}" \
  --dataset_name bridge \
  --run_root_dir "${FINETUNE_RUN_ROOT}" \
  --wandb_project univla_finetune_libero \
  --wandb_entity "${WANDB_ENTITY}" \
  --run_id_note bridge_lam_stage2_b200_tf21_vla50k_cleanenv \
  --batch_size "${FINETUNE_BATCH_SIZE:-8}" \
  --max_steps "${FINETUNE_MAX_STEPS:-30000}" \
  --save_steps "${FINETUNE_SAVE_STEPS:-30000}" \
  2>&1 | tee "${LOG_ROOT}/finetune_bridge.log"

echo "[chain] all done"
