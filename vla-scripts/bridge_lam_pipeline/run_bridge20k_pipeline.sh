#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong}"
REPO="${REPO:-${ROOT}/UniVLA}"
NEW_PY="${NEW_PY:-${ROOT}/.venvs/univla-tfcheck-py310/bin/python}"

STAGE1_60K="${STAGE1_60K:-${ROOT}/outputs/lam_bridge/logs/task_centric_lam_stage1_bridge/epoch=1-step=60000.ckpt}"
LOCAL_STAGE2_CKPT="${LOCAL_STAGE2_CKPT:-${ROOT}/outputs/lam_bridge/logs/task_centric_lam_stage2_bridge/last.ckpt}"
OFFICIAL_STAGE2_CKPT="${OFFICIAL_STAGE2_CKPT:-${ROOT}/data/univla_checkpoints/univla-latent-action-model/lam-stage-2.ckpt}"
BASE_VLM="${BASE_VLM:-${ROOT}/data/univla_checkpoints/prismatic-vlms/prism-dinosiglip-224px+7b}"
BRIDGE_DATA="${BRIDGE_DATA:-${ROOT}/data/rlds_bridge_orig}"

RUN_ROOT="${RUN_ROOT:-${ROOT}/outputs/univla_bridge_lam_local}"
HF_VLA_DIR="${HF_VLA_DIR:-${ROOT}/outputs/hf_univla_bridge_lam_stage2_b200_tf21}"
LOG_DIR="${LOG_DIR:-${ROOT}/outputs/bridge20k_pipeline_logs}"
mkdir -p "${LOG_DIR}"

WANDB_ENTITY="${WANDB_ENTITY:-joonstack}"
WANDB_PROJECT_VLA="${WANDB_PROJECT_VLA:-univla_bridge_lam_local}"
WANDB_PROJECT_FT="${WANDB_PROJECT_FT:-univla_finetune_libero}"

export HF_HOME="${HF_HOME:-${ROOT}/data/hf_cache}"
export PYTHONPATH="${REPO}:${REPO}/latent_action_model:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"

NVIDIA_LIB_ROOT="${NEW_VENV_NVIDIA_LIB_ROOT:-${ROOT}/.venvs/univla-tfcheck-py310/lib/python3.10/site-packages/nvidia}"
export LD_LIBRARY_PATH="${NVIDIA_LIB_ROOT}/cublas/lib:${NVIDIA_LIB_ROOT}/cuda_cupti/lib:${NVIDIA_LIB_ROOT}/cuda_nvrtc/lib:${NVIDIA_LIB_ROOT}/cuda_runtime/lib:${NVIDIA_LIB_ROOT}/cudnn/lib:${NVIDIA_LIB_ROOT}/cufft/lib:${NVIDIA_LIB_ROOT}/curand/lib:${NVIDIA_LIB_ROOT}/cusolver/lib:${NVIDIA_LIB_ROOT}/cusparse/lib:${NVIDIA_LIB_ROOT}/nccl/lib:${NVIDIA_LIB_ROOT}/nvjitlink/lib:${LD_LIBRARY_PATH:-}"

usage() {
  cat <<'EOF'
Usage:
  run_bridge20k_pipeline.sh smoke-vla       # short new-venv VLA run for GPU util check
  run_bridge20k_pipeline.sh stage2-20k      # old-venv LAM stage2 from stage1 60k, max_steps=20000
  run_bridge20k_pipeline.sh vla             # new-venv Bridge VLA pretrain using local stage2 last.ckpt
  run_bridge20k_pipeline.sh convert         # convert latest VLA .pt checkpoint to HF format
  run_bridge20k_pipeline.sh finetune-libero # new-venv LIBERO finetune; requires LIBERO_DATA_ROOT
  run_bridge20k_pipeline.sh pipeline        # run stage2-20k then vla in one tmux session

Useful overrides:
  SMOKE_STEPS=20
  VLA_MAX_STEPS=20000
  LAM_STAGE2_STEPS=20000
  LIBERO_DATA_ROOT=/path/to/libero_rlds
  LIBERO_DATASET_NAME=libero_spatial_no_noops
EOF
}

require_file() {
  test -f "$1" || { echo "missing file: $1" >&2; exit 1; }
}

require_dir() {
  test -d "$1" || { echo "missing dir: $1" >&2; exit 1; }
}

run_tmux() {
  local session="$1"
  local command="$2"
  tmux new-session -d -s "${session}" "bash -lc '${command}'"
  echo "[started] tmux session: ${session}"
  echo "[logs] tmux capture-pane -pt ${session} -S -120"
}

common_new_env() {
  cat <<EOF
cd ${REPO} &&
export LD_LIBRARY_PATH='${LD_LIBRARY_PATH}' &&
export PYTHONPATH='${PYTHONPATH}' &&
export HF_HOME='${HF_HOME}' &&
export WANDB_MODE='${WANDB_MODE}' &&
export OMP_NUM_THREADS='\${OMP_NUM_THREADS:-8}' &&
export UNIVLA_TRAJ_THREADS='\${UNIVLA_TRAJ_THREADS:-4}' &&
export UNIVLA_TRAJ_READ_THREADS='\${UNIVLA_TRAJ_READ_THREADS:-4}' &&
export UNIVLA_FRAME_THREADS='\${UNIVLA_FRAME_THREADS:-8}' &&
export UNIVLA_TF_RAM_BUDGET_MB='\${UNIVLA_TF_RAM_BUDGET_MB:-512}' &&
export TF_FORCE_GPU_ALLOW_GROWTH=true
EOF
}

stage2_20k_cmd() {
  cat <<EOF
cd ${REPO} &&
test -f '${STAGE1_60K}' &&
UNIVLA_LAM_NUM_WORKERS='\${UNIVLA_LAM_NUM_WORKERS:-2}' \
UNIVLA_LAM_PREFETCH_FACTOR='\${UNIVLA_LAM_PREFETCH_FACTOR:-2}' \
WANDB_MODE='${WANDB_MODE}' \
./vla-scripts/bridge_lam_pipeline/01_train_lam_stage2_bridge.sh \
  --trainer.max_steps '\${LAM_STAGE2_STEPS:-20000}' \
  2>&1 | tee '${LOG_DIR}/lam_stage2_20k.log'
EOF
}

vla_train_cmd() {
  local lam_ckpt="$1"
  local run_note="$2"
  local max_steps="$3"
  local profile_steps="$4"
  cat <<EOF
$(common_new_env) &&
export UNIVLA_PROFILE_STEPS='${profile_steps}' &&
${NEW_PY} -m torch.distributed.run \
  --standalone --nproc_per_node 8 --master_port '\${MASTER_PORT:-28611}' \
  vla-scripts/train.py \
  --vla.type prism-dinosiglip-224px+mx-bridge \
  --vla.max_steps '${max_steps}' \
  --vla.shuffle_buffer_size '\${SHUFFLE_BUFFER_SIZE:-20000}' \
  --image_aug true \
  --pretrain_vlm '${BASE_VLM}' \
  --lam_path '${lam_ckpt}' \
  --data_root_dir '${BRIDGE_DATA}' \
  --run_root_dir '${RUN_ROOT}' \
  --wandb_project '${WANDB_PROJECT_VLA}' \
  --wandb_entity '${WANDB_ENTITY}' \
  --run_id_note '${run_note}' \
  2>&1 | tee '${LOG_DIR}/${run_note}.log'
EOF
}

vla_run_dir() {
  local run_note="$1"
  echo "${RUN_ROOT}/prism-dinosiglip-224px+mx-bridge+n1+b32+x42--${run_note}--image_aug-Latent-Action-Pretraining"
}

case "${1:-}" in
  smoke-vla)
    require_file "${NEW_PY}"
    require_file "${OFFICIAL_STAGE2_CKPT}"
    require_file "${BASE_VLM}/checkpoints/latest-checkpoint.pt"
    require_dir "${BRIDGE_DATA}/bridge_dataset/1.0.0"
    run_tmux "vla_smoke_newtf" "$(vla_train_cmd "${OFFICIAL_STAGE2_CKPT}" "vla_smoke_tf21_util" "${SMOKE_STEPS:-20}" "${UNIVLA_PROFILE_STEPS:-5}")"
    ;;

  stage2-20k)
    require_file "${STAGE1_60K}"
    require_dir "${BRIDGE_DATA}/bridge_dataset/1.0.0"
    run_tmux "lam_bridge_stage2_20k_old" "$(stage2_20k_cmd)"
    ;;

  vla)
    require_file "${NEW_PY}"
    require_file "${LOCAL_STAGE2_CKPT}"
    require_file "${BASE_VLM}/checkpoints/latest-checkpoint.pt"
    require_dir "${BRIDGE_DATA}/bridge_dataset/1.0.0"
    run_tmux "vla_bridge_newtf" "$(vla_train_cmd "${LOCAL_STAGE2_CKPT}" "bridge_dataset_local_lam_stage2_20k_b200_tf21" "${VLA_MAX_STEPS:-20000}" "${UNIVLA_PROFILE_STEPS:-0}")"
    ;;

  convert)
    run_note="${VLA_RUN_NOTE:-bridge_dataset_local_lam_stage2_20k_b200_tf21}"
    run_dir="${VLA_RUN_DIR:-$(vla_run_dir "${run_note}")}"
    require_file "${NEW_PY}"
    require_dir "${run_dir}/checkpoints"
    ckpt_name="${VLA_CKPT_NAME:-$(basename "$(ls -t "${run_dir}"/checkpoints/*.pt | head -1)")}"
    cd "${REPO}"
    "${NEW_PY}" vla-scripts/extern/convert_univla_weights_to_hf.py \
      --openvla_model_path_or_id "${run_dir}" \
      --ckpt_name "${ckpt_name}" \
      --output_hf_model_local_path "${HF_VLA_DIR}"
    ;;

  finetune-libero)
    require_file "${NEW_PY}"
    require_dir "${HF_VLA_DIR}"
    require_file "${LOCAL_STAGE2_CKPT}"
    test -n "${LIBERO_DATA_ROOT:-}" || { echo "set LIBERO_DATA_ROOT=/path/to/libero_rlds" >&2; exit 1; }
    run_tmux "finetune_libero_newtf" "$(common_new_env) &&
${NEW_PY} -m torch.distributed.run \
  --standalone --nproc_per_node 8 --master_port '\${MASTER_PORT:-28621}' \
  vla-scripts/finetune_libero.py \
  --vla_path '${HF_VLA_DIR}' \
  --lam_path '${LOCAL_STAGE2_CKPT}' \
  --data_root_dir '${LIBERO_DATA_ROOT}' \
  --dataset_name '${LIBERO_DATASET_NAME:-libero_spatial_no_noops}' \
  --run_root_dir '${ROOT}/outputs/finetune_libero' \
  --wandb_project '${WANDB_PROJECT_FT}' \
  --wandb_entity '${WANDB_ENTITY}' \
  --run_id_note 'bridge_lam_stage2_20k_b200_tf21' \
  --batch_size '\${FT_BATCH_SIZE:-8}' \
  --max_steps '\${FT_MAX_STEPS:-30000}' \
  --save_steps '\${FT_SAVE_STEPS:-30000}' \
  2>&1 | tee '${LOG_DIR}/finetune_libero.log'"
    ;;

  pipeline)
    require_file "${STAGE1_60K}"
    require_file "${NEW_PY}"
    require_file "${BASE_VLM}/checkpoints/latest-checkpoint.pt"
    require_dir "${BRIDGE_DATA}/bridge_dataset/1.0.0"
    run_tmux "bridge20k_pipeline" "$(stage2_20k_cmd) && $(vla_train_cmd "${LOCAL_STAGE2_CKPT}" "bridge_dataset_local_lam_stage2_20k_b200_tf21" "${VLA_MAX_STEPS:-20000}" "${UNIVLA_PROFILE_STEPS:-0}")"
    ;;

  ""|-h|--help|help)
    usage
    ;;

  *)
    echo "unknown command: $1" >&2
    usage >&2
    exit 2
    ;;
esac
