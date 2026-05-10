#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bridge_factorized_pipeline.sh <stage> [extra args...]

Stages:
  train-lam          Train factorized hyperbolic visual VQ LAM.
  train-vla          Train Bridge VLA with the factorized LAM checkpoint.
  convert-hf         Convert a VLA .pt checkpoint to HF format.
  finetune-simpler   Finetune the converted VLA on Simpler success50.

Common overrides are environment variables, for example:
  VLA_MAX_STEPS=50000 VLA_CKPT_STEP=050000 ./.../bridge_factorized_pipeline.sh convert-hf
  FINETUNE_BATCH_SIZE=16 FINETUNE_GRAD_ACCUMULATION_STEPS=1 ./.../bridge_factorized_pipeline.sh finetune-simpler
  HF_UPLOAD_ENABLED=1 HF_UPLOAD_APPROVED=1 HF_UPLOAD_REPO_ID=owner/repo ./.../bridge_factorized_pipeline.sh finetune-simpler
USAGE
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

STAGE="$1"
shift

ROOT="${ROOT:-/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong}"
REPO="${REPO:-${ROOT}/UniVLA-hyper}"
PYTHON="${UNIVLA_PIPELINE_PYTHON:-${PYTHON:-${ROOT}/UniVLA/.venvs/sibal2/bin/python}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

WANDB_ENTITY="${WANDB_ENTITY:-joonstack}"
WANDB_MODE="${WANDB_MODE:-online}"

BRIDGE_DATA_ROOT="${BRIDGE_DATA_ROOT:-${ROOT}/data/rlds_bridge_orig}"
SIMPLER_DATA_ROOT="${SIMPLER_DATA_ROOT:-${ROOT}/data/simpler_mvplam_noeval_success50_rlds}"
BASE_VLM="${BASE_VLM:-${ROOT}/data/univla_checkpoints/prismatic-vlms/prism-dinosiglip-224px+7b}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${ROOT}/data/hf_cache/hub/models--NousResearch--Llama-2-7b-hf/snapshots/8efe6c9b93655b934e27bd9981e3ec13e55aee9d}"

LAM_CONFIG="${LAM_CONFIG:-config/lam-visual-vq-bridge-hyperbolic-factorized-50k.yaml}"
LAM_CONFIG_PATH="${LAM_CONFIG_PATH:-${REPO}/latent_action_model/config/lam-visual-vq-bridge-hyperbolic-factorized-rad0to4-50k.yaml}"
LAM_CKPT="${LAM_CKPT:-${ROOT}/outputs/lam_bridge/logs/visual_vq_lam_bridge_hyperbolic_factorized_rad16_0to4_dir16_rad1_dir4tokens_prelift_hmax9_50k_workers2/epoch=0-step=30000.ckpt}"
LAM_KIND="${LAM_KIND:-visual_vq_factorized}"
LAM_CODEBOOK_SIZE="${LAM_CODEBOOK_SIZE:-32}"
LAM_LATENT_ACTION_TOKEN_LEN="${LAM_LATENT_ACTION_TOKEN_LEN:-5}"
LAM_TOKEN_VIEW="${LAM_TOKEN_VIEW:-indices}"
LAM_WANDB_PROJECT="${LAM_WANDB_PROJECT:-univla_lam_bridge}"
LAM_WANDB_NAME="${LAM_WANDB_NAME:-visual_vq_lam_bridge_hyperbolic_factorized_rad16_0to6_dir16_rad1_dir4tokens_prelift_hmax9_50k_workers2}"

VLA_RUN_ROOT="${VLA_RUN_ROOT:-${ROOT}/outputs/univla_bridge_factorized_lam30k}"
VLA_RUN_NOTE="${VLA_RUN_NOTE:-bridge_dataset_factorized_lam30k_rad16dir16_50k}"
VLA_RUN_DIR="${VLA_RUN_DIR:-${VLA_RUN_ROOT}/prism-dinosiglip-224px+mx-bridge+n1+b32+x42--${VLA_RUN_NOTE}--image_aug-Latent-Action-Pretraining}"
VLA_MAX_STEPS="${VLA_MAX_STEPS:-50000}"
VLA_SHUFFLE_BUFFER_SIZE="${VLA_SHUFFLE_BUFFER_SIZE:-20000}"
VLA_WANDB_PROJECT="${VLA_WANDB_PROJECT:-univla_bridge_lam_local}"

VLA_CKPT_STEP="${VLA_CKPT_STEP:-050000}"
VLA_CKPT_NAME="${VLA_CKPT_NAME:-}"
HF_VLA_DIR="${HF_VLA_DIR:-${ROOT}/outputs/hf_univla_bridge_factorized_lam30k_rad16dir16_50k_step${VLA_CKPT_STEP}}"
HF_VLA_TMP="${HF_VLA_TMP:-${HF_VLA_DIR}.tmp}"

FINETUNE_RUN_ROOT="${FINETUNE_RUN_ROOT:-${ROOT}/outputs/finetune_simpler_success50_factorized_lam30k_vla50k_ws10_b16_acc1_8gpu}"
FINETUNE_ADAPTER_TMP_DIR="${FINETUNE_ADAPTER_TMP_DIR:-${ROOT}/outputs/finetune_simpler_success50_factorized_lam30k_vla50k_ws10_b16_acc1_8gpu_adapter_tmp}"
FINETUNE_RUN_ID_NOTE="${FINETUNE_RUN_ID_NOTE:-simpler_success50_factorized_lam30k_vla50k_step050000_10k_ws10_b16_acc1_8gpu}"
FINETUNE_WANDB_PROJECT="${FINETUNE_WANDB_PROJECT:-univla_finetune_simpler}"
FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-16}"
FINETUNE_GRAD_ACCUMULATION_STEPS="${FINETUNE_GRAD_ACCUMULATION_STEPS:-1}"
FINETUNE_LEARNING_RATE="${FINETUNE_LEARNING_RATE:-0.00035}"
FINETUNE_USE_SCHEDULER="${FINETUNE_USE_SCHEDULER:-false}"
FINETUNE_MAX_STEPS="${FINETUNE_MAX_STEPS:-20000}"
FINETUNE_SAVE_STEPS="${FINETUNE_SAVE_STEPS:-5000}"
FINETUNE_WINDOW_SIZE="${FINETUNE_WINDOW_SIZE:-10}"
FINETUNE_SHUFFLE_BUFFER_SIZE="${FINETUNE_SHUFFLE_BUFFER_SIZE:-1024}"
FINETUNE_IMAGE_AUG="${FINETUNE_IMAGE_AUG:-true}"
FINETUNE_MASTER_PORT="${FINETUNE_MASTER_PORT:-28761}"
VLA_MASTER_PORT="${VLA_MASTER_PORT:-28741}"

LOG_ROOT="${LOG_ROOT:-${ROOT}/outputs/bridge_pipeline_logs/reusable_factorized_pipeline}"

HF_UPLOAD_ENABLED="${HF_UPLOAD_ENABLED:-0}"
HF_UPLOAD_APPROVED="${HF_UPLOAD_APPROVED:-0}"
HF_UPLOAD_REPO_ID="${HF_UPLOAD_REPO_ID:-}"
HF_UPLOAD_PRIVATE="${HF_UPLOAD_PRIVATE:-0}"
HF_UPLOAD_DRY_RUN="${HF_UPLOAD_DRY_RUN:-0}"
HF_UPLOAD_SCRIPT="${HF_UPLOAD_SCRIPT:-/home/snu_wnsd/.codex/skills/hf-checkpoint-uploader/scripts/upload_hf_checkpoint.py}"
HF_UPLOAD_TOKENIZER_MODEL="${HF_UPLOAD_TOKENIZER_MODEL:-${TOKENIZER_PATH}/tokenizer.model}"
HF_UPLOAD_LAM_FAMILY="${HF_UPLOAD_LAM_FAMILY:-${LAM_KIND}}"
HF_UPLOAD_NOTES="${HF_UPLOAD_NOTES:-UniVLA Simpler finetune: ${FINETUNE_RUN_ID_NOTE}}"
HF_UPLOAD_COMMIT_MESSAGE="${HF_UPLOAD_COMMIT_MESSAGE:-Upload UniVLA Simpler finetune export}"

NVIDIA_SITE="$("${PYTHON}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"] + "/nvidia")')"
NVIDIA_LD_PATH=""
if [[ -d "${NVIDIA_SITE}" ]]; then
  for package_dir in "${NVIDIA_SITE}"/*; do
    if [[ -d "${package_dir}/lib" ]]; then
      NVIDIA_LD_PATH="${NVIDIA_LD_PATH:+${NVIDIA_LD_PATH}:}${package_dir}/lib"
    fi
  done
fi

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

set_common_env() {
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  export HF_HOME="${HF_HOME:-${ROOT}/data/hf_cache}"
  export PYTHONPATH="${REPO}:${REPO}/latent_action_model:${PYTHONPATH:-}"
  export WANDB_MODE
  export TOKENIZERS_PARALLELISM=false
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
  export TF_FORCE_GPU_ALLOW_GROWTH=true
  export TORCH_HOME="${TORCH_HOME:-${ROOT}/torch_hub_cache_dinov2}"
  export UNIVLA_DUMMY_LATENT_ACTIONS="${UNIVLA_DUMMY_LATENT_ACTIONS:-0}"
  export UNIVLA_PROFILE_STEPS="${UNIVLA_PROFILE_STEPS:-20}"
  export UNIVLA_TRAJ_THREADS="${UNIVLA_TRAJ_THREADS:-4}"
  export UNIVLA_TRAJ_READ_THREADS="${UNIVLA_TRAJ_READ_THREADS:-4}"
  export UNIVLA_FRAME_THREADS="${UNIVLA_FRAME_THREADS:-8}"
  export UNIVLA_TF_RAM_BUDGET_MB="${UNIVLA_TF_RAM_BUDGET_MB:-512}"
  if [[ -n "${NVIDIA_LD_PATH}" ]]; then
    export LD_LIBRARY_PATH="${NVIDIA_LD_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
}

set_finetune_env() {
  set_common_env
  export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
  export UNIVLA_BATCH_LAM_IN_COLLATOR="${UNIVLA_BATCH_LAM_IN_COLLATOR:-1}"
  export UNIVLA_VLA_DATALOADER_WORKERS="${UNIVLA_VLA_DATALOADER_WORKERS:-0}"
  export UNIVLA_TF_DISABLE_GPU="${UNIVLA_TF_DISABLE_GPU:-1}"
  export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
  export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
  export UNIVLA_FRAME_THREADS="${UNIVLA_FRAME_THREADS:-4}"
  export UNIVLA_TF_RAM_BUDGET_MB="${UNIVLA_TF_RAM_BUDGET_MB:-512}"
}

resolve_vla_ckpt_name() {
  if [[ -n "${VLA_CKPT_NAME}" ]]; then
    echo "${VLA_CKPT_NAME}"
    return
  fi
  find "${VLA_RUN_DIR}/checkpoints" -maxdepth 1 -type f -name "step-${VLA_CKPT_STEP}-*.pt" -printf '%f\n' | sort | tail -n 1
}

resolve_finetune_export_dir() {
  if [[ -n "${FINETUNE_HF_EXPORT_DIR:-}" ]]; then
    echo "${FINETUNE_HF_EXPORT_DIR}"
    return
  fi

  local candidate
  candidate="$(
    find "${FINETUNE_RUN_ROOT}" -mindepth 1 -maxdepth 1 -type d \
      -name "*--${FINETUNE_RUN_ID_NOTE}*" \
      -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
  )"
  if [[ -z "${candidate}" ]]; then
    candidate="$(
      find "${FINETUNE_RUN_ROOT}" -mindepth 1 -maxdepth 1 -type d \
        -name "*+simpler+*" \
        -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
    )"
  fi
  if [[ -z "${candidate}" ]]; then
    echo "[hf-upload] no finetune export dir found under ${FINETUNE_RUN_ROOT}" >&2
    return 1
  fi
  echo "${candidate}"
}

upload_finetune_export_to_hf() {
  if [[ "${HF_UPLOAD_ENABLED}" != "1" ]]; then
    echo "[hf-upload] skipped because HF_UPLOAD_ENABLED=${HF_UPLOAD_ENABLED}"
    return
  fi
  if [[ -z "${HF_UPLOAD_REPO_ID}" ]]; then
    echo "[hf-upload] set HF_UPLOAD_REPO_ID=owner/repo-name to enable upload" >&2
    return 1
  fi
  if [[ "${HF_UPLOAD_DRY_RUN}" != "1" && "${HF_UPLOAD_DRY_RUN}" != "true" && "${HF_UPLOAD_APPROVED}" != "1" ]]; then
    echo "[hf-upload] refusing external upload without HF_UPLOAD_APPROVED=1" >&2
    echo "[hf-upload] repo=${HF_UPLOAD_REPO_ID} private=${HF_UPLOAD_PRIVATE}" >&2
    return 1
  fi

  local export_dir
  export_dir="$(resolve_finetune_export_dir)"
  require_file "${HF_UPLOAD_SCRIPT}"
  require_file "${export_dir}/config.json"
  require_file "${export_dir}/tokenizer_config.json"
  if [[ ! -f "${export_dir}/model.safetensors.index.json" ]] || ! compgen -G "${export_dir}/model-*.safetensors" >/dev/null; then
    echo "[hf-upload] missing HF safetensors export in ${export_dir}" >&2
    return 1
  fi
  if ! compgen -G "${export_dir}/action_decoder-*.pt" >/dev/null; then
    echo "[hf-upload] missing action_decoder-*.pt in ${export_dir}" >&2
    return 1
  fi

  local upload_args=(
    "${HF_UPLOAD_SCRIPT}"
    --folder "${export_dir}"
    --repo-id "${HF_UPLOAD_REPO_ID}"
    --pred-action-horizon "${FINETUNE_WINDOW_SIZE}"
    --latent-action-token-len "${LAM_LATENT_ACTION_TOKEN_LEN}"
    --codebook-size "${LAM_CODEBOOK_SIZE}"
    --base-vla "$(basename "${HF_VLA_DIR}")"
    --lam-family "${HF_UPLOAD_LAM_FAMILY}"
    --notes "${HF_UPLOAD_NOTES}"
    --commit-message "${HF_UPLOAD_COMMIT_MESSAGE}"
  )
  if [[ -f "${HF_UPLOAD_TOKENIZER_MODEL}" ]]; then
    upload_args+=(--tokenizer-model "${HF_UPLOAD_TOKENIZER_MODEL}")
  fi
  if [[ "${HF_UPLOAD_PRIVATE}" == "1" || "${HF_UPLOAD_PRIVATE}" == "true" ]]; then
    upload_args+=(--private)
  fi
  if [[ "${HF_UPLOAD_DRY_RUN}" == "1" || "${HF_UPLOAD_DRY_RUN}" == "true" ]]; then
    upload_args+=(--dry-run)
  else
    upload_args+=(--yes)
  fi

  echo "[hf-upload] repo=${HF_UPLOAD_REPO_ID}"
  echo "[hf-upload] folder=${export_dir}"
  "${PYTHON}" "${upload_args[@]}" 2>&1 | tee "${LOG_ROOT}/hf_upload.log"
}

print_context() {
  echo "[context] stage=${STAGE}"
  echo "[context] repo=${REPO}"
  echo "[context] python=${PYTHON}"
  echo "[context] cuda=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  echo "[context] lam_token_view=${LAM_TOKEN_VIEW}"
  echo "[context] log_root=${LOG_ROOT}"
}

mkdir -p "${LOG_ROOT}"
require_file "${PYTHON}"
require_dir "${REPO}"

case "${STAGE}" in
  train-lam)
    set_common_env
    export UNIVLA_RLDS_LEN_OVERRIDE="${UNIVLA_RLDS_LEN_OVERRIDE:-20000000}"
    require_file "${REPO}/latent_action_model/${LAM_CONFIG}"
    cd "${REPO}/latent_action_model"
    print_context
    echo "[train-lam] config=${LAM_CONFIG}"
    "${PYTHON}" -m torch.distributed.run \
      --standalone \
      --nnodes 1 \
      --nproc-per-node "${NPROC_PER_NODE}" \
      main_visual_vq.py fit \
      --config "${LAM_CONFIG}" \
      --trainer.logger "[{\"class_path\":\"lightning.pytorch.loggers.WandbLogger\",\"init_args\":{\"project\":\"${LAM_WANDB_PROJECT}\",\"entity\":\"${WANDB_ENTITY}\",\"name\":\"${LAM_WANDB_NAME}\"}}]" \
      "$@" \
      2>&1 | tee "${LOG_ROOT}/train_lam.log"
    ;;

  train-vla)
    set_common_env
    unset UNIVLA_RLDS_LEN_OVERRIDE
    require_file "${BASE_VLM}/checkpoints/latest-checkpoint.pt"
    require_file "${LAM_CKPT}"
    require_file "${LAM_CONFIG_PATH}"
    require_file "${BRIDGE_DATA_ROOT}/bridge_orig/1.0.0/dataset_info.json"
    cd "${REPO}"
    print_context
    echo "[train-vla] lam=${LAM_CKPT}"
    "${PYTHON}" -m torch.distributed.run \
      --standalone \
      --nnodes 1 \
      --nproc-per-node "${NPROC_PER_NODE}" \
      --master_port "${VLA_MASTER_PORT}" \
      vla-scripts/train.py \
      --vla.type prism-dinosiglip-224px+mx-bridge \
      --vla.max_steps "${VLA_MAX_STEPS}" \
      --vla.shuffle_buffer_size "${VLA_SHUFFLE_BUFFER_SIZE}" \
      --image_aug true \
      --pretrain_vlm "${BASE_VLM}" \
      --lam_path "${LAM_CKPT}" \
      --lam_kind "${LAM_KIND}" \
      --lam_config_path "${LAM_CONFIG_PATH}" \
      --lam_token_view "${LAM_TOKEN_VIEW}" \
      --codebook_size "${LAM_CODEBOOK_SIZE}" \
      --latent_action_token_len "${LAM_LATENT_ACTION_TOKEN_LEN}" \
      --data_root_dir "${BRIDGE_DATA_ROOT}" \
      --run_root_dir "${VLA_RUN_ROOT}" \
      --wandb_project "${VLA_WANDB_PROJECT}" \
      --wandb_entity "${WANDB_ENTITY}" \
      --run_id_note "${VLA_RUN_NOTE}" \
      "$@" \
      2>&1 | tee "${LOG_ROOT}/train_vla.log"
    ;;

  convert-hf)
    set_common_env
    require_dir "${VLA_RUN_DIR}"
    require_dir "${VLA_RUN_DIR}/checkpoints"
    require_file "${TOKENIZER_PATH}/tokenizer_config.json"
    VLA_CKPT_NAME="$(resolve_vla_ckpt_name)"
    if [[ -z "${VLA_CKPT_NAME}" ]]; then
      echo "[missing checkpoint] ${VLA_RUN_DIR}/checkpoints/step-${VLA_CKPT_STEP}-*.pt" >&2
      exit 1
    fi
    require_file "${VLA_RUN_DIR}/checkpoints/${VLA_CKPT_NAME}"
    cd "${REPO}"
    print_context
    echo "[convert-hf] ckpt=${VLA_CKPT_NAME}"
    echo "[convert-hf] out=${HF_VLA_DIR}"
    rm -rf "${HF_VLA_TMP}"
    mkdir -p "${HF_VLA_TMP}"
    "${PYTHON}" vla-scripts/extern/convert_univla_weights_to_hf.py \
      --openvla_model_path_or_id "${VLA_RUN_DIR}" \
      --ckpt_name "${VLA_CKPT_NAME}" \
      --output_hf_model_local_path "${HF_VLA_TMP}" \
      --tokenizer_path "${TOKENIZER_PATH}" \
      --codebook_size "${LAM_CODEBOOK_SIZE}" \
      --latent_action_token_len "${LAM_LATENT_ACTION_TOKEN_LEN}" \
      "$@" \
      2>&1 | tee "${LOG_ROOT}/convert_hf.log"
    rm -rf "${HF_VLA_DIR}"
    mv "${HF_VLA_TMP}" "${HF_VLA_DIR}"
    echo "[convert-hf] done ${HF_VLA_DIR}"
    ;;

  finetune-simpler)
    set_finetune_env
    require_file "${HF_VLA_DIR}/config.json"
    require_file "${LAM_CKPT}"
    require_file "${LAM_CONFIG_PATH}"
    require_file "${SIMPLER_DATA_ROOT}/carrot/1.0.0/dataset_info.json"
    require_file "${SIMPLER_DATA_ROOT}/eggplant/1.0.0/dataset_info.json"
    require_file "${SIMPLER_DATA_ROOT}/spoon/1.0.0/dataset_info.json"
    require_file "${SIMPLER_DATA_ROOT}/stack/1.0.0/dataset_info.json"
    mkdir -p "${FINETUNE_RUN_ROOT}" "${FINETUNE_ADAPTER_TMP_DIR}"
    cd "${REPO}"
    print_context
    echo "[finetune-simpler] vla=${HF_VLA_DIR}"
    echo "[finetune-simpler] data=${SIMPLER_DATA_ROOT}"
    "${PYTHON}" -m torch.distributed.run \
      --standalone \
      --nproc_per_node "${NPROC_PER_NODE}" \
      --master_port "${FINETUNE_MASTER_PORT}" \
      vla-scripts/finetune_bridge.py \
      --vla_path "${HF_VLA_DIR}" \
      --lam_path "${LAM_CKPT}" \
      --lam_kind "${LAM_KIND}" \
      --lam_config_path "${LAM_CONFIG_PATH}" \
      --lam_token_view "${LAM_TOKEN_VIEW}" \
      --codebook_size "${LAM_CODEBOOK_SIZE}" \
      --latent_action_token_len "${LAM_LATENT_ACTION_TOKEN_LEN}" \
      --data_root_dir "${SIMPLER_DATA_ROOT}" \
      --dataset_name simpler \
      --run_root_dir "${FINETUNE_RUN_ROOT}" \
      --adapter_tmp_dir "${FINETUNE_ADAPTER_TMP_DIR}" \
      --wandb_project "${FINETUNE_WANDB_PROJECT}" \
      --wandb_entity "${WANDB_ENTITY}" \
      --run_id_note "${FINETUNE_RUN_ID_NOTE}" \
      --batch_size "${FINETUNE_BATCH_SIZE}" \
      --grad_accumulation_steps "${FINETUNE_GRAD_ACCUMULATION_STEPS}" \
      --learning_rate "${FINETUNE_LEARNING_RATE}" \
      --use_scheduler "${FINETUNE_USE_SCHEDULER}" \
      --max_steps "${FINETUNE_MAX_STEPS}" \
      --save_steps "${FINETUNE_SAVE_STEPS}" \
      --shuffle_buffer_size "${FINETUNE_SHUFFLE_BUFFER_SIZE}" \
      --image_aug "${FINETUNE_IMAGE_AUG}" \
      --window_size "${FINETUNE_WINDOW_SIZE}" \
      "$@" \
      2>&1 | tee "${LOG_ROOT}/finetune_simpler.log"
    upload_finetune_export_to_hf
    ;;

  *)
    echo "[unknown stage] ${STAGE}" >&2
    usage >&2
    exit 2
    ;;
esac
