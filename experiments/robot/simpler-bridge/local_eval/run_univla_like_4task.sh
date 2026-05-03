#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong}"
REPO="${REPO:-${ROOT}/UniVLA}"
SIMPLERENV_DIR="${SIMPLERENV_DIR:-${ROOT}/third_party/SimplerEnv-maniskill3}"
PYTHON="${PYTHON:-python}"
GPU_ID="${GPU_ID:-0}"
NUM_EPISODES="${NUM_EPISODES:-24}"
NUM_ENVS="${NUM_ENVS:-1}"
SEED="${SEED:-0}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
PRED_ACTION_HORIZON="${PRED_ACTION_HORIZON:-10}"
SIM_BACKEND="${SIM_BACKEND:-physx_cpu}"
RENDER_BACKEND="${RENDER_BACKEND:-sapien_cpu}"
RECORD_DIR="${RECORD_DIR:-${ROOT}/eval_logs/simplerenv_univla_like}"

if [[ -z "${CKPT_PATH:-}" || -z "${ACTION_DECODER_PATH:-}" ]]; then
  echo "Set CKPT_PATH and ACTION_DECODER_PATH first." >&2
  echo "Example:" >&2
  echo "  source ${REPO}/experiments/robot/simpler-bridge/local_eval/resolve_univla_finetune_artifacts.sh" >&2
  exit 1
fi

if [[ ! -f "${SIMPLERENV_DIR}/real2sim_eval_maniskill3.py" ]]; then
  echo "[missing] ${SIMPLERENV_DIR}/real2sim_eval_maniskill3.py" >&2
  echo "Run prepare_simplerenv_workspace.sh first." >&2
  exit 1
fi

export PYTHONPATH="${REPO}:${REPO}/latent_action_model:${SIMPLERENV_DIR}:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TOKENIZERS_PARALLELISM=false
export UNIVLA_EVAL_ATTN_IMPLEMENTATION="${UNIVLA_EVAL_ATTN_IMPLEMENTATION:-sdpa}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/lvp_icd.json}"
export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/dev/null}"
export DISPLAY="${DISPLAY:-}"

TASKS=(
  "PutSpoonOnTableClothInScene-v1"
  "PutCarrotOnPlateInScene-v1"
  "StackGreenCubeOnYellowCubeBakedTexInScene-v1"
  "PutEggplantInBasketScene-v1"
)

SAVE_VIDEO_ARG="--save-video"
if [[ "${SAVE_VIDEO}" == "0" || "${SAVE_VIDEO}" == "false" || "${SAVE_VIDEO}" == "False" ]]; then
  SAVE_VIDEO_ARG="--no-save-video"
fi

cd "${SIMPLERENV_DIR}"
for task in "${TASKS[@]}"; do
  echo "[eval] ${task}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" real2sim_eval_maniskill3.py \
    --model="univla" \
    -e "${task}" \
    -s "${SEED}" \
    --num-episodes "${NUM_EPISODES}" \
    --num-envs "${NUM_ENVS}" \
    --sim-backend "${SIM_BACKEND}" \
    --render-backend "${RENDER_BACKEND}" \
    --record-dir "${RECORD_DIR}" \
    "${SAVE_VIDEO_ARG}" \
    --pred-action-horizon "${PRED_ACTION_HORIZON}" \
    --action_decoder_path "${ACTION_DECODER_PATH}" \
    --ckpt_path "${CKPT_PATH}"
done
