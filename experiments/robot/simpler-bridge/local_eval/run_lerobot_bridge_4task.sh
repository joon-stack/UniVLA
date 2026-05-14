#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "${SCRIPT_DIR}/simpler_env" && -f "${SCRIPT_DIR}/real2sim_eval_maniskill3.py" ]]; then
  DEFAULT_SIMPLERENV_DIR="${SCRIPT_DIR}"
  DEFAULT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
  if [[ -d "${DEFAULT_ROOT}/UniVLA-hyper" ]]; then
    DEFAULT_REPO="${DEFAULT_ROOT}/UniVLA-hyper"
  else
    DEFAULT_REPO="${DEFAULT_ROOT}/UniVLA"
  fi
else
  DEFAULT_REPO="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
  DEFAULT_ROOT="$(cd -- "${DEFAULT_REPO}/.." && pwd)"
  DEFAULT_SIMPLERENV_DIR="${DEFAULT_ROOT}/third_party/SimplerEnv-maniskill3"
fi
ROOT="${ROOT:-${DEFAULT_ROOT}}"
REPO="${REPO:-${DEFAULT_REPO}}"
SIMPLERENV_DIR="${SIMPLERENV_DIR:-${DEFAULT_SIMPLERENV_DIR}}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${REPO}/experiments/robot/simpler-bridge/real2sim_eval_maniskill3.py}"
if [[ ! -f "${EVAL_SCRIPT}" ]]; then
  EVAL_SCRIPT="${SIMPLERENV_DIR}/real2sim_eval_maniskill3.py"
fi
VENV="${VENV:-${ROOT}/.venvs/simplerenv-lerobot-eval-py312}"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "${VENV}/bin/python" ]]; then
    PYTHON="${VENV}/bin/python"
  else
    PYTHON="python"
  fi
fi

GPU_ID="${GPU_ID:-0}"
POLICY_NAME="${POLICY_NAME:-pi05}"
DEFAULT_POLICY_PATH="${ROOT}/outputs/lerobot_pi05_simpler_success50_nobase_gbs128_4gpu/checkpoints/010000/pretrained_model"
POLICY_PATH="${POLICY_PATH:-${DEFAULT_POLICY_PATH}}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
POLICY_DEBUG_ACTIONS="${POLICY_DEBUG_ACTIONS:-3}"
POLICY_ACTION_EXECUTION="${POLICY_ACTION_EXECUTION:-queue}"
POLICY_TEMPORAL_AGG_COEFF="${POLICY_TEMPORAL_AGG_COEFF:--0.1}"
NUM_EPISODES="${NUM_EPISODES:-24}"
NUM_ENVS="${NUM_ENVS:-1}"
SEED="${SEED:-0}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
SIM_BACKEND="${SIM_BACKEND:-physx_cpu}"
RENDER_BACKEND="${RENDER_BACKEND:-sapien_cpu}"
RECORD_DIR="${RECORD_DIR:-${ROOT}/eval_logs/simplerenv_lerobot_${POLICY_NAME}}"

if [[ ! -f "${EVAL_SCRIPT}" ]]; then
  echo "[missing] eval script: ${EVAL_SCRIPT}" >&2
  echo "Run ${REPO}/experiments/robot/simpler-bridge/local_eval/prepare_simplerenv_workspace.sh first." >&2
  exit 1
fi

if [[ -d "${POLICY_PATH}" && ! -f "${POLICY_PATH}/config.json" ]]; then
  echo "[bad policy path] missing config.json under ${POLICY_PATH}" >&2
  exit 1
fi

if ! "${PYTHON}" - <<'PY' >/dev/null 2>&1
import lerobot
PY
then
  echo "[missing] lerobot is not installed in ${PYTHON}" >&2
  echo "Run ${REPO}/experiments/robot/simpler-bridge/local_eval/setup_lerobot_eval_venv.sh, or install it into the SimplerEnv eval venv:" >&2
  echo "  uv pip install --python \"${PYTHON}\" 'lerobot[pi]' tensorflow-datasets" >&2
  exit 1
fi

export PYTHONPATH="${REPO}:${REPO}/latent_action_model:${SIMPLERENV_DIR}:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TOKENIZERS_PARALLELISM=false
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/lvp_icd.json}"
export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/dev/null}"
export DISPLAY="${DISPLAY:-}"

if [[ -n "${TASK_LIST:-}" ]]; then
  read -r -a TASKS <<< "${TASK_LIST}"
else
  TASKS=(
    "PutSpoonOnTableClothInScene-v1"
    "PutCarrotOnPlateInScene-v1"
    "StackGreenCubeOnYellowCubeBakedTexInScene-v1"
    "PutEggplantInBasketScene-v1"
  )
fi

SAVE_VIDEO_ARG="--save-video"
if [[ "${SAVE_VIDEO}" == "0" || "${SAVE_VIDEO}" == "false" || "${SAVE_VIDEO}" == "False" ]]; then
  SAVE_VIDEO_ARG="--no-save-video"
fi

cd "${SIMPLERENV_DIR}"
for task in "${TASKS[@]}"; do
  echo "[eval] model=${POLICY_NAME} task=${task} policy=${POLICY_PATH}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" "${EVAL_SCRIPT}" \
    --model="${POLICY_NAME}" \
    --policy-path "${POLICY_PATH}" \
    --policy-device "${POLICY_DEVICE}" \
    --policy-debug-actions "${POLICY_DEBUG_ACTIONS}" \
    --policy-action-execution "${POLICY_ACTION_EXECUTION}" \
    --policy-temporal-agg-coeff "${POLICY_TEMPORAL_AGG_COEFF}" \
    -e "${task}" \
    -s "${SEED}" \
    --num-episodes "${NUM_EPISODES}" \
    --num-envs "${NUM_ENVS}" \
    --sim-backend "${SIM_BACKEND}" \
    --render-backend "${RENDER_BACKEND}" \
    --record-dir "${RECORD_DIR}" \
    "${SAVE_VIDEO_ARG}"
done
