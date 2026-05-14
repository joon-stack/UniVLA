#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/run_lerobot_bridge_4task.sh" ]]; then
  RUNNER="${SCRIPT_DIR}/run_lerobot_bridge_4task.sh"
  DEFAULT_REPO="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
  DEFAULT_ROOT="$(cd -- "${DEFAULT_REPO}/.." && pwd)"
else
  RUNNER="${SCRIPT_DIR}/eval_lerobot_bridge_4task.sh"
  DEFAULT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
  if [[ -d "${DEFAULT_ROOT}/UniVLA-hyper" ]]; then
    DEFAULT_REPO="${DEFAULT_ROOT}/UniVLA-hyper"
  else
    DEFAULT_REPO="${DEFAULT_ROOT}/UniVLA"
  fi
fi

ROOT="${ROOT:-${DEFAULT_ROOT}}"
REPO="${REPO:-${DEFAULT_REPO}}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-${ROOT}/outputs/lerobot_pi05_simpler_success50_nobase_gbs128_4gpu}"
EVAL_ROOT="${EVAL_ROOT:-${ROOT}/eval_logs/simplerenv_lerobot_pi05_5k_10k}"
LOG_ROOT="${LOG_ROOT:-${EVAL_ROOT}/logs/${RUN_ID}}"

GPU_ID="${GPU_ID:-0}"
NUM_EPISODES="${NUM_EPISODES:-24}"
NUM_ENVS="${NUM_ENVS:-1}"
SEED="${SEED:-0}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
POLICY_DEBUG_ACTIONS="${POLICY_DEBUG_ACTIONS:-0}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
SIM_BACKEND="${SIM_BACKEND:-physx_cpu}"
RENDER_BACKEND="${RENDER_BACKEND:-sapien_cpu}"

mkdir -p "${LOG_ROOT}"

run_step() {
  local step="$1"
  local tag="$2"
  local policy_path="${BASE_OUTPUT_DIR}/checkpoints/${step}/pretrained_model"
  local record_dir="${EVAL_ROOT}/${RUN_ID}/${tag}"
  local log_file="${LOG_ROOT}/${tag}.log"

  if [[ ! -f "${policy_path}/config.json" ]]; then
    echo "[missing checkpoint] ${policy_path}" >&2
    return 1
  fi

  mkdir -p "${record_dir}"
  {
    echo "[start] $(date -Is)"
    echo "[tag] ${tag}"
    echo "[checkpoint] ${policy_path}"
    echo "[record_dir] ${record_dir}"
    echo "[log_file] ${log_file}"
    echo "[num_episodes] ${NUM_EPISODES}"
    echo "[save_video] ${SAVE_VIDEO}"
    echo
  } | tee "${log_file}"

  env \
    ROOT="${ROOT}" \
    REPO="${REPO}" \
    GPU_ID="${GPU_ID}" \
    POLICY_NAME="pi05" \
    POLICY_PATH="${policy_path}" \
    POLICY_DEVICE="${POLICY_DEVICE}" \
    POLICY_DEBUG_ACTIONS="${POLICY_DEBUG_ACTIONS}" \
    NUM_EPISODES="${NUM_EPISODES}" \
    NUM_ENVS="${NUM_ENVS}" \
    SEED="${SEED}" \
    SAVE_VIDEO="${SAVE_VIDEO}" \
    SIM_BACKEND="${SIM_BACKEND}" \
    RENDER_BACKEND="${RENDER_BACKEND}" \
    RECORD_DIR="${record_dir}" \
    TASK_LIST="${TASK_LIST:-}" \
    "${RUNNER}" 2>&1 | tee -a "${log_file}"

  echo "[done] $(date -Is) ${tag}" | tee -a "${log_file}"
}

echo "[run_id] ${RUN_ID}"
echo "[eval_root] ${EVAL_ROOT}/${RUN_ID}"
echo "[log_root] ${LOG_ROOT}"

run_step "005000" "step005000"
run_step "010000" "step010000"

cat <<EOF
[complete] RUN_ID=${RUN_ID}
Logs:
  ${LOG_ROOT}/step005000.log
  ${LOG_ROOT}/step010000.log
Videos/metrics:
  ${EVAL_ROOT}/${RUN_ID}/step005000/real2sim_eval/
  ${EVAL_ROOT}/${RUN_ID}/step010000/real2sim_eval/
EOF

