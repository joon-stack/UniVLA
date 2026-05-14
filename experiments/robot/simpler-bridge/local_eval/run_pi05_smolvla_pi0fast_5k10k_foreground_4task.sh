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
RUN_ID="${RUN_ID:-pi05_smolvla_pi0fast_5k10k_fg_$(date +%Y%m%d_%H%M%S)}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-${ROOT}/outputs}"
EVAL_ROOT="${EVAL_ROOT:-${ROOT}/eval_logs/simplerenv_lerobot_pi05_smolvla_pi0fast_5k10k_foreground}"
LOG_ROOT="${LOG_ROOT:-${EVAL_ROOT}/logs/${RUN_ID}}"

GPU_PI05_5K="${GPU_PI05_5K:-2}"
GPU_PI05_10K="${GPU_PI05_10K:-3}"
GPU_SMOLVLA_5K="${GPU_SMOLVLA_5K:-4}"
GPU_SMOLVLA_10K="${GPU_SMOLVLA_10K:-5}"
GPU_PI0_FAST_5K="${GPU_PI0_FAST_5K:-6}"
GPU_PI0_FAST_10K="${GPU_PI0_FAST_10K:-7}"

NUM_EPISODES="${NUM_EPISODES:-24}"
NUM_ENVS="${NUM_ENVS:-1}"
SEED="${SEED:-0}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
POLICY_DEBUG_ACTIONS="${POLICY_DEBUG_ACTIONS:-0}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
POLICY_ACTION_EXECUTION="${POLICY_ACTION_EXECUTION:-queue}"
POLICY_TEMPORAL_AGG_COEFF="${POLICY_TEMPORAL_AGG_COEFF:-0.01}"
SIM_BACKEND="${SIM_BACKEND:-physx_cpu}"
RENDER_BACKEND="${RENDER_BACKEND:-sapien_cpu}"

mkdir -p "${LOG_ROOT}"

pids=()
names=()

cleanup() {
  if ((${#pids[@]})); then
    echo
    echo "[interrupt] stopping child evals: ${pids[*]}" >&2
    kill "${pids[@]}" 2>/dev/null || true
  fi
}
trap cleanup INT TERM

launch_eval() {
  local policy_name="$1"
  local output_name="$2"
  local step="$3"
  local gpu="$4"
  local tag="${policy_name}_step${step}"
  local policy_path="${BASE_OUTPUT_DIR}/${output_name}/checkpoints/${step}/pretrained_model"
  local record_dir="${EVAL_ROOT}/${RUN_ID}/${tag}"
  local log_file="${LOG_ROOT}/${tag}_gpu${gpu}.log"

  if [[ ! -f "${policy_path}/config.json" ]]; then
    echo "[missing checkpoint] ${policy_path}" >&2
    return 1
  fi

  mkdir -p "${record_dir}"
  {
    echo "[start] $(date -Is)"
    echo "[tag] ${tag}"
    echo "[gpu] ${gpu}"
    echo "[checkpoint] ${policy_path}"
    echo "[record_dir] ${record_dir}"
    echo "[log_file] ${log_file}"
    echo "[num_episodes] ${NUM_EPISODES}"
    echo "[seed_base] ${SEED}"
    echo "[save_video] ${SAVE_VIDEO}"
    echo "[policy_action_execution] ${POLICY_ACTION_EXECUTION}"
    echo "[policy_temporal_agg_coeff] ${POLICY_TEMPORAL_AGG_COEFF}"
    echo
  } | tee "${log_file}"

  (
    set -o pipefail
    env \
      ROOT="${ROOT}" \
      REPO="${REPO}" \
      GPU_ID="${gpu}" \
      POLICY_NAME="${policy_name}" \
      POLICY_PATH="${policy_path}" \
      POLICY_DEVICE="${POLICY_DEVICE}" \
      POLICY_DEBUG_ACTIONS="${POLICY_DEBUG_ACTIONS}" \
      POLICY_ACTION_EXECUTION="${POLICY_ACTION_EXECUTION}" \
      POLICY_TEMPORAL_AGG_COEFF="${POLICY_TEMPORAL_AGG_COEFF}" \
      NUM_EPISODES="${NUM_EPISODES}" \
      NUM_ENVS="${NUM_ENVS}" \
      SEED="${SEED}" \
      SAVE_VIDEO="${SAVE_VIDEO}" \
      SIM_BACKEND="${SIM_BACKEND}" \
      RENDER_BACKEND="${RENDER_BACKEND}" \
      RECORD_DIR="${record_dir}" \
      TASK_LIST="${TASK_LIST:-}" \
      PYTHONUNBUFFERED=1 \
      "${RUNNER}" 2>&1 \
      | sed -u "s/^/[${tag}:gpu${gpu}] /" \
      | tee -a "${log_file}"
  ) &

  pids+=("$!")
  names+=("${tag}:gpu${gpu}")
  echo "[launched] ${tag} gpu=${gpu} pid=${pids[-1]}"
}

echo "[run_id] ${RUN_ID}"
echo "[eval_root] ${EVAL_ROOT}/${RUN_ID}"
echo "[log_root] ${LOG_ROOT}"
echo "[runner] ${RUNNER}"
echo

launch_eval pi05 lerobot_pi05_simpler_success50_nobase_gbs128_4gpu 005000 "${GPU_PI05_5K}"
launch_eval pi05 lerobot_pi05_simpler_success50_nobase_gbs128_4gpu 010000 "${GPU_PI05_10K}"
launch_eval smolvla lerobot_smolvla_simpler_success50_nobase_gbs128_4gpu 005000 "${GPU_SMOLVLA_5K}"
launch_eval smolvla lerobot_smolvla_simpler_success50_nobase_gbs128_4gpu 010000 "${GPU_SMOLVLA_10K}"
launch_eval pi0_fast lerobot_pi0_fast_simpler_success50_nobase_gbs128_4gpu 005000 "${GPU_PI0_FAST_5K}"
launch_eval pi0_fast lerobot_pi0_fast_simpler_success50_nobase_gbs128_4gpu 010000 "${GPU_PI0_FAST_10K}"

echo
echo "[monitor] output is live above; logs are also written to:"
for i in "${!names[@]}"; do
  echo "  ${names[$i]} -> ${LOG_ROOT}/${names[$i]//:/_}.log"
done
echo

status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[done] ${names[$i]}"
  else
    rc=$?
    echo "[failed] ${names[$i]} exit=${rc}" >&2
    status=1
  fi
done

echo "[complete] RUN_ID=${RUN_ID}"
echo "Videos/metrics:"
echo "  ${EVAL_ROOT}/${RUN_ID}/pi05_step005000/real2sim_eval/"
echo "  ${EVAL_ROOT}/${RUN_ID}/pi05_step010000/real2sim_eval/"
echo "  ${EVAL_ROOT}/${RUN_ID}/smolvla_step005000/real2sim_eval/"
echo "  ${EVAL_ROOT}/${RUN_ID}/smolvla_step010000/real2sim_eval/"
echo "  ${EVAL_ROOT}/${RUN_ID}/pi0_fast_step005000/real2sim_eval/"
echo "  ${EVAL_ROOT}/${RUN_ID}/pi0_fast_step010000/real2sim_eval/"

exit "${status}"
