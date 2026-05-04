#!/usr/bin/env bash
set -uo pipefail

WAIT_SESSION="${WAIT_SESSION:-ft_lr1e4_vlamult01_4run_queue}"
LOG_DIR="${LOG_DIR:-/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong/outputs/lam_bridge/logs/hyperlam_queue_20260504}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${LOG_DIR}"

log() {
  echo "[$(date)] $*" | tee -a "${LOG_DIR}/queue.log"
}

watch_wandb() {
  local name="$1"
  local train_log="$2"
  local marker="${LOG_DIR}/${name}.wandb.ok"
  rm -f "${marker}"

  for _ in $(seq 1 60); do
    if grep -E "wandb: (Tracking run|View run|Syncing run)" "${train_log}" >> "${LOG_DIR}/wandb_check.log" 2>/dev/null; then
      touch "${marker}"
      log "${name} wandb lines detected; see ${LOG_DIR}/wandb_check.log"
      return 0
    fi
    sleep 60
  done

  log "WARNING: ${name} wandb lines not detected within 60 minutes"
  return 0
}

run_training() {
  local name="$1"
  local script="$2"
  local train_log="$3"

  log "starting ${name}"
  : > "${train_log}"
  watch_wandb "${name}" "${train_log}" &
  local watcher_pid=$!

  set +e
  bash -o pipefail -c "\"${script}\" 2>&1 | tee -a \"${train_log}\""
  local status=$?
  set -e

  wait "${watcher_pid}" || true
  log "${name} finished with status ${status}"
  return "${status}"
}

log "waiting for ${WAIT_SESSION} to finish"
while true; do
  if ! tmux has-session -t "${WAIT_SESSION}" 2>/dev/null; then
    break
  fi
  sleep 300
  log "still waiting for ${WAIT_SESSION}"
done

cd "${SCRIPT_DIR}"

run_training \
  "hyperbolic_radprog" \
  "${SCRIPT_DIR}/train_lam_bridge_visual_vq.sh" \
  "${LOG_DIR}/hyperbolic_radprog_train.log"
hyperbolic_status=$?
if [[ "${hyperbolic_status}" -ne 0 ]]; then
  log "stopping queue because hyperbolic_radprog failed"
  exit "${hyperbolic_status}"
fi

run_training \
  "euclidean_control" \
  "${SCRIPT_DIR}/train_lam_bridge_visual_vq_euclidean_control.sh" \
  "${LOG_DIR}/euclidean_control_train.log"
control_status=$?
exit "${control_status}"
