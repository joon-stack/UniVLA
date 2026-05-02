#!/usr/bin/env bash
set -euo pipefail

ROOT="/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong"
REPO="${ROOT}/UniVLA"
PIPELINE_DIR="${REPO}/vla-scripts/bridge_lam_pipeline"
LOG_DIR="${ROOT}/outputs/univla_bridge_lam_local/run_logs"
SUMMARY_LOG="${LOG_DIR}/overnight_stage2_cache_vla.log"

STAGE2_CKPT="${ROOT}/outputs/lam_bridge/logs/task_centric_lam_stage2_bridge/last.ckpt"
CACHE_DB="${CACHE_DB:-${ROOT}/outputs/univla_bridge_lam_local/cache/bridge_latent_actions_stage2.sqlite}"

mkdir -p "${LOG_DIR}"

{
  echo "[overnight] start $(date)"
  echo "[overnight] repo=${REPO}"
  echo "[overnight] stage2_ckpt=${STAGE2_CKPT}"
  echo "[overnight] cache_db=${CACHE_DB}"

  if [[ -f "${STAGE2_CKPT}" && "${FORCE_STAGE2:-0}" != "1" ]]; then
    echo "[overnight] skip stage2: checkpoint exists"
  else
    echo "[overnight] run stage2"
    "${PIPELINE_DIR}/01_train_lam_stage2_bridge.sh"
  fi
  test -f "${STAGE2_CKPT}"

  if [[ -f "${CACHE_DB}" && "${FORCE_CACHE:-0}" != "1" ]]; then
    echo "[overnight] skip cache: cache db exists"
  else
    echo "[overnight] run cache"
    CACHE_DB="${CACHE_DB}" "${PIPELINE_DIR}/02_cache_bridge_latent_actions.sh"
  fi
  test -f "${CACHE_DB}"

  echo "[overnight] run cached VLA"
  CACHE_DB="${CACHE_DB}" "${PIPELINE_DIR}/03_train_vla_bridge_cached.sh"

  echo "[overnight] done $(date)"
} 2>&1 | tee "${SUMMARY_LOG}"
