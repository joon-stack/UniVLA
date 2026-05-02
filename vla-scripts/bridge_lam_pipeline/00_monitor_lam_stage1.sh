#!/usr/bin/env bash
set -euo pipefail

ROOT="/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong"
LOG="${ROOT}/outputs/lam_bridge/run_logs/lam_stage1_bridge_8xb200.log"
CKPT_DIR="${ROOT}/outputs/lam_bridge/logs/task_centric_lam_stage1_bridge"

echo "[stage1] tmux sessions"
tmux ls 2>/dev/null || true

echo
echo "[stage1] recent log"
tail -80 "${LOG}"

echo
echo "[stage1] checkpoints"
find "${CKPT_DIR}" -maxdepth 1 -type f -name "*.ckpt" -printf "%f %s bytes\n" | sort || true

echo
echo "[gpu]"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits

