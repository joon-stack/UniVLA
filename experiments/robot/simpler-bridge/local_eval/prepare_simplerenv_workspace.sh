#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
ROOT="${ROOT:-$(cd -- "${DEFAULT_REPO}/.." && pwd)}"
REPO="${REPO:-${DEFAULT_REPO}}"
SIMPLERENV_DIR="${SIMPLERENV_DIR:-${ROOT}/third_party/SimplerEnv-maniskill3}"
ADAPTER_SRC="${ADAPTER_SRC:-${REPO}/experiments/robot/simpler-bridge}"

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

require_dir "${SIMPLERENV_DIR}"
require_dir "${SIMPLERENV_DIR}/simpler_env"
require_dir "${ADAPTER_SRC}/policies/univla"
require_dir "${ADAPTER_SRC}/policies/lerobot_bridge"
require_file "${ADAPTER_SRC}/real2sim_eval_maniskill3.py"
require_file "${ADAPTER_SRC}/eval_simpler_bridge_4task.sh"
require_file "${ADAPTER_SRC}/local_eval/run_lerobot_bridge_4task.sh"
require_file "${ADAPTER_SRC}/local_eval/run_pi05_5k_10k_4task.sh"
require_file "${ADAPTER_SRC}/local_eval/run_pi05_5k_10k_parallel_4task.sh"
require_file "${ADAPTER_SRC}/local_eval/run_smolvla_pi0fast_5k10k_foreground_4task.sh"
require_file "${ADAPTER_SRC}/local_eval/run_pi05_smolvla_pi0fast_5k10k_foreground_4task.sh"
require_file "${ADAPTER_SRC}/local_eval/run_pi05_smolvla_pi0fast_5k10k_temporal_agg_4task.sh"
require_file "${ADAPTER_SRC}/local_eval/run_pi05_smolvla_diffusion_5k10k_temporal_agg_4task.sh"

mkdir -p "${SIMPLERENV_DIR}/simpler_env/policies"
rm -rf "${SIMPLERENV_DIR}/simpler_env/policies/univla"
rm -rf "${SIMPLERENV_DIR}/simpler_env/policies/lerobot_bridge"
cp -a "${ADAPTER_SRC}/policies/univla" "${SIMPLERENV_DIR}/simpler_env/policies/univla"
cp -a "${ADAPTER_SRC}/policies/lerobot_bridge" "${SIMPLERENV_DIR}/simpler_env/policies/lerobot_bridge"
cp "${ADAPTER_SRC}/real2sim_eval_maniskill3.py" "${SIMPLERENV_DIR}/real2sim_eval_maniskill3.py"
cp "${ADAPTER_SRC}/eval_simpler_bridge_4task.sh" "${SIMPLERENV_DIR}/eval_univla_4task_ref.sh"
cp "${ADAPTER_SRC}/local_eval/run_lerobot_bridge_4task.sh" "${SIMPLERENV_DIR}/eval_lerobot_bridge_4task.sh"
cp "${ADAPTER_SRC}/local_eval/run_pi05_5k_10k_4task.sh" "${SIMPLERENV_DIR}/eval_pi05_5k_10k_4task.sh"
cp "${ADAPTER_SRC}/local_eval/run_pi05_5k_10k_parallel_4task.sh" "${SIMPLERENV_DIR}/eval_pi05_5k_10k_parallel_4task.sh"
cp "${ADAPTER_SRC}/local_eval/run_smolvla_pi0fast_5k10k_foreground_4task.sh" "${SIMPLERENV_DIR}/eval_smolvla_pi0fast_5k10k_foreground_4task.sh"
cp "${ADAPTER_SRC}/local_eval/run_pi05_smolvla_pi0fast_5k10k_foreground_4task.sh" "${SIMPLERENV_DIR}/eval_pi05_smolvla_pi0fast_5k10k_foreground_4task.sh"
cp "${ADAPTER_SRC}/local_eval/run_pi05_smolvla_pi0fast_5k10k_temporal_agg_4task.sh" "${SIMPLERENV_DIR}/eval_pi05_smolvla_pi0fast_5k10k_temporal_agg_4task.sh"
cp "${ADAPTER_SRC}/local_eval/run_pi05_smolvla_diffusion_5k10k_temporal_agg_4task.sh" "${SIMPLERENV_DIR}/eval_pi05_smolvla_diffusion_5k10k_temporal_agg_4task.sh"

cat > "${SIMPLERENV_DIR}/SIMPLERENV_UNIVLA_READY.txt" <<EOF
Prepared on: $(date -Is)
Adapter source: ${ADAPTER_SRC}
UniVLA policy target: ${SIMPLERENV_DIR}/simpler_env/policies/univla
LeRobot policy target: ${SIMPLERENV_DIR}/simpler_env/policies/lerobot_bridge
Eval script: ${SIMPLERENV_DIR}/real2sim_eval_maniskill3.py
EOF

echo "[ready] SimpleREnv UniVLA/LeRobot adapters installed at ${SIMPLERENV_DIR}"
echo "[note] If this checkout is ManiSkill2-only, use a fresh SimplerEnv maniskill3 checkout."
