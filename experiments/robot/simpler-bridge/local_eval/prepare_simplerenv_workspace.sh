#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong}"
REPO="${REPO:-${ROOT}/UniVLA}"
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
require_file "${ADAPTER_SRC}/real2sim_eval_maniskill3.py"
require_file "${ADAPTER_SRC}/eval_simpler_bridge_4task.sh"

mkdir -p "${SIMPLERENV_DIR}/simpler_env/policies"
rm -rf "${SIMPLERENV_DIR}/simpler_env/policies/univla"
cp -a "${ADAPTER_SRC}/policies/univla" "${SIMPLERENV_DIR}/simpler_env/policies/univla"
cp "${ADAPTER_SRC}/real2sim_eval_maniskill3.py" "${SIMPLERENV_DIR}/real2sim_eval_maniskill3.py"
cp "${ADAPTER_SRC}/eval_simpler_bridge_4task.sh" "${SIMPLERENV_DIR}/eval_univla_4task_ref.sh"

cat > "${SIMPLERENV_DIR}/SIMPLERENV_UNIVLA_READY.txt" <<EOF
Prepared on: $(date -Is)
Adapter source: ${ADAPTER_SRC}
Policy target: ${SIMPLERENV_DIR}/simpler_env/policies/univla
Eval script: ${SIMPLERENV_DIR}/real2sim_eval_maniskill3.py
EOF

echo "[ready] SimpleREnv UniVLA adapter installed at ${SIMPLERENV_DIR}"
echo "[note] If this checkout is ManiSkill2-only, use a fresh SimplerEnv maniskill3 checkout."
