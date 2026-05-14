#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/run_lerobot_bridge_4task.sh" ]]; then
  DEFAULT_REPO="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
  DEFAULT_ROOT="$(cd -- "${DEFAULT_REPO}/.." && pwd)"
else
  DEFAULT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
fi

ROOT="${ROOT:-${DEFAULT_ROOT}}"
export ROOT
export POLICY_ACTION_EXECUTION="${POLICY_ACTION_EXECUTION:-temporal_agg}"
export POLICY_TEMPORAL_AGG_COEFF="${POLICY_TEMPORAL_AGG_COEFF:--0.1}"
export RUN_ID="${RUN_ID:-pi05_smolvla_pi0fast_5k10k_tagg_$(date +%Y%m%d_%H%M%S)}"
export EVAL_ROOT="${EVAL_ROOT:-${ROOT}/eval_logs/simplerenv_lerobot_pi05_smolvla_pi0fast_5k10k_temporal_agg}"

exec "${SCRIPT_DIR}/run_pi05_smolvla_pi0fast_5k10k_foreground_4task.sh" "$@"
