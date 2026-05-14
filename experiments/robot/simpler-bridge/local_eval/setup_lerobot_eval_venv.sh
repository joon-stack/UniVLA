#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
ROOT="${ROOT:-$(cd -- "${DEFAULT_REPO}/.." && pwd)}"
REPO="${REPO:-${DEFAULT_REPO}}"
SIMPLERENV_DIR="${SIMPLERENV_DIR:-${ROOT}/third_party/SimplerEnv-maniskill3}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
VENV="${VENV:-${ROOT}/.venvs/simplerenv-lerobot-eval-py312}"

if [[ ! -d "${SIMPLERENV_DIR}" ]]; then
  echo "[missing] ${SIMPLERENV_DIR}" >&2
  echo "Set SIMPLERENV_DIR to a SimplerEnv ManiSkill3 checkout." >&2
  exit 1
fi

if [[ ! -d "${VENV}" ]]; then
  uv venv --python "${PYTHON_VERSION}" "${VENV}"
fi

uv pip install --python "${VENV}/bin/python" --upgrade pip setuptools wheel
uv pip install --python "${VENV}/bin/python" -e "${SIMPLERENV_DIR}"
uv pip install --python "${VENV}/bin/python" --upgrade "git+https://github.com/haosulab/ManiSkill.git"
uv pip install --python "${VENV}/bin/python" \
  tyro gymnasium sapien transforms3d opencv-python matplotlib pillow tree sentencepiece tensorflow
uv pip install --python "${VENV}/bin/python" \
  "lerobot[dataset,pi,smolvla,diffusion] @ git+https://github.com/huggingface/lerobot.git"
uv pip install --python "${VENV}/bin/python" tensorflow-datasets

cat <<EOF
[ready] LeRobot SimplerEnv eval venv: ${VENV}
Use:
  VENV="${VENV}" ${REPO}/experiments/robot/simpler-bridge/local_eval/run_lerobot_bridge_4task.sh
EOF
