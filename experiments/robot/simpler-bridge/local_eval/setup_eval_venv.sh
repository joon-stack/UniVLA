#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong}"
REPO="${REPO:-${ROOT}/UniVLA}"
SIMPLERENV_DIR="${SIMPLERENV_DIR:-${ROOT}/third_party/SimplerEnv-maniskill3}"
VENV="${VENV:-${ROOT}/.venvs/simplerenv-eval-py310}"

if [[ ! -d "${VENV}" ]]; then
  uv venv --python 3.10 "${VENV}"
fi

# Local package first; this is cheap and does not require network.
uv pip install --python "${VENV}/bin/python" --upgrade pip setuptools wheel
uv pip install --python "${VENV}/bin/python" -e "${SIMPLERENV_DIR}"

# ManiSkill3 is required by real2sim_eval_maniskill3.py. This usually needs
# network access unless the package is already cached.
uv pip install --python "${VENV}/bin/python" --upgrade "git+https://github.com/haosulab/ManiSkill.git"

# Policy-side dependencies. Torch/TF/Transformers versions may need to be pinned
# to match the model being evaluated; keep this script minimal and explicit.
uv pip install --python "${VENV}/bin/python" \
  tyro gymnasium sapien transforms3d opencv-python matplotlib pillow tree sentencepiece \
  tensorflow

# Install UniVLA for the prismatic package and local policy dependencies.
uv pip install --python "${VENV}/bin/python" -e "${REPO}"

# Match the UniVLA training stack for HF/OpenVLA compatibility.
uv pip install --python "${VENV}/bin/python" \
  "transformers==4.40.1" \
  "tokenizers==0.19.1" \
  "huggingface-hub==0.26.1" \
  "accelerate==0.32.1" \
  "peft==0.11.1"

cat <<EOF
[ready] eval venv: ${VENV}
Use:
  source ${VENV}/bin/activate
EOF
