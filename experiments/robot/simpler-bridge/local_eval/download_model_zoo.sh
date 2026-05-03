#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong}"
HF_HOME="${HF_HOME:-${ROOT}/data/hf_cache}"
MODEL_ZOO="${MODEL_ZOO:-${ROOT}/data/model_zoo}"
PYTHON="${PYTHON:-${ROOT}/.venvs/simplerenv-eval-py310/bin/python}"

mkdir -p "${MODEL_ZOO}" "${HF_HOME}" "${ROOT}/LAPA/lapa_checkpoints"
export HF_HOME

download_hf() {
  local repo_id="$1"
  local out_dir="$2"
  shift 2
  "${PYTHON}" - "${repo_id}" "${out_dir}" "$@" <<'PY'
from pathlib import Path
import sys

from huggingface_hub import hf_hub_download, snapshot_download

repo_id, out_dir, *files = sys.argv[1:]
Path(out_dir).mkdir(parents=True, exist_ok=True)
if files:
    for filename in files:
        hf_hub_download(
            repo_id,
            filename,
            local_dir=out_dir,
            local_dir_use_symlinks=False,
        )
else:
    snapshot_download(repo_id, local_dir=out_dir, local_dir_use_symlinks=False)
PY
}

echo "[model-zoo] downloading LAPA OpenX checkpoint files"
download_hf latent-action-pretraining/LAPA-7B-openx "${ROOT}/LAPA/lapa_checkpoints" \
  tokenizer.model vqgan params

if [[ "${DOWNLOAD_MVP_SIMPLER:-0}" == "1" ]]; then
  echo "[model-zoo] downloading MVP-LAM SimpleREnv finetuned checkpoint"
  download_hf JM-Lee/mvp-lam-7b-224-simpler "${MODEL_ZOO}/mvp-lam-7b-224-simpler"
else
  echo "[skip] Set DOWNLOAD_MVP_SIMPLER=1 to download JM-Lee/mvp-lam-7b-224-simpler"
fi

if [[ "${DOWNLOAD_VILLAX:-0}" == "1" ]]; then
  echo "[model-zoo] downloading villa-X model repo"
  download_hf microsoft/villa-x "${MODEL_ZOO}/villa-x"
else
  echo "[skip] Set DOWNLOAD_VILLAX=1 to download microsoft/villa-x"
fi

cat <<EOF
[done] model zoo root: ${MODEL_ZOO}
LAPA checkpoint dir: ${ROOT}/LAPA/lapa_checkpoints
EOF
