#!/usr/bin/env bash
set -euo pipefail

ROOT="/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong"
BASE_VLM="${ROOT}/data/univla_checkpoints/prismatic-vlms/prism-dinosiglip-224px+7b"

test -f "${BASE_VLM}/config.json"
test -f "${BASE_VLM}/config.yaml"
test -f "${BASE_VLM}/checkpoints/latest-checkpoint.pt"

echo "Base VLM is ready:"
echo "${BASE_VLM}"
ls -lh "${BASE_VLM}/checkpoints/latest-checkpoint.pt"

