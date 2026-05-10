# BridgeV2 Fast Run Notes

This note records the local settings that made UniVLA BridgeV2 runs usable on the
B200 node. It is intentionally operational: use it before restarting LAM stage2,
VLA pretraining, or Bridge finetuning.

## Environment

- Repo: `/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong/UniVLA`
- Root: `/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong`
- BridgeV2 data: `${ROOT}/data/rlds_bridge_orig`
- LAM stage1 checkpoint: `${ROOT}/outputs/lam_bridge/logs/task_centric_lam_stage1_bridge/epoch=1-step=60000.ckpt`
- LAM stage2 checkpoint: `${ROOT}/outputs/lam_bridge/logs/task_centric_lam_stage2_bridge/last.ckpt`
- VLA venv: `${ROOT}/.venvs/univla-tfcheck-py310`
- LAM stage1/stage2 venv: `${REPO}/.venv`

The VLA/finetune venv needs its NVIDIA wheel library path on `LD_LIBRARY_PATH`:

```bash
NEW_VENV=/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong/.venvs/univla-tfcheck-py310
NVIDIA_SITE=$NEW_VENV/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$NVIDIA_SITE/cublas/lib:$NVIDIA_SITE/cuda_cupti/lib:$NVIDIA_SITE/cuda_nvrtc/lib:$NVIDIA_SITE/cuda_runtime/lib:$NVIDIA_SITE/cudnn/lib:$NVIDIA_SITE/cufft/lib:$NVIDIA_SITE/curand/lib:$NVIDIA_SITE/cusolver/lib:$NVIDIA_SITE/cusparse/lib:$NVIDIA_SITE/nccl/lib:$NVIDIA_SITE/nvjitlink/lib:${LD_LIBRARY_PATH:-}
```

## LAM Stage1

Stage1 was not the bottleneck in the local runs. Keep the official/local recipe
unless there is a new failure.

Known good output:

```text
outputs/lam_bridge/logs/task_centric_lam_stage1_bridge/epoch=1-step=60000.ckpt
```

## LAM Stage2

The original slow path was mostly model compute, not just dataloading. The local
fix was to use PyTorch SDPA/fused attention in the LAM attention block. After
that change, stage2 returned to roughly `~3 it/s` in the observed run.

Use the old repo venv:

```bash
cd /NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong/UniVLA
source .venv/bin/activate
```

Recommended stage2 env:

```bash
export UNIVLA_LAM_NUM_WORKERS=2
export UNIVLA_LAM_PREFETCH_FACTOR=2
export UNIVLA_LAM_DISABLE_USAGE_UPDATE=1
export UNIVLA_TRAJ_THREADS=4
export UNIVLA_TRAJ_READ_THREADS=4
export UNIVLA_FRAME_THREADS=8
export UNIVLA_TF_RAM_BUDGET_MB=512
```

Use `UNIVLA_RLDS_LEN_OVERRIDE` only for LAM stage2 when running a fixed
`max_steps`. This avoids hitting the RLDS epoch boundary, where the run was seen
to stall after codebook restart logs.

```bash
export LAM_STAGE2_MAX_STEPS=60000
export UNIVLA_RLDS_LEN_OVERRIDE=$(((LAM_STAGE2_MAX_STEPS + 1000) * 64))
```

Important: unset this before VLA.

```bash
unset UNIVLA_RLDS_LEN_OVERRIDE
```

Do not resume from an epoch-boundary `last*.ckpt` if the previous run stalled
right after `Restarting 16 codes`. Resume from a clean step checkpoint instead,
for example:

```bash
export LAM_STAGE2_RESUME_CKPT=/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong/outputs/lam_bridge/logs/task_centric_lam_stage2_bridge/epoch=0-step=30000.ckpt
```

## VLA Pretraining

Use the new TF-check venv:

```bash
ROOT=/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong
REPO=$ROOT/UniVLA
NEW_VENV=$ROOT/.venvs/univla-tfcheck-py310
cd $REPO
```

Known-good VLA env:

```bash
unset UNIVLA_RLDS_LEN_OVERRIDE
export PYTHONPATH=$REPO:$REPO/latent_action_model:${PYTHONPATH:-}
export HF_HOME=$ROOT/data/hf_cache
export WANDB_MODE=online
export UNIVLA_DUMMY_LATENT_ACTIONS=0
export UNIVLA_TRAJ_THREADS=4
export UNIVLA_TRAJ_READ_THREADS=4
export UNIVLA_FRAME_THREADS=8
export UNIVLA_TF_RAM_BUDGET_MB=512
export OMP_NUM_THREADS=8
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

Known-good VLA settings:

```text
--vla.type prism-dinosiglip-224px+mx-bridge
--vla.max_steps 50000
--vla.shuffle_buffer_size 20000
--image_aug true
--pretrain_vlm ${ROOT}/data/univla_checkpoints/prismatic-vlms/prism-dinosiglip-224px+7b
--lam_path ${ROOT}/outputs/lam_bridge/logs/task_centric_lam_stage2_bridge/last.ckpt
--data_root_dir ${ROOT}/data/rlds_bridge_orig
```

Observed healthy speed after warmup:

```text
~1.24-1.30 sec/step
GPU util ~96-100% on all 8 GPUs
global batch size 256
per-device batch size 32
```

The slow failed VLA run was around `~48 sec/step`. The concrete runtime
difference we found was that the pipeline shell had inherited the LAM-only
`UNIVLA_RLDS_LEN_OVERRIDE`. That variable must not leak into VLA. The current
known-good chain script explicitly unsets it and pins `OMP_NUM_THREADS=8`.

B200 TensorFlow warning:

```text
TensorFlow was not built with CUDA kernel binaries compatible with compute capability 10.0a.
```

This warning still appears with the TF 2.21 wheel because the wheel does not
ship native B200 `sm_100/sm_100a` cubins. In the observed good VLA run, it did
not prevent normal training speed after warmup. Without changing Docker/base
CUDA, treat this as a warning unless step time or GPU util is bad.

## Bridge Finetuning

The chain after VLA is:

```text
VLA checkpoint -> HF convert -> Bridge raw-action finetune
```

Current chain script:

```bash
./vla-scripts/bridge_lam_pipeline/run_vla50k_then_bridge_finetune.sh
```

It runs:

```text
VLA pretrain: 50000 steps
convert_univla_weights_to_hf.py
Bridge finetune: 30000 steps
finetune batch size: 8
finetune save steps: 30000
```

Finetune uses BridgeV2 data, not LIBERO:

```text
--dataset_name bridge
--data_root_dir ${ROOT}/data/rlds_bridge_orig
```

## Live Run Checks

Attach to the current chain:

```bash
tmux attach -t univla_vla50k_then_finetune
```

Check active processes:

```bash
pgrep -af 'vla-scripts/train.py|vla-scripts/finetune_bridge.py|torch.distributed.run'
```

Check VLA speed:

```bash
tail -n 120 /NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong/outputs/univla_bridge_lam_local/run_logs/vla_bridge_dataset_local_lam_stage2_b200_tf21_vla50k_cleanenv.log
```

Check GPU util:

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits
```

Check env on one VLA worker:

```bash
PID=$(pgrep -f 'vla-scripts/train.py.*vla50k_cleanenv' | head -n 1)
tr '\0' '\n' < /proc/$PID/environ | rg 'UNIVLA_RLDS_LEN_OVERRIDE|UNIVLA_DUMMY_LATENT_ACTIONS|WANDB_MODE|UNIVLA_TRAJ|UNIVLA_FRAME|UNIVLA_TF_RAM|OMP_NUM_THREADS'
```

Expected env:

```text
UNIVLA_DUMMY_LATENT_ACTIONS=0
WANDB_MODE=online
OMP_NUM_THREADS=8
UNIVLA_TRAJ_THREADS=4
UNIVLA_TRAJ_READ_THREADS=4
UNIVLA_FRAME_THREADS=8
UNIVLA_TF_RAM_BUDGET_MB=512
```

There should be no `UNIVLA_RLDS_LEN_OVERRIDE` in VLA or finetune.

## Current Known-Good Run

Started on 2026-05-03:

```text
tmux session: univla_vla50k_then_finetune
VLA W&B run: https://wandb.ai/joonstack/univla_bridge_lam_local/runs/z4odey9n
VLA run note: bridge_dataset_local_lam_stage2_b200_tf21_vla50k_cleanenv
```

Expected duration at observed speed:

```text
VLA 50k: about 17-18 hours
convert: minutes to tens of minutes
Bridge finetune 30k: several hours to low tens of hours, depending on observed step time
```

## Factorized LAM 30k VLA Handoff - 2026-05-07

This is the known-good command shape for loading the factorized hyperbolic LAM
30k checkpoint into VLA pretraining. Keep this as the canonical reproduction
command for the local factorized Bridge run.

Important distinction: W&B records the script argv, but not the full distributed
wrapper and environment. Reproduce with the `tfcheck` Python and
`python -m torch.distributed.run`, not by copying only the W&B command text.

```bash
cd /NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong/UniVLA-hyper

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_MODE=online
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export UNIVLA_PROFILE_STEPS=20
export UNIVLA_DUMMY_LATENT_ACTIONS=0
export UNIVLA_VLA_DATALOADER_WORKERS=0
export UNIVLA_TRAJ_THREADS=4
export UNIVLA_TRAJ_READ_THREADS=4
export UNIVLA_FRAME_THREADS=8
export UNIVLA_TF_RAM_BUDGET_MB=512
export PYTHONPATH=/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong/UniVLA-hyper:${PYTHONPATH:-}

/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong/.venvs/univla-tfcheck-py310/bin/python \
  -m torch.distributed.run \
  --standalone --nnodes 1 --nproc-per-node 8 \
  vla-scripts/train.py \
  --vla.type prism-dinosiglip-224px+mx-bridge \
  --vla.max_steps 50000 \
  --vla.shuffle_buffer_size 20000 \
  --image_aug true \
  --pretrain_vlm /NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong/data/univla_checkpoints/prismatic-vlms/prism-dinosiglip-224px+7b \
  --lam_path /NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong/outputs/lam_bridge/logs/visual_vq_lam_bridge_hyperbolic_factorized_rad16_0to4_dir16_rad1_dir4tokens_prelift_hmax9_50k_workers2/epoch=0-step=30000.ckpt \
  --lam_kind visual_vq_factorized \
  --lam_config_path /NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong/UniVLA-hyper/latent_action_model/config.yaml \
  --codebook_size 32 \
  --latent_action_token_len 5 \
  --data_root_dir /NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong/data/rlds_bridge_orig \
  --run_root_dir /NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong/outputs/univla_bridge_factorized_lam30k \
  --wandb_project univla_bridge_lam_local \
  --wandb_entity joonstack \
  --run_id_note bridge_dataset_factorized_lam30k_rad16dir16_50k
```

Known-good live run:

```text
tmux session: vla_bridge_factorized_lam30k_50k_localproject
log: outputs/bridge_pipeline_logs/manual_vla_factorized_20260507/vla_bridge_factorized_lam30k_50k_localproject_accsplit.log
W&B: https://wandb.ai/joonstack/univla_bridge_lam_local/runs/n5yf89oh
```

Healthy profile from the local-project run:

```text
Threads per Dataset: [4]
Reads per Dataset: [4]

step 1: total=10.835s data_wait=1.267s dataset_fetch=0.661s collator_lam=0.598s
step 2: total=1.744s  data_wait=0.251s dataset_fetch=0.061s collator_lam=0.182s
step 3: total=1.289s  dataset_fetch=0.192s collator_lam=0.178s
step 14: total=1.270s dataset_fetch=0.120s collator_lam=0.180s
step 18: total=1.369s dataset_fetch=0.157s collator_lam=0.178s
```

Why the thread knobs were touched:

- The original symptom was `data_wait` around 50-60 seconds, which means GPU
  workers were starving on input rather than spending time in model compute.
- In `UniVLA_fresh`, these are optional env overrides read by the dataset code,
  not required exports in the fresh training scripts. If unset, Bridge uses
  roughly `traj/read=1/1` because the mixture has one dataset, while frame
  transforms default to `8`.
- `UNIVLA_TRAJ_THREADS`, `UNIVLA_TRAJ_READ_THREADS`, and
  `UNIVLA_FRAME_THREADS` are per rank / per dataset input-pipeline knobs. On an
  8-rank run, blindly increasing them multiplies CPU pressure across ranks.
- `1/1/1` can underfeed TF/RLDS; very high values can oversubscribe CPU and IO.
  The stable known-good value is `4/4/8`.
- Candidates like `6/4`, `6/6`, or `8/4` are only for short 200-step smoke
  tests. Do not replace the full-run recipe unless `dataset_fetch`,
  `data_wait`, and step time improve together.

Lessons from the failed attempts:

- `UNIVLA_DUMMY_LATENT_ACTIONS` must be off. The 2026-05-07 live run left it
  unset, which is equivalent to `0` because the code only enables dummy latent
  actions when the env var is exactly `1`. The reproduction command pins it to
  `0` to remove ambiguity.
- `UNIVLA_TF_RAM_BUDGET_MB=512` is essential for this VLA run. The code default
  is `1` MB in the RLDS dataset path, and that produced `dataset_fetch` around
  40 seconds in the slow run.
- Use `/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong/.venvs/univla-tfcheck-py310/bin/python`.
  The old repo `.venv` was not the known-good VLA environment.
- Keep `UNIVLA_VLA_DATALOADER_WORKERS=0` for this path. The collator does LAM
  work, and extra PyTorch workers are not the proven fix here.
- The correct W&B project for this local VLA run is
  `univla_bridge_lam_local`. A previous launch used
  `univla_bridge_lam_factorized`; that was the wrong project for this run.
- Current `dataset_fetch` is already about `0.06-0.19s` after warmup, and
  `collator_lam` is about `0.18s`. Further dataloading tuning is not worth
  interrupting a healthy full run.
