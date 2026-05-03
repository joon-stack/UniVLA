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
