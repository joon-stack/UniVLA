# BridgeV2 LAM -> VLA Pipeline

This directory records the commands for the BridgeV2 LAM experiment.

## Reusable Factorized Pipeline

Do not create a new one-off shell file for every run. Use this entrypoint and override only environment variables:

```bash
vla-scripts/bridge_lam_pipeline/bridge_factorized_pipeline.sh <stage>
```

Stages:

```bash
# 1. Factorized hyperbolic visual VQ LAM
./vla-scripts/bridge_lam_pipeline/bridge_factorized_pipeline.sh train-lam

# 2. Bridge VLA pretraining from a LAM checkpoint
LAM_CKPT=/path/to/lam.ckpt \
VLA_RUN_NOTE=bridge_dataset_factorized_lam30k_rad16dir16_50k \
./vla-scripts/bridge_lam_pipeline/bridge_factorized_pipeline.sh train-vla

# 3. Convert a VLA checkpoint to HuggingFace format
VLA_CKPT_STEP=050000 \
./vla-scripts/bridge_lam_pipeline/bridge_factorized_pipeline.sh convert-hf

# 4. Simpler success50 finetuning
FINETUNE_BATCH_SIZE=16 \
FINETUNE_GRAD_ACCUMULATION_STEPS=1 \
./vla-scripts/bridge_lam_pipeline/bridge_factorized_pipeline.sh finetune-simpler
```

Current defaults are set for the factorized LAM/VLA run:

- LAM config: `latent_action_model/config/lam-visual-vq-bridge-hyperbolic-factorized-50k.yaml`
- LAM training sets `UNIVLA_RLDS_LEN_OVERRIDE=20000000` by default to keep Lightning from reaching the Bridge RLDS epoch-boundary rebuild/hang before `max_steps`.
- VLA/finetune explicitly do not inherit `UNIVLA_RLDS_LEN_OVERRIDE`.
- VLA ckpt search: `outputs/univla_bridge_factorized_lam30k/.../checkpoints/step-${VLA_CKPT_STEP}-*.pt`
- HF output: `outputs/hf_univla_bridge_factorized_lam30k_rad16dir16_50k_step${VLA_CKPT_STEP}`
- Simpler data: `data/simpler_mvplam_noeval_success50_rlds`
- Pipeline finetune defaults: 8 GPU, batch 16, grad accumulation 1, window 10, max steps 20000, scheduler off.
- Known-good Simpler 4-GPU finetune is documented below and overrides these defaults.

Common env overrides:

```bash
ROOT=/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong
REPO=${ROOT}/UniVLA-hyper
UNIVLA_PIPELINE_PYTHON=${ROOT}/UniVLA/.venvs/sibal2/bin/python
PYTHON=${ROOT}/UniVLA/.venvs/sibal2/bin/python
NPROC_PER_NODE=8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

LAM_CKPT=${ROOT}/outputs/lam_bridge/logs/.../epoch=0-step=30000.ckpt
LAM_CONFIG_PATH=${REPO}/latent_action_model/config.yaml
VLA_CKPT_STEP=050000
VLA_CKPT_NAME=step-050000-epoch-05-loss=0.5906.pt
HF_VLA_DIR=${ROOT}/outputs/hf_univla_bridge_factorized_lam30k_rad16dir16_50k_step050000
SIMPLER_DATA_ROOT=${ROOT}/data/simpler_mvplam_noeval_success50_rlds
FINETUNE_USE_SCHEDULER=false
```

The older `run_*.sh` files in this directory are legacy records. Keep them for provenance. For new runs, prefer `bridge_factorized_pipeline.sh` plus environment overrides, except for the known-good Simpler wrapper below.

## Known-good Simpler finetune path

Known-good wrappers in this directory:

```bash
# 1. Factorized LAM, 8 GPU, sibal2 env
./vla-scripts/bridge_lam_pipeline/run_train_lam_factorized_8gpu_sibal2.sh

# 2. Bridge VLA pretraining with the factorized LAM checkpoint, 8 GPU, sibal2 env
./vla-scripts/bridge_lam_pipeline/run_train_vla_factorized_8gpu_sibal2.sh

# 3. Bridge VLA pretraining with the euclidean visual VQ LAM checkpoint, 8 GPU, sibal2 env
./vla-scripts/bridge_lam_pipeline/run_train_vla_euclidean_visual_vq_8gpu_sibal2.sh

# 4. Simpler success50 finetune, 4 GPU, sibal2 env
./vla-scripts/bridge_lam_pipeline/run_finetune_simpler_success50_4gpu_sibal2.sh
```

Use this wrapper for the factorized LAM Simpler success50 finetune:

```bash
cd /NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong/UniVLA-hyper
./vla-scripts/bridge_lam_pipeline/run_finetune_simpler_success50_4gpu_sibal2.sh
```

This is the tested 4-GPU configuration:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3
NPROC_PER_NODE=4
UNIVLA_PIPELINE_PYTHON=${ROOT}/UniVLA/.venvs/sibal2/bin/python
SIMPLER_DATA_ROOT=/tmp/youngjoon_univla_data/simpler_mvplam_noeval_success50_rlds
FINETUNE_BATCH_SIZE=32
FINETUNE_GRAD_ACCUMULATION_STEPS=1
FINETUNE_MAX_STEPS=20000
FINETUNE_SAVE_STEPS=5000
FINETUNE_SHUFFLE_BUFFER_SIZE=512
FINETUNE_IMAGE_AUG=true
UNIVLA_LAM_IN_TRAIN_LOOP=1
UNIVLA_BATCH_LAM_IN_COLLATOR=1
UNIVLA_VLA_DATALOADER_WORKERS=2
UNIVLA_VLA_DATALOADER_PREFETCH_FACTOR=2
UNIVLA_TRAJ_THREADS=4
UNIVLA_TRAJ_READ_THREADS=4
UNIVLA_FRAME_THREADS=4
UNIVLA_TF_RAM_BUDGET_MB=512
```

Expected early profile after loader warmup:

```text
FINETUNE_PROFILE step=000002 data_wait=0.001s ... total=1.416s
FINETUNE_PROFILE step=000003 data_wait=0.003s ... total=1.066s
FINETUNE_PROFILE step=000010 data_wait=0.002s ... total=1.056s
FINETUNE_COLLATOR_PROFILE ... collator_lam=0.000s
```

If `data_wait` is tens of seconds, do not tune blindly. Check these first:

```bash
pgrep -af 'finetune_bridge.py|torch.distributed.run|bridge_factorized_pipeline|finetune-simpler'
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits
tail -f ${ROOT}/outputs/bridge_pipeline_logs/factorized_simpler_success50_sibal2_workers2_4gpu/finetune_simpler.log
```

The common failure modes are:

- Wrong Python: anything under `UniVLA-hyper/.venv/bin/python` is suspect for this finetune. The known-good env is `${ROOT}/UniVLA/.venvs/sibal2/bin/python`.
- Duplicate run: a stale shell or tmux can restart an old 8-GPU command. Kill duplicates before profiling.
- LAM in DataLoader/collator: if `UNIVLA_LAM_IN_TRAIN_LOOP=0`, LAM/token creation can move back into the input path and stall GPU training.
- No DataLoader prefetch: `UNIVLA_VLA_DATALOADER_WORKERS=2` plus `UNIVLA_LAM_IN_TRAIN_LOOP=1` was the setting that made `data_wait` collapse to near zero.

### What was fixed

The sustainable fix is not a memorized one-off command. It is three invariants:

1. The pipeline Python is explicit. `bridge_factorized_pipeline.sh` resolves Python as:

   ```bash
   UNIVLA_PIPELINE_PYTHON > PYTHON > ${ROOT}/UniVLA/.venvs/sibal2/bin/python
   ```

2. LAM token materialization is out of the DataLoader hot path for finetune:

   ```text
   DataLoader / collator:
     stack VLA pixels, LAM pixels, language, actions
     do not run LAM
     do not tokenize ACT tokens

   train loop on each GPU rank:
     move LAM pixels to GPU
     run latent_action_model.vq_encode(...)
     build ACT token prompt / labels
     run VLA forward/backward
   ```

3. Success is verified by numbers, not by process existence:

   ```text
   data_wait ~= 0
   collator_lam = 0
   step total ~= 1.0-1.4s after warmup
   exactly 4 training ranks on GPUs 0,1,2,3
   ```

### Relation to the UniVLA finetune script

Reference script:

```bash
${ROOT}/UniVLA/vla-scripts/finetune_simpler_success50_ws10_4gpu_.sh
${ROOT}/UniVLA/vla-scripts/finetune_simpler_success10_ws10_4gpu_auto_retry.sh
```

The current factorized finetune intentionally keeps the same outer training recipe:

- 4 GPUs
- batch size 32
- gradient accumulation 1
- max steps 20000
- save steps 5000
- window size 10
- shuffle buffer 512
- image augmentation on
- scheduler off
- LoRA/VLA/action-decoder training loop unchanged in structure
- same Simpler task mix contract: `carrot`, `eggplant`, `spoon`, `stack`

It is not bit-identical to UniVLA, because the LAM contract changed:

- UniVLA loads `ControllableDINOLatentActionModel` and assumes the original 4 latent action tokens.
- UniVLA-hyper can load `visual_vq_factorized` and uses `latent_action_token_len=5` for radius + direction tokens.
- UniVLA creates LAM tokens inside `RLDSBatchTransformLIBERO_withHis` one example at a time.
- UniVLA-hyper can return raw LAM image pairs from the batch transform and create the factorized ACT tokens in the GPU train loop when `UNIVLA_LAM_IN_TRAIN_LOOP=1`.
- UniVLA uses `num_workers=0`; the known-good UniVLA-hyper path uses `num_workers=2` only after LAM is removed from the worker path.

So the right claim is:

```text
Same downstream finetune objective and optimizer/training structure.
Different latent-action tokenizer implementation and data plumbing.
The data plumbing change is intended to be behavior-preserving for labels,
but much faster because it batches LAM token creation on GPU instead of
blocking the input pipeline.
```

Current baseline:
- Dataset: BridgeV2 RLDS via `data_mix=bridge`
- TFDS root: `/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong/data/rlds_bridge_orig`
- LAM stage1: batch 64 per GPU, 8 GPUs, global batch 512, 100k steps
- LAM stage2: batch 64 per GPU, 8 GPUs, global batch 512, 40k steps from the stage1 60k checkpoint
- VLM: reuse UniVLA base VLM `prism-dinosiglip-224px+7b`
- VLA: Bridge pretraining with the newly trained LAM stage2 checkpoint

Run order:

```bash
# 0. Check the currently running LAM stage1.
./vla-scripts/bridge_lam_pipeline/00_monitor_lam_stage1.sh

# 1. Run LAM stage2 from the stage1 60k checkpoint.
./vla-scripts/bridge_lam_pipeline/01_train_lam_stage2_bridge.sh

# 2. Validate the base VLM checkpoint before policy pretraining.
./vla-scripts/bridge_lam_pipeline/02_prepare_vlm_bridge.sh

# 3. After LAM stage2 finishes, run BridgeV2 VLA/policy pretraining.
./vla-scripts/bridge_lam_pipeline/03_train_vla_bridge_with_lam.sh
```

Long jobs are best launched inside tmux, for example:

```bash
tmux new-session -d -s lam_bridge_stage2 "./vla-scripts/bridge_lam_pipeline/01_train_lam_stage2_bridge.sh"
tmux new-session -d -s vla_bridge_lam "./vla-scripts/bridge_lam_pipeline/03_train_vla_bridge_with_lam.sh"
```
