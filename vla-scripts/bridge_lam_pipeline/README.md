# BridgeV2 LAM -> VLA Pipeline

This directory records the commands for the BridgeV2 LAM experiment.

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
