# LAM Hyperbolic Experiment Log - 2026-05-06

This file records the concrete training, monitoring, and stop commands used while debugging the UniVLA-hyper visual VQ LAM Bridge run.

## Current Decision Snapshot

- RadProg target: `emb` / VQ embedding. The claim is that VQ preserves this geometry when quantization error is low.
- VQ dead-code restart interval: `30000` steps.
- Spread diagnostics: log cosine and variance separately for `emb`, `z_q`, and the VQ codebook.
- Order diagnostics: log RadProg order fractions for both `emb` and `z_q`; only `emb` receives the RadProg loss.
- Current config: `latent_action_model/config/lam-visual-vq-bridge-hyperbolic-prelift-50k.yaml`.
- Current run name: `visual_vq_lam_bridge_hyperbolic_prelift_emb_scale0p05_vqinit0p40_radprog_emb_codeaux0p25_hmax9_50k_workers2_restart30k`.
- Current VQ init range: `UNIVLA_VQ_INIT_RANGE=0.40`.
- Current codebook auxiliary loss: selected-code RadProg on `z_self_code`, `z_mid_code`, and `z_future_code` at `0.25x` the emb RadProg weights.
- Fallback candidate if the current run does not recover quantized geometry: keep `h2` fixed at max offset and sample only `h1` randomly.

## Current 10K Checkpoint Criteria

Run to about 10000 steps before judging, unless q/commit explodes early.

- `train/q_loss` and `train/commit_loss` should stay near the stable regime and preferably move below `0.05`.
- `train/code_usage` should rise above `0.5` and ideally continue toward full code usage.
- `scale/cosine/emb` should decrease or stabilize rather than monotonically collapse upward.
- `scale/cosine/z_q` and `scale/cosine/codebook` should not remain pinned above `0.8`.
- `radprog/order/radial_frac` and `radprog/order/norm_frac` show whether the `emb` geometry is improving.
- `radprog/z_q/order/radial_frac` and `radprog/z_q/order/norm_frac` show whether the quantized latent action preserves that ordering.

## Primary Hyperbolic Visual VQ Command

The reusable script currently used for the hyperbolic 8 GPU run is:

```bash
outputs/bridge_pipeline_logs/manual_hyper_prelift_emb_20260506_084648/run_hyperbolic_prelift_emb_50k_workers2.sh
```

Its effective command is:

```bash
cd /NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong/UniVLA-hyper/latent_action_model

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCHRUN=/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong/UniVLA/.venv/bin/torchrun
export LAM_NPROC_PER_NODE=8
export UNIVLA_LAM_NUM_WORKERS=2
export UNIVLA_LAM_PROFILE_STEPS=5
export UNIVLA_LAM_MODULE_PROFILE_STEPS=5
export UNIVLA_VQ_INIT_RANGE=0.40
export UNIVLA_VQ_RESTART_NOISE=0.01
export WANDB_MODE=online
export WANDB_PROJECT=univla_lam_bridge
export WANDB_ENTITY=joonstack

"${TORCHRUN}" --standalone --nnodes 1 --nproc-per-node "${LAM_NPROC_PER_NODE}" main_visual_vq.py fit \
  --config config/lam-visual-vq-bridge-hyperbolic-prelift-50k.yaml
```

## Factorized VQ LAM Checkpoint Used For VLA - 2026-05-07

The VLA handoff uses the factorized hyperbolic LAM checkpoint below:

```text
/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong/outputs/lam_bridge/logs/visual_vq_lam_bridge_hyperbolic_factorized_rad16_0to4_dir16_rad1_dir4tokens_prelift_hmax9_50k_workers2/epoch=0-step=30000.ckpt
```

The matching config snapshot used by VLA is:

```text
/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong/UniVLA-hyper/latent_action_model/config.yaml
```

Do not assume the mutable training config path still reconstructs this exact
checkpoint. The active file
`config/lam-visual-vq-bridge-hyperbolic-factorized-50k.yaml` has since been
edited for later experiments. For VLA, the stable handoff artifact is the 30k
checkpoint plus the copied `latent_action_model/config.yaml` snapshot.

Factorized token contract:

```text
LAM kind: visual_vq_factorized
radius codes: 16
direction codes: 16
radius values in snapshot: 0.0 to 4.0, linearly spaced
VLA latent_action_token_len: 5
VLA codebook_size: 32
VLA token layout: 1 radius token + 4 direction tokens
```

Reusable LAM launch shape for the factorized run:

```bash
cd /NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong/UniVLA-hyper/latent_action_model

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCHRUN=/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong/UniVLA/.venv/bin/torchrun
export LAM_NPROC_PER_NODE=8
export UNIVLA_LAM_NUM_WORKERS=2
export UNIVLA_LAM_PROFILE_STEPS=5
export UNIVLA_LAM_MODULE_PROFILE_STEPS=5
export UNIVLA_TRAJ_THREADS=4
export UNIVLA_TRAJ_READ_THREADS=4
export UNIVLA_FRAME_THREADS=8
export UNIVLA_TF_RAM_BUDGET_MB=512
export WANDB_MODE=online
export WANDB_PROJECT=univla_lam_bridge
export WANDB_ENTITY=joonstack

./train_lam_bridge_visual_vq_factorized.sh
```

The script resolves to:

```bash
${TORCHRUN} --standalone --nnodes 1 --nproc-per-node ${LAM_NPROC_PER_NODE:-8} main_visual_vq.py fit \
  --config config/lam-visual-vq-bridge-hyperbolic-factorized-50k.yaml
```

Important lesson: `UNIVLA_TRAJ_THREADS=4`,
`UNIVLA_TRAJ_READ_THREADS=4`, `UNIVLA_FRAME_THREADS=8`, and
`UNIVLA_TF_RAM_BUDGET_MB=512` are part of the known-good input pipeline recipe.
The thread values were introduced because `data_wait` was the visible failure
mode. In the final VLA diagnosis, the missing `TF_RAM=512` setting was the
largest difference between the slow and healthy commands.

## Runs Started And Stopped

### Original 100k Hyperbolic Visual VQ Queue Run

Initial command shape:

```bash
cd /NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong/UniVLA-hyper/latent_action_model
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
LAM_NPROC_PER_NODE=8 \
UNIVLA_TRAJ_THREADS=1 \
UNIVLA_TRAJ_READ_THREADS=1 \
UNIVLA_FRAME_THREADS=1 \
UNIVLA_TF_RAM_BUDGET_MB=1 \
WANDB_MODE=online \
./train_lam_bridge_visual_vq.sh \
  --trainer.logger '[{"class_path":"lightning.pytorch.loggers.WandbLogger","init_args":{"project":"univla_lam_bridge","entity":"joonstack","name":"visual_vq_lam_bridge_hyperbolic_radprog_hmax9_100k_tf215_8gpu_freshthreads"}}]'
```

Effective torchrun:

```bash
/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong/UniVLA/.venv/bin/torchrun \
  --standalone --nnodes 1 --nproc-per-node 8 \
  main_visual_vq.py fit \
  --config config/lam-visual-vq-bridge.yaml \
  --trainer.logger '[{"class_path":"lightning.pytorch.loggers.WandbLogger","init_args":{"project":"univla_lam_bridge","entity":"joonstack","name":"visual_vq_lam_bridge_hyperbolic_radprog_hmax9_100k_tf215_8gpu_freshthreads"}}]'
```

Log:

```text
outputs/bridge_pipeline_logs/controlled_univla_queue_20260505_175856/1_hyperbolic_radprog_visual_vq_lam_bridge_100k.log
```

Observed profile issue:

```text
LAM_PROFILE step=000001 data_wait=61.135s shared_step=4.426s backward=0.746s total=5.240s
LAM_PROFILE step=000002 data_wait=58.925s shared_step=0.355s backward=1.217s total=1.621s
LAM_PROFILE step=000003 data_wait=58.830s shared_step=0.311s backward=0.486s total=0.821s
```

### Hyperbolic Prelift Scale 0.15 Run

Launched with the reusable script above when the config used `hyperbolic_prelift_scale: 0.15`.

Log:

```text
outputs/bridge_pipeline_logs/manual_hyper_prelift_emb_20260506_084648/hyperbolic_prelift_emb_radprog_hmax9_50k_workers2.log
```

W&B:

```text
https://wandb.ai/joonstack/univla_lam_bridge/runs/39vtxha5
```

Stopped after code usage stayed low around the 2000-step dead-code restart checkpoint.

### Hyperbolic Prelift Scale 0.05 Run

Launched with the reusable script above after changing the config to `hyperbolic_prelift_scale: 0.05`.

Log:

```text
outputs/bridge_pipeline_logs/manual_hyper_prelift_emb_20260506_084648/hyperbolic_prelift_emb_scale0p05_radprog_hmax9_50k_workers2.log
```

W&B:

```text
https://wandb.ai/joonstack/univla_lam_bridge/runs/jskp58d6
```

Stopped manually:

```bash
tmux send-keys -t lam_hyper_prelift_50k C-c
```

Last useful status before stop:

```text
step ~15726
train/code_usage=1
train/q_loss=0.008-0.010 range
scale/latent_action/emb_norm ~4.24
scale/latent_action/z_q_norm ~4.16
scale/radprog/prelift_norm_future ~4.76
```

### Restart30k Smoke Run

Launched after changing `vq_restart_interval_steps` to `30000` and adding `_restart30k` to the run name:

```bash
tmux new-session -d -s lam_hyper_prelift_50k_restart30k \
  'cd /NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong/UniVLA-hyper && bash outputs/bridge_pipeline_logs/manual_hyper_prelift_emb_20260506_084648/run_hyperbolic_prelift_emb_50k_workers2.sh > outputs/bridge_pipeline_logs/manual_hyper_prelift_emb_restart30k_20260506_1334/hyperbolic_prelift_emb_scale0p05_radprog_hmax9_50k_workers2_restart30k.log 2>&1'
```

Stopped manually before training reached the first batch:

```bash
tmux send-keys -t lam_hyper_prelift_50k_restart30k C-c
```

Log:

```text
outputs/bridge_pipeline_logs/manual_hyper_prelift_emb_restart30k_20260506_1334/hyperbolic_prelift_emb_scale0p05_radprog_hmax9_50k_workers2_restart30k.log
```

### VQ Init 0.40 Restart30k Run

Launched after changing `UNIVLA_VQ_INIT_RANGE` from `0.25` to `0.40`.

tmux session:

```text
lam_hyper_prelift_50k_emb_vqinit0p40_restart30k
```

Log:

```text
outputs/bridge_pipeline_logs/manual_hyper_prelift_emb_vqinit0p40_restart30k_20260506/hyperbolic_prelift_emb_scale0p05_vqinit0p40_radprog_emb_50k_workers2_restart30k.log
```

W&B:

```text
https://wandb.ai/joonstack/univla_lam_bridge/runs/oeeeyn92
```

Early status at step 178:

```text
loss/total=0.704519
train/mse_loss=0.223216
train/q_loss=0.036927
train/commit_loss=0.036927
train/code_usage=0.125
radprog/order/radial_frac=0.839623
radprog/order/norm_frac=0.811321
radprog/z_q/order/radial_frac=0.160377
radprog/z_q/order/norm_frac=0.132075
scale/cosine/emb=0.833984
scale/cosine/z_q=0.930176
scale/cosine/codebook=0.013802
scale/variance/emb=0.031053
scale/variance/z_q=0.003246
scale/variance/codebook=0.048370
scale/radprog/prelift_norm_self=0.402997
scale/radprog/prelift_norm_mid=2.963986
scale/radprog/prelift_norm_future=4.268975
scale/radprog/z_q/prelift_norm_self=2.323071
scale/radprog/z_q/prelift_norm_mid=2.551210
scale/radprog/z_q/prelift_norm_future=2.557887
```

Interpretation: `emb` order is strong early, but quantized `z_q` order is still weak and `z_q` cosine is high. Continue watching before judging.

### Codebook Auxiliary RadProg 0.25x Patch

Decision:

```text
Keep emb RadProg unchanged.
Add auxiliary RadProg on selected codebook vectors only:
  RadProg(z_self_code, z_mid_code, z_future_code)
Weight it by radprog_codebook_aux_weight=0.25.
Do not add z_self to train/code_usage.
```

Code changes:

```text
latent_action_model/genie/modules/lam_visual_vq.py
  outputs["radprog_code_self"] = z_self_code
  outputs["radprog_code_mid"] = z_mid_code
  outputs["radprog_code_future"] = z_future_code

latent_action_model/genie/model_visual_vq.py
  radprog_codebook_aux_weight config argument
  codebook RadProg loss and logs under:
    loss/radprog_codebook_radial
    loss/radprog_codebook_progress
    radprog/codebook/order/*
    scale/radprog/codebook/*
```

Config:

```text
radprog_codebook_aux_weight: 0.25
```

Proposed restart command:

```bash
tmux new-session -d -s lam_hyper_prelift_50k_emb_codeaux0p25_restart30k \
  'cd /NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong/UniVLA-hyper && bash outputs/bridge_pipeline_logs/manual_hyper_prelift_emb_20260506_084648/run_hyperbolic_prelift_emb_50k_workers2.sh > outputs/bridge_pipeline_logs/manual_hyper_prelift_emb_codeaux0p25_restart30k_20260506/hyperbolic_prelift_emb_scale0p05_vqinit0p40_radprog_emb_codeaux0p25_50k_workers2_restart30k.log 2>&1'
```

Launched run:

```text
tmux: lam_hyper_prelift_50k_emb_codeaux0p25_restart30k
W&B: https://wandb.ai/joonstack/univla_lam_bridge/runs/bm003tad
log: outputs/bridge_pipeline_logs/manual_hyper_prelift_emb_codeaux0p25_restart30k_20260506/hyperbolic_prelift_emb_scale0p05_vqinit0p40_radprog_emb_codeaux0p25_50k_workers2_restart30k.log
```

Early status at step 87:

```text
loss/total=0.973647
train/q_loss=0.017555
train/commit_loss=0.017555
train/code_usage=0.125
loss/radprog_codebook_radial=0.680072
loss/radprog_codebook_progress=1.483853
radprog/codebook/order/radial_frac=0.018868
radprog/codebook/order/norm_frac=0.0
scale/radprog/codebook/r_self=4.923010
scale/radprog/codebook/r_mid=4.982555
scale/radprog/codebook/r_future=4.987135
```

### Zero Shells Init + Codebook Auxiliary RadProg 1.0x Run

Decision:

```text
Use one shared codebook.
Initialize code[0] as the zero/no-op candidate.
Initialize codes 1-5 at radius 1.0.
Initialize codes 6-10 at radius 2.5.
Initialize codes 11-15 at radius 4.0.
Increase radprog_codebook_aux_weight from 0.25 to 1.0.
```

Environment:

```bash
export UNIVLA_VQ_INIT_MODE=zero_shells
export UNIVLA_VQ_INIT_SHELL_RADII=1.0,2.5,4.0
export UNIVLA_VQ_INIT_SHELL_COUNTS=5,5,5
```

Init sanity check:

```text
[0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.5, 2.5, 2.5, 2.5, 2.5, 4.0, 4.0, 4.0, 4.0, 4.0]
```

Launched run:

```text
tmux: lam_hyper_prelift_zero_shells_codeaux1p0_restart30k
W&B: https://wandb.ai/joonstack/univla_lam_bridge/runs/kqpyf3cu
log: outputs/bridge_pipeline_logs/manual_hyper_prelift_emb_zero_shells_codeaux1p0_restart30k_20260506/hyperbolic_prelift_emb_zero_shells_radprog_emb_codeaux1p0_50k_workers2_restart30k.log
```

Early status at step 51:

```text
loss/total=1.885275
train/mse_loss=0.341689
train/q_loss=0.068491
train/commit_loss=0.068491
train/code_usage=0.125
loss/radprog_codebook_radial=0.693147
loss/radprog_codebook_progress=1.055645
radprog/codebook/order/radial_frac=0.0
radprog/codebook/order/norm_frac=0.0
scale/radprog/codebook/r_self=0.466359
scale/radprog/codebook/r_mid=2.010128
scale/radprog/codebook/r_future=2.010128
```

Interpretation: zero shell immediately lowers `codebook/r_self`, but q/commit are higher than the previous init early. Watch 200-1000 steps before judging.

## Monitoring Commands Used

Check active training processes:

```bash
pgrep -af "main_visual_vq.py fit --config config/lam-visual-vq-bridge-hyperbolic-prelift-50k.yaml"
```

Check tmux session:

```bash
tmux has-session -t lam_hyper_prelift_50k
tmux has-session -t lam_hyper_prelift_50k_restart30k
```

Tail latest log:

```bash
tail -c 4000 outputs/bridge_pipeline_logs/manual_hyper_prelift_emb_vqinit0p40_restart30k_20260506/hyperbolic_prelift_emb_scale0p05_vqinit0p40_radprog_emb_50k_workers2_restart30k.log
```

Extract dead-code restart events:

```bash
tr '\r' '\n' < outputs/bridge_pipeline_logs/manual_hyper_prelift_emb_20260506_084648/hyperbolic_prelift_emb_scale0p05_radprog_hmax9_50k_workers2.log | rg "Restarting [0-9]+ codes"
```

Query the 0.05 W&B run:

```bash
/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong/UniVLA/.venv/bin/python -c "import wandb; api=wandb.Api(); run=api.run('joonstack/univla_lam_bridge/jskp58d6'); keys=['_step','train/code_usage','train/q_loss','train/commit_loss','train/mse_loss','loss/total','scale/latent_action/emb_norm','scale/latent_action/z_q_norm','scale/radprog/prelift_norm_self','scale/radprog/prelift_norm_mid','scale/radprog/prelift_norm_future','scale/cosine_similarity','scale/variance']; rows=[r for r in run.scan_history(keys=keys, page_size=2000) if r.get('train/code_usage') is not None]; print(rows[-1] if rows else None)"
```

## Next Intended Command

The current active run was launched with:

```bash
tmux new-session -d -s lam_hyper_prelift_50k_emb_vqinit0p40_restart30k \
  'cd /NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong/UniVLA-hyper && bash outputs/bridge_pipeline_logs/manual_hyper_prelift_emb_20260506_084648/run_hyperbolic_prelift_emb_50k_workers2.sh > outputs/bridge_pipeline_logs/manual_hyper_prelift_emb_vqinit0p40_restart30k_20260506/hyperbolic_prelift_emb_scale0p05_vqinit0p40_radprog_emb_50k_workers2_restart30k.log 2>&1'
```

If this run does not recover the `z_q` geometry, the next planned ablation is an offset sampler variant where `h2` is fixed to the max offset and only `h1` is sampled randomly.

## Latent Action MI Evidence - 2026-05-07

The current evidence note for `z_q -> action` information and radius ablation is:

```text
docs/latent-action-mi-evidence-20260507.md
```

Short result:

```text
Main metric: raw_global(z_q) -> zscore(action sequence)
KSG k=5, samples=1024, forced k=9 cache

dim 256:
factorized hyperbolic 0.836
euclidean visual VQ   0.732
univla stage2         0.475
```

Radius ablation at 256D:

```text
full factorized hyperbolic z_q 0.825
no-radius token_unit          0.290
no-radius sample_unit         0.259
```

Interpretation: the recommended main metric preserves the radial structure of
`z_q` while removing arbitrary global scale, and the radius ablation supports
the claim that the hyperbolic/factorized model carries action-relevant
information in radial magnitude.
