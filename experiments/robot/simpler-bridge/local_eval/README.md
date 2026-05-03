# Local SimpleREnv Evaluation Prep

This directory prepares a local SimpleREnv evaluation workspace for Bridge-style
policies after finetuning finishes.

Target models:

- `univla`: current UniVLA / custom latent-action run.
- `mvp-lam`: MVP-LAM-style UniVLA checkpoint with an action decoder.
- `lapa`: LAPA uses its own legacy SimplerEnv/ManiSkill2 launcher.
- `villa-x`: current public repo only provides LAM inference, not a Bridge action
  policy adapter. It needs an adapter/checkpoint before it can be evaluated here.
- `latent-action`: any local model that exposes the same UniVLA-style HF
  checkpoint plus `action_decoder-*.pt`.

## What This Sets Up

The MVP-LAM README expects a ManiSkill3 SimpleREnv checkout with:

```text
simpler_env/policies/univla/
real2sim_eval_maniskill3.py
```

The helper scripts copy the local UniVLA/MVP-LAM policy adapter into a
SimpleREnv checkout and provide one launcher for the four Bridge tasks:

```text
PutSpoonOnTableClothInScene-v1
PutCarrotOnPlateInScene-v1
StackGreenCubeOnYellowCubeBakedTexInScene-v1
PutEggplantInBasketScene-v1
```

## One-Time Prep

Use the local `third_party/SimplerEnv-maniskill3` checkout as the target
workspace. This was cloned from the official `simpler-env/SimplerEnv`
`maniskill3` branch because the local `LAPA/SimplerEnv` checkout is
ManiSkill2-style and imports `mani_skill2_real2sim`.

```bash
cd /NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong/UniVLA

./experiments/robot/simpler-bridge/local_eval/prepare_simplerenv_workspace.sh
```

If the target checkout is missing, use a fresh checkout:

```bash
git clone -b maniskill3 https://github.com/simpler-env/SimplerEnv.git \
  /NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong/third_party/SimplerEnv-maniskill3
./experiments/robot/simpler-bridge/local_eval/prepare_simplerenv_workspace.sh
```

## Evaluation Venv

The training venv is intentionally not modified. To create a separate eval venv:

```bash
./experiments/robot/simpler-bridge/local_eval/setup_eval_venv.sh
```

This may need network access because ManiSkill3 is installed from GitHub. The
venv has been smoke-tested for imports with ManiSkill3 Bridge task registration
and the UniVLA policy adapter.

## Official Model Zoo Downloads

Optional helper:

```bash
./experiments/robot/simpler-bridge/local_eval/download_model_zoo.sh
```

By default, this downloads LAPA's three official checkpoint files. Large 7B HF
snapshots are opt-in:

```bash
DOWNLOAD_MVP_SIMPLER=1 DOWNLOAD_VILLAX=1 \
  ./experiments/robot/simpler-bridge/local_eval/download_model_zoo.sh
```

Official references checked on 2026-05-03:

```text
MVP-LAM SimpleREnv model: JM-Lee/mvp-lam-7b-224-simpler
LAPA OpenX checkpoint: latent-action-pretraining/LAPA-7B-openx
villa-X model repo: microsoft/villa-x
```

## Current UniVLA Run

After the current `univla_vla50k_then_finetune` chain finishes, resolve its
finetune artifacts:

```bash
source ./experiments/robot/simpler-bridge/local_eval/resolve_univla_finetune_artifacts.sh
echo "$CKPT_PATH"
echo "$ACTION_DECODER_PATH"
```

Then run a smoke test:

```bash
PYTHON=/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong/.venvs/simplerenv-eval-py310/bin/python \
NUM_EPISODES=1 NUM_ENVS=1 SAVE_VIDEO=0 \
  ./experiments/robot/simpler-bridge/local_eval/run_univla_like_4task.sh
```

The local launcher defaults `UNIVLA_EVAL_ATTN_IMPLEMENTATION=sdpa` so evaluation
does not require a working `flash-attn` build. Set
`UNIVLA_EVAL_ATTN_IMPLEMENTATION=flash_attention_2` if that package is installed
and known-good in the eval venv.

The import smoke test currently emits TensorFlow GPU and SAPIEN Vulkan ICD
warnings on this machine. Policy import and ManiSkill3 task registration still
pass; if an actual episode fails at render/reset time, check the Vulkan ICD
installation first.

Full four-task run:

```bash
PYTHON=/NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong/.venvs/simplerenv-eval-py310/bin/python \
NUM_EPISODES=24 NUM_ENVS=1 SAVE_VIDEO=1 \
  ./experiments/robot/simpler-bridge/local_eval/run_univla_like_4task.sh
```

## Model Path Contract

`run_univla_like_4task.sh` expects:

```bash
export CKPT_PATH=/path/to/HF/finetuned/model
export ACTION_DECODER_PATH=/path/to/action_decoder-30000.pt
```

For UniVLA/MVP-LAM-style policies, `CKPT_PATH` must contain HF model files and
`dataset_statistics.json`; `ACTION_DECODER_PATH` must be the matching action
decoder weights.

Set `PRED_ACTION_HORIZON` to match the action decoder. The public UniVLA
SimpleREnv checkpoint uses horizon 10; the public MVP-LAM SimpleREnv checkpoint
uses `action_decoder-10000.pt` with horizon 5.

## LAPA

LAPA currently has its own legacy launcher and official checkpoint:

```bash
cd /NHNHOME/WORKSPACE/0526040036_A/BASE/user/01/youngjoonjeong/LAPA/SimplerEnv
bash scripts/lapa_bridge.sh
```

That script targets the older ManiSkill2 setup and uses `*-v0` task names. Keep
LAPA separate unless a ManiSkill3 LAPA adapter is added.

## Villa-X

The local `villa-x` repo currently releases LAM inference only. There is no
Bridge action policy adapter or SimpleREnv policy wrapper in the repo. To
evaluate it in SimpleREnv, add a policy adapter that implements:

```python
reset(task_description: str) -> None
step(image: np.ndarray, task_description: str) -> tuple[raw_action, action]
```

and emits the same action dict keys as the UniVLA adapter:

```text
world_vector
rot_axangle
gripper
terminate_episode
```
