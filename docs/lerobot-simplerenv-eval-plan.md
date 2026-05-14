# LeRobot Policies in SimplerEnv Eval

This is the concrete eval integration plan for LeRobot policies trained on the
local Simpler success50 LeRobot dataset:

- `pi05`
- `smolvla`
- `pi0_fast`
- `diffusion`

The goal is to evaluate these policies in the external SimplerEnv ManiSkill3
workspace with the same Bridge task loop used by the current UniVLA adapter.
This document describes the code changes to make. It does not require running
eval on the training server.

## Current State

The repo-side SimplerEnv adapter is:

```text
experiments/robot/simpler-bridge/real2sim_eval_maniskill3.py
experiments/robot/simpler-bridge/policies/univla/univla_model.py
experiments/robot/simpler-bridge/local_eval/prepare_simplerenv_workspace.sh
experiments/robot/simpler-bridge/local_eval/run_univla_like_4task.sh
```

`real2sim_eval_maniskill3.py` currently supports only:

```python
elif args.model == "univla":
    from simpler_env.policies.univla.univla_model import UniVLABridgeInference
```

The eval loop calls:

```python
raw_action, action = model.step(images[-1].cpu().numpy()[0], instruction[0])
action = torch.cat([
    torch.tensor(action["world_vector"][None, :]),
    torch.tensor(action["rot_axangle"][None, :]),
    torch.tensor(action["gripper"][None, :]),
], dim=1)
obs, reward, terminated, truncated, info = env.step(action)
```

So a LeRobot adapter only needs to implement the same contract:

```python
reset(task_description: str) -> None
step(image: np.ndarray, task_description: str) -> tuple[raw_action, action]
```

and return an `action` dict containing:

```text
world_vector
rot_axangle
gripper
terminate_episode
```

Keep `--num-envs 1` unless the eval loop is vectorized. The current loop takes
only `images[-1][0]` and `instruction[0]`, so multi-env execution is not really
batched through the policy adapter.

## Checkpoint Path Contract

LeRobot saves checkpoints under:

```text
${OUTPUT_DIR}/checkpoints/${STEP}/pretrained_model/
${OUTPUT_DIR}/checkpoints/last -> ${STEP}
```

For eval, pass the `pretrained_model` directory, not the output root:

```bash
export POLICY_PATH=/NHNHOME/WORKSPACE/0526040036_A/user/01/youngjoonjeong/outputs/lerobot_pi05_simpler_success50_nobase_gbs128_4gpu/checkpoints/last/pretrained_model
```

Equivalent policy paths:

```bash
PI05_POLICY_PATH="${ROOT}/outputs/lerobot_pi05_simpler_success50_nobase_gbs128_4gpu/checkpoints/last/pretrained_model"
SMOLVLA_POLICY_PATH="${ROOT}/outputs/lerobot_smolvla_simpler_success50_nobase_gbs128_4gpu/checkpoints/last/pretrained_model"
PI0_FAST_POLICY_PATH="${ROOT}/outputs/lerobot_pi0_fast_simpler_success50_nobase_gbs128_4gpu/checkpoints/last/pretrained_model"
DIFFUSION_POLICY_PATH="${ROOT}/outputs/lerobot_diffusion_simpler_success50_nobase_gbs128_4gpu/checkpoints/last/pretrained_model"
```

The Python process that runs SimplerEnv must be able to import both
`simpler_env` and `lerobot`. You cannot import LeRobot from a different venv.
Either install LeRobot policy deps into the SimplerEnv eval venv, or install
SimplerEnv into the LeRobot no-BASE venv.

## Dataset Contract To Preserve

The converter used these feature names:

```text
image:  observation.images.primary, shape (480, 640, 3), uint8 in dataset
state:  observation.state, shape (7,), all zeros
action: action, shape (7,), names x y z roll pitch yaw gripper
task:   lowercased language instruction
```

For eval, feed exactly the same keys:

```python
batch = {
    "observation.images.primary": image_tensor_chw_float_0_to_1,
    "observation.state": torch.zeros(7, dtype=torch.float32),
    "task": task_description.lower(),
}
```

Do not add real proprio unless the training converter is changed and the model
is retrained. The current runs were trained with a zero state placeholder.

Do not manually unnormalize actions. Load the saved LeRobot postprocessor from
the checkpoint and call it after `policy.select_action`.

The converted action stats show the gripper dimension is already `-1` or `1`,
so SimplerEnv should receive:

```python
gripper = 1.0 if action_7d[6] > 0.0 else -1.0
```

This differs from the UniVLA adapter, where the decoded `open_gripper` was
thresholded at `0.5`.

## Add A New Policy Adapter

Add:

```text
experiments/robot/simpler-bridge/policies/lerobot_bridge/__init__.py
experiments/robot/simpler-bridge/policies/lerobot_bridge/lerobot_bridge_model.py
```

Recommended `lerobot_bridge_model.py`:

```python
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transforms3d.euler import euler2axangle


class LeRobotBridgeInference:
    def __init__(
        self,
        policy_path: str,
        policy_setup: str = "widowx_bridge",
        image_key: str = "observation.images.primary",
        state_key: str = "observation.state",
        action_scale: float = 1.0,
        device: Optional[str] = None,
    ) -> None:
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        self.policy_setup = policy_setup
        self.image_key = image_key
        self.state_key = state_key
        self.action_scale = action_scale
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_path = self._resolve_policy_path(policy_path)

        self.cfg = PreTrainedConfig.from_pretrained(self.policy_path)
        self.cfg.device = self.device
        self.cfg.pretrained_path = Path(self.policy_path)

        policy_cls = get_policy_class(self.cfg.type)
        self.policy = policy_cls.from_pretrained(self.policy_path, config=self.cfg)
        self.policy.eval()

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.cfg,
            pretrained_path=self.policy_path,
            preprocessor_overrides={"device_processor": {"device": self.device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )

        state_feature = self.cfg.input_features.get(self.state_key)
        self.state_dim = state_feature.shape[0] if state_feature is not None else 7
        self.task_description = None

    @staticmethod
    def _resolve_policy_path(path: str) -> str:
        p = Path(path).expanduser()
        if (p / "checkpoints" / "last" / "pretrained_model").is_dir():
            p = p / "checkpoints" / "last" / "pretrained_model"
        elif (p / "pretrained_model").is_dir():
            p = p / "pretrained_model"
        if not (p / "config.json").is_file():
            raise FileNotFoundError(
                f"Expected a LeRobot pretrained_model directory with config.json, got: {p}"
            )
        return str(p)

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.policy.reset()

    def step(self, image: np.ndarray, task_description: Optional[str] = None, *args, **kwargs):
        if task_description is not None and task_description != self.task_description:
            self.reset(task_description)

        batch = self._make_lerobot_batch(image, self.task_description or task_description or "")
        batch = self.preprocessor(batch)

        with torch.inference_mode():
            action = self.policy.select_action(batch)
            action = self.postprocessor(action)

        action_7d = action.squeeze(0).detach().cpu().float().numpy()
        raw_action, simpler_action = self._to_simpler_action(action_7d)
        return raw_action, simpler_action

    def _make_lerobot_batch(self, image: np.ndarray, task_description: str) -> dict:
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        image_tensor = image_tensor.to(dtype=torch.float32) / 255.0

        state = torch.zeros(self.state_dim, dtype=torch.float32)

        return {
            self.image_key: image_tensor,
            self.state_key: state,
            "task": task_description.lower(),
        }

    def _to_simpler_action(self, action_7d: np.ndarray):
        action_7d = np.asarray(action_7d, dtype=np.float64).reshape(-1)
        if action_7d.shape[0] != 7:
            raise ValueError(f"Expected 7D Bridge action, got shape {action_7d.shape}")

        raw_action = {
            "world_vector": action_7d[:3].astype(np.float32),
            "rotation_delta": action_7d[3:6].astype(np.float32),
            "gripper": np.array([action_7d[6]], dtype=np.float32),
        }

        roll, pitch, yaw = action_7d[3:6]
        axis, angle = euler2axangle(roll, pitch, yaw)
        rot_axangle = axis * angle

        action = {
            "world_vector": (action_7d[:3] * self.action_scale).astype(np.float32),
            "rot_axangle": (rot_axangle * self.action_scale).astype(np.float32),
            "gripper": np.array([1.0 if action_7d[6] > 0.0 else -1.0], dtype=np.float32),
            "terminate_episode": np.array([0.0], dtype=np.float32),
        }
        return raw_action, action
```

Why this adapter shape:

- LeRobot processors expect an unbatched raw observation. The saved
  `AddBatchDimensionProcessorStep` adds batch size 1.
- The image is CHW float in `[0, 1]`. Policy-specific image resizing happens
  inside the saved processor/model path.
- The zero state matches training.
- `policy.select_action` handles each policy's action queue.
- `postprocessor` unnormalizes the action back to the raw Bridge action scale.

## Modify `real2sim_eval_maniskill3.py`

Add args:

```python
    policy_path: str = ""
    """LeRobot pretrained_model path. If empty, ckpt_path is used."""

    policy_device: str = "cuda"
    """Torch device for LeRobot policy inference."""
```

Add the model branch next to `univla`:

```python
        elif args.model in {"lerobot", "pi05", "smolvla", "pi0_fast", "diffusion"}:
            from simpler_env.policies.lerobot_bridge.lerobot_bridge_model import LeRobotBridgeInference

            model = LeRobotBridgeInference(
                policy_path=args.policy_path or args.ckpt_path,
                policy_setup=policy_setup,
                action_scale=1,
                device=args.policy_device,
            )
```

No change is required in the rollout loop if `NUM_ENVS=1`, because the adapter
returns the same action dict as UniVLA.

If you want true `--num-envs > 1`, the eval loop must be changed to call the
policy for each environment image/instruction and stack the resulting actions.
That is a separate vectorization change.

## Modify `prepare_simplerenv_workspace.sh`

Copy the new adapter folder into external SimplerEnv:

```bash
require_dir "${ADAPTER_SRC}/policies/lerobot_bridge"

rm -rf "${SIMPLERENV_DIR}/simpler_env/policies/lerobot_bridge"
cp -a "${ADAPTER_SRC}/policies/lerobot_bridge" \
  "${SIMPLERENV_DIR}/simpler_env/policies/lerobot_bridge"
```

Keep the existing UniVLA copy logic. The two adapters can coexist:

```text
simpler_env/policies/univla/
simpler_env/policies/lerobot_bridge/
```

## Add A 4-Task Launcher

Add:

```text
experiments/robot/simpler-bridge/local_eval/run_lerobot_bridge_4task.sh
```

Recommended launcher:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/NHNHOME/WORKSPACE/0526040036_A/user/01/youngjoonjeong}"
REPO="${REPO:-${ROOT}/UniVLA-hyper}"
SIMPLERENV_DIR="${SIMPLERENV_DIR:-${ROOT}/third_party/SimplerEnv-maniskill3}"
PYTHON="${PYTHON:-python}"
GPU_ID="${GPU_ID:-0}"
POLICY_NAME="${POLICY_NAME:-lerobot}"
POLICY_PATH="${POLICY_PATH:?Set POLICY_PATH to a LeRobot pretrained_model directory.}"
NUM_EPISODES="${NUM_EPISODES:-24}"
NUM_ENVS="${NUM_ENVS:-1}"
SEED="${SEED:-0}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
SIM_BACKEND="${SIM_BACKEND:-physx_cpu}"
RENDER_BACKEND="${RENDER_BACKEND:-sapien_cpu}"
RECORD_DIR="${RECORD_DIR:-${ROOT}/eval_logs/simplerenv_lerobot_${POLICY_NAME}}"

if [[ ! -f "${SIMPLERENV_DIR}/real2sim_eval_maniskill3.py" ]]; then
  echo "[missing] ${SIMPLERENV_DIR}/real2sim_eval_maniskill3.py" >&2
  echo "Run prepare_simplerenv_workspace.sh first." >&2
  exit 1
fi

export PYTHONPATH="${REPO}:${SIMPLERENV_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/lvp_icd.json}"
export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/dev/null}"
export DISPLAY="${DISPLAY:-}"

TASKS=(
  "PutSpoonOnTableClothInScene-v1"
  "PutCarrotOnPlateInScene-v1"
  "StackGreenCubeOnYellowCubeBakedTexInScene-v1"
  "PutEggplantInBasketScene-v1"
)

SAVE_VIDEO_ARG="--save-video"
if [[ "${SAVE_VIDEO}" == "0" || "${SAVE_VIDEO}" == "false" || "${SAVE_VIDEO}" == "False" ]]; then
  SAVE_VIDEO_ARG="--no-save-video"
fi

cd "${SIMPLERENV_DIR}"
for task in "${TASKS[@]}"; do
  echo "[eval] ${POLICY_NAME} ${task}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" real2sim_eval_maniskill3.py \
    --model="${POLICY_NAME}" \
    --policy-path "${POLICY_PATH}" \
    --policy-device "cuda" \
    -e "${task}" \
    -s "${SEED}" \
    --num-episodes "${NUM_EPISODES}" \
    --num-envs "${NUM_ENVS}" \
    --sim-backend "${SIM_BACKEND}" \
    --render-backend "${RENDER_BACKEND}" \
    --record-dir "${RECORD_DIR}" \
    "${SAVE_VIDEO_ARG}"
done
```

Example:

```bash
ROOT=/NHNHOME/WORKSPACE/0526040036_A/user/01/youngjoonjeong
POLICY_NAME=pi05 \
POLICY_PATH="${ROOT}/outputs/lerobot_pi05_simpler_success50_nobase_gbs128_4gpu/checkpoints/last/pretrained_model" \
NUM_EPISODES=24 NUM_ENVS=1 SAVE_VIDEO=1 \
  ./experiments/robot/simpler-bridge/local_eval/run_lerobot_bridge_4task.sh
```

## Smoke Tests Before Full Eval

1. Confirm checkpoint directory:

```bash
ls "${POLICY_PATH}/config.json" "${POLICY_PATH}/model.safetensors"
ls "${POLICY_PATH}/preprocessor.json" "${POLICY_PATH}/postprocessor.json"
```

2. Confirm imports in the eval Python:

```bash
python - <<'PY'
import lerobot
import simpler_env
import torch
print("ok", torch.cuda.is_available())
PY
```

3. Run one episode with no video:

```bash
POLICY_NAME=pi05 POLICY_PATH="${PI05_POLICY_PATH}" \
NUM_EPISODES=1 NUM_ENVS=1 SAVE_VIDEO=0 \
  ./experiments/robot/simpler-bridge/local_eval/run_lerobot_bridge_4task.sh
```

4. Inspect the first action by temporarily printing in the adapter:

```python
print("raw lerobot action:", action_7d)
print("simpler action:", action)
```

Expected ranges:

```text
xyz roughly around [-0.03, 0.03]
roll/pitch/yaw roughly around [-0.2, 0.3]
gripper exactly -1 or 1 after thresholding
```

## Policy-Specific Notes

`pi05`

- Needs `task` and zero `observation.state`.
- The processor tokenizes `Task: ..., State: ...; Action:`.
- Use the saved postprocessor. Do not manually apply action stats.

`smolvla`

- Needs `task`, image, and state.
- The saved processor tokenizes with the SmolVLM tokenizer.
- It was trained with one image key. Missing extra cameras are not needed.

`pi0_fast`

- Needs `task`, image, and state.
- Requires FAST action tokenizer and PaliGemma tokenizer availability in the
  eval environment or HF cache.
- The default training wrapper was scratch unless a `POLICY_PATH` was supplied
  at train time, so expect it to need a real trained checkpoint before eval is
  meaningful.

`diffusion`

- Does not use language, but passing `task` is harmless.
- Needs image and zero state.
- It uses its own observation/action queues. Call `policy.reset()` on every
  environment reset through the adapter reset method.
- This policy is most likely to be sensitive to image resolution and GPU memory.

## Minimum Patch Summary

The minimum eval integration is:

```text
1. Add simpler_env/policies/lerobot_bridge/ adapter.
2. Add Args.policy_path and Args.policy_device to real2sim_eval_maniskill3.py.
3. Add an args.model branch for lerobot/pi05/smolvla/pi0_fast/diffusion.
4. Copy the adapter in prepare_simplerenv_workspace.sh.
5. Add run_lerobot_bridge_4task.sh.
6. Run with NUM_ENVS=1 first.
```

The important behavioral constraints are:

```text
Use checkpoint/pretrained_model as POLICY_PATH.
Use observation.images.primary.
Use zero observation.state.
Use saved LeRobot preprocessor and postprocessor.
Convert 7D [x, y, z, roll, pitch, yaw, gripper] to Simpler action dict.
Threshold gripper at 0 because the trained dataset used -1/1 gripper values.
```
