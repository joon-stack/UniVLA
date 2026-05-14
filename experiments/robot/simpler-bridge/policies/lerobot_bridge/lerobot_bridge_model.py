from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


ACTION_DIM = 7


class TemporalActionEnsembler:
    """Online temporal ensemble over newly predicted action chunks."""

    def __init__(self, coeff: float, chunk_size: int) -> None:
        self.coeff = float(coeff)
        self.chunk_size = int(chunk_size)
        self.ensembled_actions: torch.Tensor | None = None
        self.ensembled_actions_count: torch.Tensor | None = None

    def reset(self) -> None:
        self.ensembled_actions = None
        self.ensembled_actions_count = None

    def update(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim != 3:
            raise ValueError(f"Expected action chunk shape (B, T, A), got {tuple(actions.shape)}")
        if actions.shape[1] != self.chunk_size:
            raise ValueError(f"Expected chunk_size={self.chunk_size}, got {actions.shape[1]}")

        weights = torch.exp(
            -self.coeff * torch.arange(self.chunk_size, device=actions.device, dtype=actions.dtype)
        )
        weights_cumsum = torch.cumsum(weights, dim=0)

        if self.ensembled_actions is None:
            self.ensembled_actions = actions.clone()
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=actions.device
            )
        else:
            assert self.ensembled_actions_count is not None
            self.ensembled_actions *= weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += actions[:, :-1] * weights[self.ensembled_actions_count]
            self.ensembled_actions /= weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
            self.ensembled_actions_count = torch.cat(
                [self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])]
            )

        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action


class LeRobotBridgeInference:
    """Run a trained LeRobot policy with the SimplerEnv Bridge action contract."""

    def __init__(
        self,
        policy_path: str | Path,
        policy_setup: str = "widowx_bridge",
        action_scale: float = 1.0,
        device: str | None = None,
        debug_actions: int = 0,
        action_execution: str = "queue",
        temporal_agg_coeff: float = -0.1,
    ) -> None:
        if not policy_path:
            raise ValueError("Set --policy-path or --ckpt-path to a LeRobot pretrained_model directory/HF repo.")
        if policy_setup != "widowx_bridge":
            raise ValueError(f"Unsupported policy_setup for LeRobotBridgeInference: {policy_setup}")

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        self.policy_path = str(policy_path)
        self.policy_setup = policy_setup
        self.action_scale = float(action_scale)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.debug_actions = int(debug_actions)
        self.action_execution = action_execution
        self.temporal_agg_coeff = float(temporal_agg_coeff)
        self.temporal_ensembler: TemporalActionEnsembler | None = None
        self.task_description: str | None = None
        self.step_count = 0

        if self.action_execution not in {"queue", "temporal_agg"}:
            raise ValueError(
                f"Unsupported action_execution={self.action_execution!r}; expected 'queue' or 'temporal_agg'."
            )

        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        cfg = PreTrainedConfig.from_pretrained(self.policy_path)
        cfg.device = self.device
        if hasattr(cfg, "compile_model"):
            cfg.compile_model = False
        if hasattr(cfg, "gradient_checkpointing"):
            cfg.gradient_checkpointing = False

        self.config = cfg
        self.preprocessor, self.postprocessor = make_pre_post_processors(cfg, pretrained_path=self.policy_path)

        policy_cls = get_policy_class(cfg.type)
        self.policy = policy_cls.from_pretrained(self.policy_path, config=cfg, strict=True)
        self.policy.to(self.device)
        self.policy.eval()
        self.reset()

        print(
            "[lerobot_bridge] loaded "
            f"type={cfg.type} path={self.policy_path} device={self.device} "
            f"chunk_size={getattr(cfg, 'chunk_size', 'n/a')} "
            f"n_action_steps={getattr(cfg, 'n_action_steps', 'n/a')} "
            f"action_execution={self.action_execution} "
            f"temporal_agg_coeff={self.temporal_agg_coeff}"
        )

    def reset(self, task_description: Any = None) -> None:
        self.task_description = self._normalize_task(task_description)
        self.step_count = 0
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        if self.temporal_ensembler is not None:
            self.temporal_ensembler.reset()

    def step(self, image: np.ndarray, task_description: Any = None) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        task = self._normalize_task(task_description)
        if task != self.task_description:
            self.reset(task)

        batch = {
            "observation.images.primary": self._image_to_tensor(image),
            "observation.state": torch.zeros(ACTION_DIM, dtype=torch.float32),
            "task": task,
        }

        with torch.inference_mode():
            processed = self.preprocessor(batch)
            if self.action_execution == "temporal_agg":
                raw_action = self._predict_temporal_agg_action(processed)
            else:
                raw_action = self.policy.select_action(processed)
            post_action = self.postprocessor(raw_action)

        action_7d = self._to_action_vector(post_action, name="postprocessed action")
        raw_vector = self._try_action_vector(raw_action)

        raw_action_dict = {
            "world_vector": action_7d[:3].copy(),
            "rot_axangle": action_7d[3:6].copy(),
            "gripper": action_7d[6:7].copy(),
        }
        action = {
            "world_vector": (action_7d[:3] * self.action_scale).astype(np.float32),
            "rot_axangle": (action_7d[3:6] * self.action_scale).astype(np.float32),
            "gripper": np.array([1.0 if action_7d[6] > 0.0 else -1.0], dtype=np.float32),
            "terminate_episode": np.array([0.0], dtype=np.float32),
        }

        if self.step_count < self.debug_actions:
            raw_msg = "n/a" if raw_vector is None else np.array2string(raw_vector, precision=4)
            print(
                f"[lerobot_bridge] step={self.step_count} task={task!r} "
                f"raw={raw_msg} post={np.array2string(action_7d, precision=4)} "
                f"env_gripper={float(action['gripper'][0]):.1f}"
            )
        self.step_count += 1
        return raw_action_dict, action

    def _predict_temporal_agg_action(self, batch: dict[str, Any]) -> torch.Tensor:
        if not hasattr(self.policy, "predict_action_chunk"):
            raise ValueError(f"Policy type {getattr(self.config, 'type', 'unknown')} does not support action chunks.")

        if getattr(self.config, "type", None) == "diffusion":
            from lerobot.policies.utils import populate_queues
            from lerobot.utils.constants import ACTION, OBS_IMAGES

            # DiffusionPolicy.predict_action_chunk expects its observation queues
            # to already be populated, unlike pi0.5/SmolVLA chunk predictors.
            batch = dict(batch)
            batch.pop(ACTION, None)
            if getattr(self.config, "image_features", None):
                batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
            self.policy._queues = populate_queues(self.policy._queues, batch)

        raw_chunk = self.policy.predict_action_chunk(batch)
        if raw_chunk.ndim != 3:
            raise ValueError(f"Expected raw action chunk shape (B, T, A), got {tuple(raw_chunk.shape)}")

        horizon = min(int(getattr(self.config, "n_action_steps", raw_chunk.shape[1])), raw_chunk.shape[1])
        raw_chunk = raw_chunk[:, :horizon]
        if self.temporal_ensembler is None or self.temporal_ensembler.chunk_size != horizon:
            self.temporal_ensembler = TemporalActionEnsembler(self.temporal_agg_coeff, horizon)
        return self.temporal_ensembler.update(raw_chunk)

    @staticmethod
    def _normalize_task(task_description: Any) -> str:
        if task_description is None:
            return ""
        if isinstance(task_description, str):
            return task_description.lower()
        if isinstance(task_description, (list, tuple)) and task_description:
            return str(task_description[0]).lower()
        if isinstance(task_description, np.ndarray) and task_description.size:
            return str(task_description.reshape(-1)[0]).lower()
        return str(task_description).lower()

    @staticmethod
    def _image_to_tensor(image: np.ndarray) -> torch.Tensor:
        image = np.asarray(image)
        if image.ndim == 4:
            if image.shape[0] != 1:
                raise ValueError(f"Expected one image for non-vectorized eval, got shape {image.shape}")
            image = image[0]
        if image.ndim != 3:
            raise ValueError(f"Expected image shape HWC or CHW, got {image.shape}")

        tensor = torch.as_tensor(image)
        if tensor.shape[0] not in (1, 3):
            tensor = tensor.permute(2, 0, 1)
        tensor = tensor.to(dtype=torch.float32)
        if float(tensor.max()) > 1.0:
            tensor = tensor / 255.0
        return tensor.contiguous()

    @staticmethod
    def _to_action_vector(action: Any, name: str) -> np.ndarray:
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().float().numpy()
        action = np.asarray(action, dtype=np.float32)
        if action.ndim == 0:
            raise ValueError(f"{name} is scalar, expected 7D action")
        action = action.reshape(-1, ACTION_DIM)[0]
        if action.shape != (ACTION_DIM,):
            raise ValueError(f"{name} has shape {action.shape}, expected ({ACTION_DIM},)")
        return action

    @classmethod
    def _try_action_vector(cls, action: Any) -> np.ndarray | None:
        try:
            return cls._to_action_vector(action, name="raw action")
        except Exception:
            return None
