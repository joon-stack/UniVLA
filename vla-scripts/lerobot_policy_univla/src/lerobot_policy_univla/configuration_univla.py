#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
except ImportError:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature

try:
    from lerobot.optim import AdamWConfig
except ImportError:
    from lerobot.optim.optimizers import AdamWConfig
from lerobot.utils.constants import ACTION, OBS_STATE


def _coerce_feature(feature: PolicyFeature | dict[str, Any]) -> PolicyFeature:
    if isinstance(feature, PolicyFeature):
        return feature
    if isinstance(feature, dict):
        return PolicyFeature(type=FeatureType(feature["type"]), shape=tuple(feature["shape"]))
    raise TypeError(f"Expected PolicyFeature or dict, got {type(feature)}")


def _coerce_feature_map(features: dict[str, PolicyFeature] | None) -> dict[str, PolicyFeature]:
    if not features:
        return {}
    return {key: _coerce_feature(feature) for key, feature in features.items()}


@PreTrainedConfig.register_subclass("univla")
@dataclass
class UniVLAConfig(PreTrainedConfig):
    """LeRobot BYOP config for UniVLA runtime checkpoints.

    `dummy=True` is only for policy_server/robot_client plumbing smoke tests.
    Real UniVLA inference will use `dummy=False` after the checkpoint loader is wired.
    """

    n_obs_steps: int = 1
    window_size: int = 10
    n_action_steps: int = 10
    action_dim: int = 7
    state_dim: int = 7
    image_resolution: tuple[int, int] = (224, 224)

    image_key: str = "observation.images.primary"
    wrist_image_key: str = "observation.images.wrist"
    state_key: str = OBS_STATE
    task_key: str = "task"
    fallback_image_keys: list[str] = field(
        default_factory=lambda: ["observation.images.top", "observation.image", "observation.images.image"]
    )

    dummy: bool = True
    vla_path: str | None = None
    action_decoder_path: str | None = None
    dataset_statistics_path: str | None = None
    univla_repo_root: str | None = None
    dataset_name: str | None = None
    unnorm_key: str | None = None
    latent_action_token_len: int = 4
    do_sample: bool = True
    temperature: float = 0.75
    top_p: float = 0.9
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    trust_remote_code: bool = True
    low_cpu_mem_usage: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_proprio: bool = True
    wrist_fusion: str = "none"
    decoder_output_tanh: bool = True
    use_history_action: bool = True
    action_vocab_size: int = 32
    action_token_mask: str = "none"
    action_token_id_offset: int = 32001
    radius_action_vocab_size: int = 16
    direction_action_vocab_size: int = 16

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.QUANTILES,
        }
    )

    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_obs_steps != 1:
            raise ValueError(f"UniVLA wrapper only supports n_obs_steps=1 for now, got {self.n_obs_steps}.")
        if self.n_action_steps > self.window_size:
            raise ValueError(
                f"n_action_steps must be <= window_size, got {self.n_action_steps} > {self.window_size}."
            )
        self.wrist_fusion = self._normalize_wrist_fusion(self.wrist_fusion)
        self.validate_features()

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        self.input_features = _coerce_feature_map(self.input_features)
        self.output_features = _coerce_feature_map(self.output_features)

        if self.output_features and ACTION in self.output_features:
            self.action_dim = self.output_features[ACTION].shape[-1]
        else:
            self.output_features[ACTION] = PolicyFeature(FeatureType.ACTION, (self.action_dim,))

        visual_keys = [key for key, feature in self.input_features.items() if feature.type is FeatureType.VISUAL]
        primary_visual_keys = [key for key in visual_keys if key != self.wrist_image_key]
        if primary_visual_keys and self.image_key not in self.input_features:
            self.image_key = primary_visual_keys[0]
        elif not primary_visual_keys and self.image_key not in self.input_features:
            channels = 3
            height, width = self.image_resolution
            self.input_features[self.image_key] = PolicyFeature(FeatureType.VISUAL, (channels, height, width))

        if self.wrist_fusion_enabled() and self.wrist_image_key not in self.input_features:
            primary_feature = self.input_features[self.image_key]
            self.input_features[self.wrist_image_key] = PolicyFeature(FeatureType.VISUAL, primary_feature.shape)

        if self.state_key in self.input_features:
            self.state_dim = self.input_features[self.state_key].shape[-1]
        elif self.state_dim > 0:
            self.input_features[self.state_key] = PolicyFeature(FeatureType.STATE, (self.state_dim,))

    @staticmethod
    def _normalize_wrist_fusion(value: str | bool | None) -> str:
        if isinstance(value, bool):
            return "decoder_residual" if value else "none"
        mode = str(value or "none").strip().lower()
        aliases = {
            "0": "none",
            "false": "none",
            "off": "none",
            "no": "none",
            "1": "decoder_residual",
            "true": "decoder_residual",
            "on": "decoder_residual",
            "yes": "decoder_residual",
            "late": "decoder_residual",
            "decoder_late": "decoder_residual",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"none", "decoder_residual"}:
            raise ValueError(f"Unsupported wrist_fusion={value!r}. Use 'none' or 'decoder_residual'.")
        return mode

    def wrist_fusion_enabled(self) -> bool:
        return self._normalize_wrist_fusion(self.wrist_fusion) != "none"

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.window_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
