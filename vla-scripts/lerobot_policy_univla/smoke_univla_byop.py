#!/usr/bin/env python

from __future__ import annotations

from tempfile import TemporaryDirectory

import torch

import lerobot_policy_univla  # noqa: F401
from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.import_utils import register_third_party_plugins


def _stats(action_dim: int, state_dim: int) -> dict[str, dict[str, torch.Tensor]]:
    return {
        OBS_STATE: {
            "q01": torch.full((state_dim,), -1.0),
            "q99": torch.full((state_dim,), 1.0),
        },
        ACTION: {
            "q01": torch.full((action_dim,), -1.0),
            "q99": torch.full((action_dim,), 1.0),
        },
    }


def main() -> None:
    register_third_party_plugins()
    assert "univla" in PreTrainedConfig.get_known_choices()

    image_key = "observation.images.primary"
    state_dim = 7
    action_dim = 7
    window_size = 10
    config_cls = PreTrainedConfig.get_choice_class("univla")
    config = config_cls(
        input_features={
            image_key: PolicyFeature(FeatureType.VISUAL, (3, 224, 224)),
            OBS_STATE: PolicyFeature(FeatureType.STATE, (state_dim,)),
        },
        output_features={ACTION: PolicyFeature(FeatureType.ACTION, (action_dim,))},
        device="cpu",
        dummy=True,
        window_size=window_size,
        n_action_steps=window_size,
        image_key=image_key,
    )
    policy_cls = get_policy_class("univla")
    policy = policy_cls(config).eval()

    raw_observation = {
        image_key: torch.zeros(3, 224, 224),
        OBS_STATE: torch.zeros(state_dim),
        "task": "put the object into the container",
    }
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=_stats(action_dim, state_dim))
    observation = preprocessor(raw_observation)

    chunk = policy.predict_action_chunk(observation)
    assert tuple(chunk.shape) == (1, window_size, action_dim), tuple(chunk.shape)
    assert torch.isfinite(chunk).all()
    assert tuple(policy.select_action(observation).shape) == (1, action_dim)
    assert tuple(postprocessor(chunk[:, 0, :]).shape) == (1, action_dim)

    with TemporaryDirectory() as tmp:
        config.save_pretrained(tmp)
        preprocessor.save_pretrained(tmp)
        postprocessor.save_pretrained(tmp)
        loaded = policy_cls.from_pretrained(tmp)
        loaded_preprocessor, loaded_postprocessor = make_pre_post_processors(
            loaded.config,
            pretrained_path=tmp,
        )
        loaded_observation = loaded_preprocessor(raw_observation)
        loaded_chunk = loaded.predict_action_chunk(loaded_observation)
        assert tuple(loaded_chunk.shape) == (1, window_size, action_dim), tuple(loaded_chunk.shape)
        assert tuple(loaded_postprocessor(loaded_chunk[:, 0, :]).shape) == (1, action_dim)

    print("UniVLA LeRobot BYOP plugin smoke passed.")


if __name__ == "__main__":
    main()
