#!/usr/bin/env python

from __future__ import annotations

import json
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn

from lerobot.configs import FeatureType, PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION

from .configuration_univla import UniVLAConfig


def _torch_dtype(name: str) -> torch.dtype:
    table = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = str(name).lower()
    if key not in table:
        raise ValueError(f"Unsupported torch_dtype={name!r}. Use one of {sorted(table)}.")
    return table[key]


def _strip_prefix_if_present(state_dict: dict[str, Tensor], prefix: str) -> dict[str, Tensor]:
    if state_dict and all(key.startswith(prefix) for key in state_dict):
        return {key[len(prefix) :]: value for key, value in state_dict.items()}
    return state_dict


def _extract_state_dict(payload: Any) -> dict[str, Tensor]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "module"):
            if key in payload and isinstance(payload[key], dict):
                payload = payload[key]
                break
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported action decoder checkpoint payload type: {type(payload)}")
    state_dict = {str(key): value for key, value in payload.items()}
    for prefix in ("module.", "action_decoder.", "net."):
        state_dict = _strip_prefix_if_present(state_dict, prefix)
    return state_dict


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.scale, self.eps = dim**-0.5, eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class _SwishGLU(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.act, self.project = nn.SiLU(), nn.Linear(in_dim, 2 * out_dim)

    def forward(self, x: Tensor) -> Tensor:
        projected, gate = self.project(x).tensor_split(2, dim=-1)
        return projected * self.act(gate)


class _MAPAttention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int) -> None:
        super().__init__()
        if embed_dim % n_heads != 0:
            raise ValueError("embed_dim must be divisible by n_heads")
        self.n_heads, self.scale = n_heads, (embed_dim // n_heads) ** -0.5
        self.q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.kv = nn.Linear(embed_dim, 2 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, seed: Tensor, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        (batch_seed, num_seed, dim_seed), (batch_x, num_x, dim_x) = seed.shape, x.shape
        if dim_seed != dim_x:
            raise ValueError("Seed vectors and pool inputs must have the same embedding dimensionality.")
        q = self.q(seed).reshape(batch_seed, num_seed, self.n_heads, dim_seed // self.n_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(batch_x, num_x, 2, self.n_heads, dim_x // self.n_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        scores = q @ (k.transpose(-2, -1) * self.scale)
        if attention_mask is not None:
            attention_mask = attention_mask[None, None, :, :].repeat(1, self.n_heads, 1, 1)
            scores.masked_fill_(attention_mask == 0, float("-inf"))
        attn = scores.softmax(dim=-1)
        vals = (attn @ v).transpose(1, 2).reshape(batch_seed, num_seed, dim_seed)
        return self.proj(vals)


class _MAPBlock(nn.Module):
    def __init__(
        self,
        n_latents: int,
        vis_dim: int,
        embed_dim: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        do_rms_norm: bool = True,
        do_swish_glu: bool = True,
    ) -> None:
        super().__init__()
        self.n_latents, self.embed_dim, self.n_heads = n_latents, embed_dim, n_heads
        self.projection = nn.Linear(vis_dim, self.embed_dim)
        self.latents = nn.Parameter(torch.zeros(self.n_latents, self.embed_dim), requires_grad=True)
        nn.init.normal_(self.latents, std=0.02)
        self.attn_norm = _RMSNorm(self.embed_dim) if do_rms_norm else nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.attn = _MAPAttention(self.embed_dim, n_heads=self.n_heads)
        self.mlp_norm = _RMSNorm(self.embed_dim) if do_rms_norm else nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            (
                _SwishGLU(self.embed_dim, int(mlp_ratio * self.embed_dim))
                if do_swish_glu
                else nn.Sequential(nn.Linear(self.embed_dim, int(mlp_ratio * self.embed_dim)), nn.GELU())
            ),
            nn.Linear(int(mlp_ratio * self.embed_dim), self.embed_dim),
        )

    def forward(self, x: Tensor, mask: Tensor | None = None, init_embed: Tensor | None = None) -> Tensor:
        latents = self.latents.unsqueeze(0).expand(x.shape[0], -1, -1)
        latents = latents + init_embed.unsqueeze(1) if init_embed is not None else latents
        latents = self.attn_norm(latents + self.attn(latents, self.projection(x), mask))
        latents = self.mlp_norm(latents + self.mlp(latents))
        return latents.squeeze(dim=1)


def _make_action_decoder_cls(map_block_cls: type[nn.Module] = _MAPBlock):
    class UniVLAActionDecoder(nn.Module):
        def __init__(
            self,
            *,
            window_size: int,
            action_dim: int,
            hidden_dim: int,
            latent_action_token_len: int,
            proprio_dim: int,
            use_proprio: bool,
            output_tanh: bool,
        ) -> None:
            super().__init__()
            self.window_size = int(window_size)
            self.action_dim = int(action_dim)
            self.latent_action_token_len = int(latent_action_token_len)
            self.proprio_dim = int(proprio_dim)
            self.use_proprio = bool(use_proprio)
            self.latent_action_pool = map_block_cls(
                n_latents=1,
                vis_dim=4096,
                embed_dim=hidden_dim,
                n_heads=hidden_dim // 64,
            )
            self.visual_pool = map_block_cls(
                n_latents=1,
                vis_dim=4096,
                embed_dim=hidden_dim,
                n_heads=hidden_dim // 64,
            )
            if self.use_proprio:
                self.proprio_proj = nn.Sequential(
                    nn.Linear(proprio_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                proj_in_dim = hidden_dim * 2
            else:
                proj_in_dim = hidden_dim

            layers: list[nn.Module] = [nn.Linear(proj_in_dim, action_dim * window_size)]
            if output_tanh:
                layers.append(nn.Tanh())
            self.proj = nn.Sequential(*layers)

        def forward(self, latent_action_tokens: Tensor, visual_embed: Tensor, proprio: Tensor | None = None) -> Tensor:
            visual_embed = self.visual_pool(visual_embed.to(torch.float))
            if latent_action_tokens.shape[1] < self.latent_action_token_len:
                raise ValueError(
                    "Not enough latent action tokens: "
                    f"expected at least {self.latent_action_token_len}, got {latent_action_tokens.shape[1]}."
                )
            latent_action_tokens = latent_action_tokens[:, -self.latent_action_token_len :].to(torch.float)
            action_token = self.latent_action_pool(latent_action_tokens, init_embed=visual_embed)

            if self.use_proprio:
                if proprio is None:
                    proprio = torch.zeros(
                        latent_action_tokens.shape[0],
                        self.proprio_dim,
                        device=latent_action_tokens.device,
                        dtype=torch.float32,
                    )
                proprio_embed = self.proprio_proj(proprio.to(device=latent_action_tokens.device, dtype=torch.float32))
                action_token = torch.cat([action_token, proprio_embed], dim=-1)

            return self.proj(action_token)

    return UniVLAActionDecoder


class UniVLAPolicy(PreTrainedPolicy):
    config_class = UniVLAConfig
    name = "univla"

    def __init__(self, config: UniVLAConfig, dataset_stats: dict[str, Any] | None = None):
        del dataset_stats
        super().__init__(config)
        config.validate_features()
        self._queued_actions: deque[Tensor] = deque()
        self._prev_hist_action: list[str] = [""]
        self.dataset_statistics: dict[str, Any] = {}
        self._dataset_name: str | None = None
        self._proprio_mean: Tensor | None = None
        self._proprio_std: Tensor | None = None
        self._hf_dtype = _torch_dtype(self.config.torch_dtype)
        self._action_decoder_uses_proprio = False
        self.vla = None
        self.processor = None
        self.action_decoder = None
        self.register_buffer("_device_anchor", torch.zeros(1), persistent=False)

        if not self.config.dummy:
            self._load_real_components()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        **kwargs: Any,
    ) -> "UniVLAPolicy":
        del strict
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )
        if not isinstance(config, UniVLAConfig):
            raise TypeError(f"Expected UniVLAConfig, got {type(config)}")

        policy = cls(config)
        policy.to(config.device)
        policy.eval()
        return policy

    def _load_real_components(self) -> None:
        missing = []
        for field_name in ("vla_path", "action_decoder_path"):
            value = getattr(self.config, field_name)
            if not value:
                missing.append(field_name)
        if missing:
            raise ValueError(
                "UniVLAConfig(dummy=False) requires exported checkpoint fields: "
                + ", ".join(missing)
                + ". Use dummy=True for policy_server plumbing smoke tests."
            )
        AutoModelForVision2Seq, AutoProcessor = self._register_hf_classes()
        device = torch.device(self.config.device)

        self.vla = AutoModelForVision2Seq.from_pretrained(
            self.config.vla_path,
            attn_implementation=self.config.attn_implementation,
            torch_dtype=self._hf_dtype,
            load_in_8bit=self.config.load_in_8bit,
            load_in_4bit=self.config.load_in_4bit,
            low_cpu_mem_usage=self.config.low_cpu_mem_usage,
            trust_remote_code=self.config.trust_remote_code,
        )
        self.vla.config.latent_action_token_len = self.config.latent_action_token_len
        self.vla.to(device)
        self.vla.eval()

        self.processor = AutoProcessor.from_pretrained(
            self.config.vla_path,
            trust_remote_code=self.config.trust_remote_code,
        )

        self.dataset_statistics = self._load_dataset_statistics()
        if self.dataset_statistics:
            self.vla.norm_stats = self.dataset_statistics
        self._configure_proprio_stats(device)
        self.action_decoder = self._load_action_decoder(device)
        self.action_decoder.eval()

    def _ensure_univla_import_path(self) -> None:
        candidates = []
        repo_root = getattr(self.config, "univla_repo_root", None)
        if repo_root:
            candidates.append(Path(repo_root))
        env_root = os.environ.get("UNIVLA_REPO_ROOT")
        if env_root:
            candidates.append(Path(env_root))
        candidates.append(Path(__file__).resolve().parents[4])

        for candidate in candidates:
            if (candidate / "prismatic").is_dir() and str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
                return

    def _register_hf_classes(self) -> tuple[type[Any], type[Any]]:
        self._ensure_univla_import_path()
        from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
        from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

        for register in (
            lambda: AutoConfig.register("openvla", OpenVLAConfig),
            lambda: AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor),
            lambda: AutoProcessor.register(OpenVLAConfig, PrismaticProcessor),
            lambda: AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction),
        ):
            try:
                register()
            except ValueError:
                pass
        return AutoModelForVision2Seq, AutoProcessor

    def _load_dataset_statistics(self) -> dict[str, Any]:
        stats_path = self.config.dataset_statistics_path
        if not stats_path and self.config.vla_path:
            candidate = Path(self.config.vla_path) / "dataset_statistics.json"
            if candidate.exists():
                stats_path = str(candidate)
        if not stats_path:
            return {}
        with open(stats_path, "r") as f:
            return json.load(f)

    def _resolve_dataset_name(self) -> str | None:
        if self.config.dataset_name:
            return self.config.dataset_name
        if len(self.dataset_statistics) == 1:
            return next(iter(self.dataset_statistics))
        return self.config.unnorm_key

    def _configure_proprio_stats(self, device: torch.device) -> None:
        self._dataset_name = self._resolve_dataset_name()
        if not self._dataset_name or self._dataset_name not in self.dataset_statistics:
            return
        proprio_stats = self.dataset_statistics[self._dataset_name].get("proprio")
        if not proprio_stats:
            return
        mean = torch.tensor(proprio_stats["mean"], dtype=torch.float32, device=device)
        std = torch.tensor(proprio_stats["std"], dtype=torch.float32, device=device)
        self._proprio_mean = mean
        self._proprio_std = torch.clamp(std, min=1e-2)

    def _load_action_decoder(self, device: torch.device) -> nn.Module:
        payload = torch.load(self.config.action_decoder_path, map_location="cpu")
        state_dict = _extract_state_dict(payload)
        uses_proprio = any(key.startswith("proprio_proj.") for key in state_dict)
        proj_weight = state_dict.get("proj.0.weight")
        if proj_weight is None:
            raise KeyError(
                "Could not find proj.0.weight in action decoder checkpoint. "
                f"First keys: {list(state_dict)[:8]}"
            )
        expected_out = self.config.action_dim * self.config.window_size
        if int(proj_weight.shape[0]) != expected_out:
            raise ValueError(
                "Action decoder output shape does not match config: "
                f"checkpoint proj.0.weight[0]={proj_weight.shape[0]}, expected {expected_out} "
                f"(action_dim={self.config.action_dim}, window_size={self.config.window_size})."
            )
        hidden_dim = int(proj_weight.shape[1] // 2) if uses_proprio else int(proj_weight.shape[1])
        decoder_cls = _make_action_decoder_cls()
        decoder = decoder_cls(
            window_size=self.config.window_size,
            action_dim=self.config.action_dim,
            hidden_dim=hidden_dim,
            latent_action_token_len=self.config.latent_action_token_len,
            proprio_dim=self.config.state_dim,
            use_proprio=uses_proprio,
            output_tanh=self.config.decoder_output_tanh,
        )
        decoder.load_state_dict(state_dict, strict=True)
        decoder.to(device)
        self._action_decoder_uses_proprio = uses_proprio
        return decoder

    def get_optim_params(self) -> dict[str, list[nn.Parameter]]:
        return {"params": list(self.parameters())}

    def reset(self) -> None:
        self._queued_actions.clear()
        self._prev_hist_action = [""]

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        chunk = self.predict_action_chunk(batch)
        loss = chunk.sum() * 0.0
        return loss, {"loss": float(loss.detach().cpu())}

    def _batch_size_and_device(self, batch: dict[str, Any]) -> tuple[int, torch.device, torch.dtype]:
        candidates = []
        for key in [self.config.state_key, self.config.image_key, *self.config.fallback_image_keys]:
            value = batch.get(key)
            if isinstance(value, Tensor):
                candidates.append(value)
        for key, value in batch.items():
            if key.startswith("observation.") and isinstance(value, Tensor):
                candidates.append(value)

        if not candidates:
            return 1, self._device_anchor.device, torch.float32

        tensor = candidates[0]
        batch_size = tensor.shape[0] if tensor.ndim > 1 else 1
        return batch_size, tensor.device, tensor.dtype if tensor.is_floating_point() else torch.float32

    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Any) -> Tensor:
        del kwargs
        if not self.config.dummy:
            return self._predict_real_action_chunk(batch)

        batch_size, device, dtype = self._batch_size_and_device(batch)
        return torch.zeros(
            batch_size,
            self.config.window_size,
            self.config.action_dim,
            dtype=dtype,
            device=device,
        )

    def _predict_real_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        if self.vla is None or self.processor is None or self.action_decoder is None:
            raise RuntimeError("UniVLA real components are not loaded.")

        image_tensor = self._get_image_tensor(batch)
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        if image_tensor.ndim != 4:
            raise ValueError(f"Expected image tensor shape (B,C,H,W) or (C,H,W), got {tuple(image_tensor.shape)}")

        chunks = []
        for index in range(int(image_tensor.shape[0])):
            task = self._get_task(batch, index)
            image = self._tensor_to_pil(image_tensor[index])
            prompt = self._build_prompt(task)
            inputs = self.processor(prompt, image).to(self._device_anchor.device, dtype=self._hf_dtype)
            with torch.no_grad():
                latent_action, visual_embed, generated_ids = self.vla.predict_latent_action(
                    **inputs,
                    unnorm_key=self.config.unnorm_key,
                    do_sample=self.config.do_sample,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                )
                proprio = self._get_proprio(batch, index, latent_action.device)
                action = self.action_decoder(latent_action, visual_embed, proprio)
                chunk = action.reshape(1, self.config.window_size, self.config.action_dim)
            self._remember_generated_actions(generated_ids)
            chunks.append(chunk)
        return torch.cat(chunks, dim=0)

    def _get_image_tensor(self, batch: dict[str, Any]) -> Tensor:
        for key in [self.config.image_key, *self.config.fallback_image_keys]:
            value = batch.get(key)
            if isinstance(value, Tensor):
                return value
        available = [key for key, value in batch.items() if key.startswith("observation.") and isinstance(value, Tensor)]
        raise KeyError(
            f"Could not find UniVLA image key {self.config.image_key!r}. "
            f"Fallback keys={self.config.fallback_image_keys}; available tensor observation keys={available}."
        )

    def _get_task(self, batch: dict[str, Any], index: int) -> str:
        value = batch.get(self.config.task_key, batch.get("task"))
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            return str(value[index])
        if value is None:
            raise KeyError(f"Could not find language task key {self.config.task_key!r} in LeRobot batch.")
        return str(value)

    def _get_proprio(self, batch: dict[str, Any], index: int, device: torch.device) -> Tensor | None:
        if not self.config.use_proprio or not self._action_decoder_uses_proprio:
            return None
        value = batch.get(self.config.state_key)
        if value is None:
            return torch.zeros(1, self.config.state_dim, device=device, dtype=torch.float32)
        if not isinstance(value, Tensor):
            value = torch.tensor(value, dtype=torch.float32)
        if value.ndim == 1:
            proprio = value.unsqueeze(0)
        else:
            proprio = value[index : index + 1]
        proprio = proprio.to(device=device, dtype=torch.float32)
        if self._proprio_mean is not None and self._proprio_std is not None:
            proprio = (proprio - self._proprio_mean.to(device)) / self._proprio_std.to(device)
        return proprio

    def _build_prompt(self, task: str) -> str:
        task = str(task).lower()
        if self.config.use_history_action and self._prev_hist_action and self._prev_hist_action[-1]:
            return f"In: What action should the robot take to {task}? History action {self._prev_hist_action[-1]}\nOut:"
        return f"In: What action should the robot take to {task}?\nOut:"

    @staticmethod
    def _tensor_to_pil(image_tensor: Tensor) -> Image.Image:
        tensor = image_tensor.detach().cpu()
        if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
            tensor = tensor.permute(1, 2, 0)
        array = tensor.float().numpy()
        if array.max(initial=0.0) <= 1.5:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
        if array.ndim == 3 and array.shape[2] == 1:
            array = np.repeat(array, 3, axis=2)
        return Image.fromarray(array).convert("RGB")

    def _remember_generated_actions(self, generated_ids: Tensor) -> None:
        if not self.config.use_history_action:
            return
        pieces = []
        for token_id in generated_ids[0].detach().cpu().tolist():
            action_idx = int(token_id) - 32001
            if 0 <= action_idx < self.config.action_vocab_size:
                pieces.append(f"<ACT_{action_idx}>")
        self._prev_hist_action.append("".join(pieces))

    def select_action(self, batch: dict[str, Tensor], **kwargs: Any) -> Tensor:
        if not self._queued_actions:
            chunk = self.predict_action_chunk(batch, **kwargs)
            if chunk.ndim != 3:
                raise ValueError(f"Expected action chunk shape (B,T,D), got {tuple(chunk.shape)}")
            if chunk.shape[0] != 1:
                raise NotImplementedError("select_action currently supports batch size 1.")
            for action in chunk[0, : self.config.n_action_steps]:
                self._queued_actions.append(action)

        return self._queued_actions.popleft().unsqueeze(0)

    @property
    def action_feature(self):
        return self.config.output_features[ACTION]

    @property
    def image_features(self):
        return {
            key: feature
            for key, feature in self.config.input_features.items()
            if feature.type is FeatureType.VISUAL
        }
