import os
import time
from typing import Dict

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torchvision import transforms

from latent_action_model.genie.modules.blocks import (
    SpatioTemporalTransformer,
    SpatioTransformer,
    VectorQuantizer,
)


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class VisualVQDINOLatentActionModel(nn.Module):
    """
    Visual-only latent action VQ-VAE operating in DINO patch-token space.
    """

    def __init__(
        self,
        in_dim: int,
        model_dim: int,
        latent_dim: int,
        num_latents: int,
        patch_size: int,
        enc_blocks: int,
        dec_blocks: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.patch_size = patch_size

        self.dino_transform = transforms.Normalize(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )
        self.dino_encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.dino_encoder.requires_grad_(False)

        dino_dim = 768
        self.num_codes = 4
        self.action_latent = nn.Parameter(torch.empty(1, 1, self.num_codes, dino_dim))
        nn.init.uniform_(self.action_latent, a=-1, b=1)

        self.encoder = SpatioTemporalTransformer(
            in_dim=dino_dim,
            model_dim=model_dim,
            out_dim=latent_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout,
            causal_temporal=True,
            to_out=False,
        )
        self.to_codebook = nn.Linear(model_dim, latent_dim)
        self.vq = VectorQuantizer(
            num_latents=num_latents,
            latent_dim=latent_dim,
            code_restart=True,
        )

        self.patch_up = nn.Linear(dino_dim, model_dim)
        self.action_up = nn.Linear(latent_dim, model_dim)
        self.decoder = SpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=dino_dim,
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )

        self._module_profile_steps = int(os.environ.get("UNIVLA_LAM_MODULE_PROFILE_STEPS", "0"))
        self._module_profile_count = 0

    def _extract_dino_features(self, videos: Tensor) -> Tensor:
        bsz, num_frames = videos.shape[:2]
        videos = rearrange(videos, "b t c h w -> (b t) c h w")
        videos = self.dino_transform(videos)
        dino_features = self.dino_encoder.forward_features(videos)["x_norm_patchtokens"]
        return rearrange(dino_features, "(b t) l d -> b t l d", b=bsz, t=num_frames)

    def _encode_transition_features(self, current_features: Tensor, target_features: Tensor) -> Tensor:
        bsz = current_features.shape[0]
        features = torch.stack([current_features, target_features], dim=1)
        action_pad = self.action_latent.expand(bsz, 2, -1, -1)
        padded_patches = torch.cat([action_pad, features], dim=2)
        z = self.encoder(padded_patches)
        return self.to_codebook(z[:, 1, :self.num_codes])

    def encode_transition_tokens(self, current_images: Tensor, target_images: Tensor) -> Tensor:
        videos = torch.stack([current_images, target_images], dim=1)
        features = self._extract_dino_features(videos)
        return self._encode_transition_features(features[:, 0], features[:, 1])

    def vq_encode(self, videos: Tensor) -> Dict:
        profile = (
            self._module_profile_steps
            and self._module_profile_count < self._module_profile_steps
            and os.environ.get("LOCAL_RANK", "0") == "0"
        )
        times = {}

        def mark(name: str, start: float) -> float:
            if profile and torch.cuda.is_available():
                torch.cuda.synchronize()
            now = time.perf_counter()
            if profile:
                times[name] = now - start
            return now

        start = time.perf_counter()
        bsz, num_frames = videos.shape[:2]
        dino_features = self._extract_dino_features(videos)
        start = mark("dino", start)

        action_pad = self.action_latent.expand(bsz, num_frames, -1, -1)
        padded_patches = torch.cat([action_pad, dino_features], dim=2)

        z = self.encoder(padded_patches)
        start = mark("encoder", start)

        z = self.to_codebook(z[:, 1:, :self.num_codes])
        z = z.reshape(bsz * (num_frames - 1), self.num_codes, self.latent_dim)
        z_q, z, emb, indices = self.vq(z)
        z_q = z_q.reshape(bsz, num_frames - 1, self.num_codes, self.latent_dim)
        mark("vq", start)

        return {
            "patches": dino_features,
            "z_q": z_q,
            "z": z,
            "emb": emb,
            "indices": indices,
            "_profile_times": times,
        }

    def forward(self, batch: Dict) -> Dict:
        profile = (
            self._module_profile_steps
            and self._module_profile_count < self._module_profile_steps
            and os.environ.get("LOCAL_RANK", "0") == "0"
        )

        outputs = self.vq_encode(batch["videos"])
        if "radprog_mid_pixel_values" in batch:
            current_features = outputs["patches"][:, 0]
            mid_features = self._extract_dino_features(batch["radprog_mid_pixel_values"].unsqueeze(1))[:, 0]

            outputs["radprog_z_self"] = self._encode_transition_features(current_features, current_features)
            outputs["radprog_z_mid"] = self._encode_transition_features(current_features, mid_features)
            if outputs["emb"].shape[0] == batch["videos"].shape[0]:
                outputs["radprog_z_future"] = outputs["emb"].reshape(
                    batch["videos"].shape[0],
                    self.num_codes,
                    self.latent_dim,
                )
            else:
                outputs["radprog_z_future"] = self._encode_transition_features(
                    current_features,
                    outputs["patches"][:, -1],
                )
        if profile and torch.cuda.is_available():
            torch.cuda.synchronize()
        decoder_start = time.perf_counter()

        video_patches = self.patch_up(outputs["patches"][:, :-1])
        action_patches = self.action_up(outputs["z_q"])
        video_action_patches = torch.cat([action_patches, video_patches], dim=2)
        video_recon = self.decoder(video_action_patches)
        video_recon = video_recon[:, :, self.num_codes : self.num_codes + video_patches.shape[2]]

        if profile:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times = outputs.pop("_profile_times")
            times["decoder"] = time.perf_counter() - decoder_start
            print(
                "LAM_MODULE_PROFILE "
                f"step={self._module_profile_count + 1:06d} "
                f"dino={times.get('dino', 0.0):.3f}s "
                f"encoder={times.get('encoder', 0.0):.3f}s "
                f"vq={times.get('vq', 0.0):.3f}s "
                f"decoder={times.get('decoder', 0.0):.3f}s",
                flush=True,
            )
        else:
            outputs.pop("_profile_times", None)
        self._module_profile_count += 1

        outputs.update(
            {
                "recon": video_recon,
                "target": outputs["patches"][:, 1:],
            }
        )
        return outputs

    @property
    def device(self):
        return next(self.parameters()).device
