import os
import time
from os import listdir, makedirs
from typing import Callable, Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import wandb
from accelerate import PartialState
from lightning import LightningModule
from torch import Tensor
from torch.optim import AdamW, Optimizer

from genie.modules import VisualVQDINOLatentActionModel


OptimizerCallable = Callable[[Iterable], Optimizer]


class VisualVQ_DINO_LAM(LightningModule):
    """
    Single-stage visual-only latent action model operating at DINO latent space.
    """

    def __init__(
        self,
        image_channels: int = 3,
        lam_model_dim: int = 512,
        lam_latent_dim: int = 32,
        lam_num_latents: int = 8,
        lam_patch_size: int = 16,
        lam_enc_blocks: int = 8,
        lam_dec_blocks: int = 8,
        lam_num_heads: int = 8,
        lam_dropout: float = 0.0,
        vq_beta: float = 0.25,
        radprog_radial_weight: float = 0.0,
        radprog_progress_weight: float = 0.0,
        radprog_progress_alpha: float = 0.05,
        log_interval: int = 1000,
        log_path: str = "log_imgs",
        task_name: str = "visual_vq_lam_bridge",
        optimizer: OptimizerCallable = AdamW,
        make_data_pair: bool = False,
    ) -> None:
        super().__init__()

        self.lam = VisualVQDINOLatentActionModel(
            in_dim=image_channels,
            model_dim=lam_model_dim,
            latent_dim=lam_latent_dim,
            num_latents=lam_num_latents,
            patch_size=lam_patch_size,
            enc_blocks=lam_enc_blocks,
            dec_blocks=lam_dec_blocks,
            num_heads=lam_num_heads,
            dropout=lam_dropout,
        )

        self.lam_num_latents = lam_num_latents
        self.vq_beta = vq_beta
        self.radprog_radial_weight = radprog_radial_weight
        self.radprog_progress_weight = radprog_progress_weight
        self.radprog_progress_alpha = radprog_progress_alpha
        self.log_interval = log_interval
        self.log_path = log_path
        self.optimizer = optimizer
        self.make_data_pair = make_data_pair

        self.save_hyperparameters()

        self.task_name = task_name
        self.distributed_state = PartialState()
        self._profile_steps = int(os.environ.get("UNIVLA_LAM_PROFILE_STEPS", "0"))
        self._profile_data_wait = 0.0
        self._profile_batch_start = None
        self._profile_shared_step = 0.0
        self._profile_backward_start = None
        self._profile_backward = 0.0
        self._profile_prev_batch_end = None
        self._profile_active_batch = False
        if self.distributed_state.is_main_process:
            wandb.init(name=task_name, reinit=True)

    def _code_usage(self, indices: Tensor, num_latents: int) -> Tensor:
        counts = torch.bincount(
            indices.detach().reshape(-1),
            minlength=num_latents,
        )
        return (counts != 0).float().mean()

    def _radprog_enabled(self) -> bool:
        return self.radprog_radial_weight > 0.0 or self.radprog_progress_weight > 0.0

    def _masked_mean(self, values: Tensor, mask: Tensor) -> Tensor:
        mask = mask.to(device=values.device, dtype=values.dtype)
        return (values * mask).sum() / mask.sum().clamp_min(1.0)

    def _future_latent_metrics(self, z_future: Tensor) -> Tuple:
        latent_r_future = torch.linalg.vector_norm(z_future, dim=-1).mean()
        z_flat = z_future.reshape(z_future.shape[0], -1)
        latent_batch_variance = z_flat.var(dim=0, unbiased=False).mean()
        if z_flat.shape[0] <= 1:
            latent_batch_cosine = z_flat.new_zeros(())
        else:
            z_flat = F.normalize(z_flat, dim=-1)
            cosine = z_flat @ z_flat.T
            off_diag = ~torch.eye(z_flat.shape[0], device=z_flat.device, dtype=torch.bool)
            latent_batch_cosine = cosine[off_diag].mean()
        return (
            ("latent_r_future", latent_r_future),
            ("latent_batch_cosine", latent_batch_cosine),
            ("latent_batch_variance", latent_batch_variance),
        )

    def _compute_radprog_losses(self, outputs: Dict, batch: Dict) -> Tuple[Tensor, Tensor, Tuple]:
        output_keys = ("radprog_z_self", "radprog_z_mid", "radprog_z_future")
        batch_keys = ("radprog_mid_offsets", "radprog_future_offsets", "radprog_valid")
        missing = [key for key in output_keys if key not in outputs]
        missing.extend(key for key in batch_keys if key not in batch)
        if missing:
            raise ValueError(
                "RadProg is enabled, but required keys are missing: "
                + ", ".join(sorted(missing))
            )

        z_self = outputs["radprog_z_self"]
        z_mid = outputs["radprog_z_mid"]
        z_future = outputs["radprog_z_future"]
        if z_self.shape != z_mid.shape or z_self.shape != z_future.shape:
            raise ValueError(
                "RadProg latent shapes must match, got "
                f"self={tuple(z_self.shape)}, mid={tuple(z_mid.shape)}, future={tuple(z_future.shape)}"
            )

        mid_offsets = batch["radprog_mid_offsets"].to(device=z_self.device, dtype=z_self.dtype)[:, None]
        future_offsets = batch["radprog_future_offsets"].to(device=z_self.device, dtype=z_self.dtype)[:, None]
        valid = batch["radprog_valid"].to(device=z_self.device, dtype=z_self.dtype)[:, None]
        valid = valid.expand_as(z_self[..., 0])

        d_mid = torch.linalg.vector_norm(z_mid - z_self, dim=-1)
        d_future = torch.linalg.vector_norm(z_future - z_self, dim=-1)
        radial_loss = self._masked_mean(F.softplus(d_mid - d_future), valid)

        r_self = torch.linalg.vector_norm(z_self, dim=-1)
        r_mid = torch.linalg.vector_norm(z_mid, dim=-1)
        r_future = torch.linalg.vector_norm(z_future, dim=-1)
        first_leg = F.softplus(self.radprog_progress_alpha * mid_offsets + r_self - r_mid)
        second_leg = F.softplus(self.radprog_progress_alpha * (future_offsets - mid_offsets) + r_mid - r_future)
        progress_loss = self._masked_mean(first_leg + second_leg, valid)

        logs = (
            ("radprog_radial_loss", radial_loss),
            ("radprog_progress_loss", progress_loss),
            ("radprog_valid_frac", valid.mean()),
            ("radprog_radial_order_frac", self._masked_mean((d_future > d_mid).to(d_mid.dtype), valid)),
            (
                "radprog_norm_order_frac",
                self._masked_mean(((r_mid > r_self) & (r_future > r_mid)).to(r_self.dtype), valid),
            ),
            ("radprog_d_mid", self._masked_mean(d_mid, valid)),
            ("radprog_d_future", self._masked_mean(d_future, valid)),
            ("radprog_r_self", self._masked_mean(r_self, valid)),
            ("radprog_r_mid", self._masked_mean(r_mid, valid)),
            ("radprog_r_future", self._masked_mean(r_future, valid)),
        )
        return radial_loss, progress_loss, logs

    def shared_step(self, batch: Dict) -> Tuple:
        outputs = self.lam(batch)
        gt_future_frames = outputs["target"]

        mse_loss = ((gt_future_frames - outputs["recon"]) ** 2).mean()
        q_loss = ((outputs["emb"].detach() - outputs["z"]) ** 2).mean()
        commit_loss = ((outputs["emb"] - outputs["z"].detach()) ** 2).mean()
        loss = mse_loss + q_loss + self.vq_beta * commit_loss

        code_usage = self._code_usage(outputs["indices"], self.lam_num_latents)
        loss_logs = (
            ("mse_loss", mse_loss),
            ("q_loss", q_loss),
            ("commit_loss", commit_loss),
            ("code_usage", code_usage),
        )
        loss_logs = loss_logs + self._future_latent_metrics(outputs["emb"])
        if self._radprog_enabled():
            radial_loss, progress_loss, radprog_logs = self._compute_radprog_losses(outputs, batch)
            loss = (
                loss
                + self.radprog_radial_weight * radial_loss
                + self.radprog_progress_weight * progress_loss
            )
            loss_logs = loss_logs + radprog_logs

        return outputs, loss, loss_logs

    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
        if self._profile_steps and batch_idx < self._profile_steps and torch.cuda.is_available():
            torch.cuda.synchronize()
            shared_start = time.perf_counter()
        outputs, loss, aux_losses = self.shared_step(batch)
        if self._profile_steps and batch_idx < self._profile_steps and torch.cuda.is_available():
            torch.cuda.synchronize()
            self._profile_shared_step = time.perf_counter() - shared_start

        logs = {**{"train_loss": loss}, **{f"train/{k}": v for k, v in aux_losses}}
        self.log_dict(
            logs,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=False,
        )

        if self.distributed_state.is_main_process and batch_idx % self.log_interval == 0:
            wandb.log({k: v.detach().float().cpu().item() for k, v in logs.items()})

        return loss

    def on_train_start(self) -> None:
        if self._profile_steps:
            self._profile_prev_batch_end = time.perf_counter()

    def on_train_batch_start(self, batch: Dict, batch_idx: int) -> None:
        if not self._profile_steps or batch_idx >= self._profile_steps:
            return
        now = time.perf_counter()
        self._profile_batch_start = now
        self._profile_active_batch = True
        if self._profile_prev_batch_end is not None:
            self._profile_data_wait = now - self._profile_prev_batch_end

    def on_before_backward(self, loss: Tensor) -> None:
        if not self._profile_steps or not self._profile_active_batch:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._profile_backward_start = time.perf_counter()

    def on_after_backward(self) -> None:
        if not self._profile_steps or not self._profile_active_batch or self._profile_backward_start is None:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._profile_backward = time.perf_counter() - self._profile_backward_start

    def on_train_batch_end(self, outputs: Tensor, batch: Dict, batch_idx: int) -> None:
        if not self._profile_steps:
            return
        if batch_idx < self._profile_steps:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            now = time.perf_counter()
            if not self.distributed_state.is_main_process:
                self._profile_prev_batch_end = now
                self._profile_active_batch = False
                return
            total = now - self._profile_batch_start if self._profile_batch_start is not None else 0.0
            print(
                "LAM_PROFILE "
                f"step={batch_idx + 1:06d} "
                f"data_wait={self._profile_data_wait:.3f}s "
                f"shared_step={self._profile_shared_step:.3f}s "
                f"backward={self._profile_backward:.3f}s "
                f"total={total:.3f}s",
                flush=True,
            )
            self._profile_prev_batch_end = now
        self._profile_active_batch = False

    @torch.no_grad()
    def test_step(self, batch: Dict, batch_idx: int) -> Tensor:
        outputs, loss, aux_losses = self.shared_step(batch)
        self.log_dict(
            {**{"test_loss": loss}, **{f"test/{k}": v for k, v in aux_losses}},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def on_train_epoch_end(self):
        max_steps = getattr(self.trainer, "max_steps", None)
        if max_steps is not None and max_steps > 0 and self.global_step >= max_steps:
            return
        self.lam.vq.random_restart()
        self.lam.vq.reset_usage()

    def on_test_epoch_end(self):
        if self.make_data_pair:
            completed = len(listdir("output_pairs"))
            todo_name = listdir("../data/retro")[completed]
            makedirs(f"output_pairs/{todo_name}")
            top_indices = torch.topk(self.lam.vq.usage, 16, largest=True, sorted=True).indices
            top_latents = self.lam.vq.codebook(top_indices)
            torch.save(top_latents, f"output_pairs/{todo_name}/top_16.pt")
            with open(f"output_pairs/{todo_name}/top_16.txt", "w") as f:
                f.write(" ".join([str(i) for i in top_indices.tolist()]))

        self.plot_usage_distribution(self.lam.vq.usage, "unsorted_usage")
        self.plot_usage_distribution(self.lam.vq.usage.sort().values, "sorted_usage")

    def plot_usage_distribution(self, usage, filename):
        data = usage.cpu().numpy()
        n = 1
        for n in range(1, 10):
            if (2 ** n) ** 2 <= len(data) < (2 ** (n + 1)) ** 2:
                break
        data = data.reshape(2 ** n, -1)
        fig, ax = plt.subplots()
        cax = ax.matshow(data, interpolation="nearest")
        fig.colorbar(cax)
        plt.axis("off")
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(f"{filename}.png", bbox_inches="tight", pad_inches=0.0)
        plt.close()

    def configure_optimizers(self) -> Optimizer:
        optim = self.optimizer(self.parameters())
        return optim
