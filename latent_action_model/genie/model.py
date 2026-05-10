import os
import time
from os import listdir, makedirs, path
from typing import Callable, Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import piq
import torch
import wandb
from PIL import Image
from einops import rearrange
from lightning import LightningModule
from torch import Tensor
from torch.optim import AdamW, Optimizer
from accelerate import PartialState

OptimizerCallable = Callable[[Iterable], Optimizer]

from genie.modules import UncontrolledDINOLatentActionModel, ControllableDINOLatentActionModel
import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)



class DINO_LAM(LightningModule):
    """
    A latent action model operates at the DINO latent space
    """

    def __init__(
            self,
            image_channels: int = 3,
            # Latent action model
            lam_model_dim: int = 512,
            lam_latent_dim: int = 32,
            lam_num_latents: int = 8,
            lam_patch_size: int = 16,
            lam_enc_blocks: int = 8,
            lam_dec_blocks: int = 8,
            lam_num_heads: int = 8,
            lam_dropout: float = 0.0,
            vq_beta: float = 0.25,
            log_interval: int = 1000,
            log_path: str = "log_imgs",
            task_name: str = 'lam_openx',
            stage: str = 'stage-1',
            optimizer: OptimizerCallable = AdamW,
            make_data_pair: bool = False,
            stage_one_ckpt: str = None,
    ) -> None:
        super(DINO_LAM, self).__init__()
        assert stage in ['stage-1', 'stage-2']

        lam = UncontrolledDINOLatentActionModel if stage == 'stage-1' else ControllableDINOLatentActionModel

        self.lam = lam(
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
        
        if stage_one_ckpt and path.exists(stage_one_ckpt):
            lam_ckpt = torch.load(stage_one_ckpt)['state_dict']
            stage1_ckpt = {}
            for key in lam_ckpt.keys():
                if 'vq' in key or 'action_latent' in key:
                    stage1_ckpt[key.replace("lam.", "")] = lam_ckpt[key]
            self.lam.load_state_dict(stage1_ckpt, strict=False)


        self.lam_num_latents = lam_num_latents
        self.vq_beta = vq_beta
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

    def _entropy_from_counts(self, counts: Tensor) -> Tensor:
        counts = counts.detach().float()
        total = counts.sum()
        if total <= 0:
            return counts.new_zeros(())
        probs = counts / total
        probs = probs[probs > 0]
        return -(probs * probs.log()).sum()

    def _token_entropy(self, indices: Tensor, num_latents: int) -> Tensor:
        counts = torch.bincount(indices.detach().reshape(-1).long(), minlength=num_latents)
        return self._entropy_from_counts(counts)

    def _token_perplexity(self, indices: Tensor, num_latents: int) -> Tensor:
        return self._token_entropy(indices, num_latents).exp()

    def _rank_bins(self, values: Tensor, num_bins: int) -> Tuple[Tensor, int]:
        values = values.detach().float().reshape(-1)
        if values.numel() == 0:
            return values.new_zeros((0,), dtype=torch.long), 1
        if values.numel() <= 1 or values.var(unbiased=False) <= 1e-12:
            return torch.zeros_like(values, dtype=torch.long), 1
        bins = max(1, min(int(num_bins), int(values.numel())))
        order = torch.argsort(values)
        ranks = torch.empty_like(order)
        ranks[order] = torch.arange(values.numel(), device=values.device, dtype=order.dtype)
        return torch.div(ranks * bins, values.numel(), rounding_mode="floor").long().clamp_max(bins - 1), bins

    def _discrete_mi_nmi(self, x: Tensor, num_x: int, y: Tensor, num_y: int) -> Tuple[Tensor, Tensor]:
        x = x.detach().reshape(-1).long()
        y = y.detach().reshape(-1).long()
        if x.numel() == 0 or x.numel() != y.numel() or num_x <= 1 or num_y <= 1:
            zero = x.new_zeros((), dtype=torch.float32)
            return zero, zero

        x = x.clamp(0, num_x - 1)
        y = y.clamp(0, num_y - 1)
        joint_counts = torch.bincount(x * num_y + y, minlength=num_x * num_y).reshape(num_x, num_y).float()
        total = joint_counts.sum()
        if total <= 0:
            zero = joint_counts.new_zeros(())
            return zero, zero

        p_xy = joint_counts / total
        p_x = p_xy.sum(dim=1)
        p_y = p_xy.sum(dim=0)
        h_x = self._entropy_from_counts(p_x)
        h_y = self._entropy_from_counts(p_y)
        nonzero = p_xy > 0
        expected = (p_x[:, None] * p_y[None, :]).clamp_min(1e-12)
        mi = (p_xy[nonzero] * (p_xy[nonzero].log() - expected[nonzero].log())).sum()
        denom = torch.sqrt((h_x * h_y).clamp_min(1e-12))
        nmi = torch.where((h_x > 0) & (h_y > 0), mi / denom, mi.new_zeros(()))
        return mi, nmi

    def _action_bins(self, action: Tensor) -> Tuple[Tensor, int, Tensor, int, Tensor, int]:
        action_flat = action.detach().float().reshape(action.shape[0], -1)
        norm = torch.linalg.vector_norm(action_flat, dim=-1)
        norm_bins, num_norm_bins = self._rank_bins(norm, int(os.environ.get("UNIVLA_MI_NUM_BINS", "16")))

        dominant_dim = action_flat.abs().argmax(dim=-1)
        dominant_value = action_flat.gather(1, dominant_dim[:, None]).squeeze(1)
        direction_bins = dominant_dim * 2 + (dominant_value >= 0).long()
        num_direction_bins = max(1, action_flat.shape[-1] * 2)
        action_bins = norm_bins * num_direction_bins + direction_bins
        num_action_bins = num_norm_bins * num_direction_bins
        return norm_bins, num_norm_bins, direction_bins, num_direction_bins, action_bins, num_action_bins

    def _sample_vector_norm(self, tensor: Tensor) -> Tensor:
        norms = torch.linalg.vector_norm(tensor.detach().float(), dim=-1)
        if norms.ndim == 1:
            return norms
        return norms.reshape(norms.shape[0], -1).mean(dim=1)

    def _future_standard_indices(self, outputs: Dict, batch_size: int, key: str = "indices") -> Tensor | None:
        if key not in outputs or batch_size <= 0:
            return None
        indices = outputs[key].detach()
        if indices.shape[0] < batch_size:
            return None
        return indices.reshape(indices.shape[0], -1).long()[-batch_size:]

    def _standard_token_quality_metrics(
        self,
        outputs: Dict,
        batch: Dict,
        *,
        key: str = "indices",
        prefix: str = "code",
        num_latents: int | None = None,
    ) -> Tuple:
        if key not in outputs:
            return ()
        num_latents = int(num_latents or self.lam_num_latents)
        indices = outputs[key].detach().reshape(-1).long()
        logs = (
            (f"entropy/{prefix}", self._token_entropy(indices, num_latents)),
            (f"perplexity/{prefix}", self._token_perplexity(indices, num_latents)),
        )

        batch_size = batch["videos"].shape[0] if "videos" in batch else 0
        future_indices = self._future_standard_indices(outputs, batch_size, key=key)
        if future_indices is None:
            return logs

        if "action" in batch:
            norm_bins, num_norm_bins, direction_bins, num_direction_bins, action_bins, num_action_bins = self._action_bins(
                batch["action"].to(device=future_indices.device)
            )
            token_flat = future_indices.reshape(-1)

            norm_target = norm_bins[:, None].expand_as(future_indices).reshape(-1)
            mi, nmi = self._discrete_mi_nmi(token_flat, num_latents, norm_target, num_norm_bins)
            logs = logs + ((f"mi/{prefix}_action_norm", mi), (f"nmi/{prefix}_action_norm", nmi))

            direction_target = direction_bins[:, None].expand_as(future_indices).reshape(-1)
            mi, nmi = self._discrete_mi_nmi(token_flat, num_latents, direction_target, num_direction_bins)
            logs = logs + ((f"mi/{prefix}_action_direction", mi), (f"nmi/{prefix}_action_direction", nmi))

            action_target = action_bins[:, None].expand_as(future_indices).reshape(-1)
            mi, nmi = self._discrete_mi_nmi(token_flat, num_latents, action_target, num_action_bins)
            logs = logs + ((f"mi/{prefix}_action_bin", mi), (f"nmi/{prefix}_action_bin", nmi))

        if "patches" in outputs and outputs["patches"].shape[0] == batch_size and outputs["patches"].shape[1] >= 2:
            state_change = self._sample_vector_norm(outputs["patches"][:, -1] - outputs["patches"][:, 0])
            state_bins, num_state_bins = self._rank_bins(
                state_change.to(device=future_indices.device),
                int(os.environ.get("UNIVLA_MI_NUM_BINS", "16")),
            )
            token_flat = future_indices.reshape(-1)
            state_target = state_bins[:, None].expand_as(future_indices).reshape(-1)
            mi, nmi = self._discrete_mi_nmi(token_flat, num_latents, state_target, num_state_bins)
            logs = logs + (
                (f"mi/{prefix}_dino_state_change_norm", mi),
                (f"nmi/{prefix}_dino_state_change_norm", nmi),
            )
        return logs

    def shared_step(self, batch: Dict) -> Tuple:
        # batch: keys['videos', 'task_instruction', 'action', 'dataset_names']

        outputs = self.lam(batch)
        gt_future_frames = outputs["target"]

        # Compute loss
        mse_loss = ((gt_future_frames - outputs["recon"]) ** 2).mean()
        q_loss = ((outputs["emb"].detach() - outputs["z"]) ** 2).mean()
        commit_loss = ((outputs["emb"] - outputs["z"].detach()) ** 2).mean()

        loss = mse_loss + q_loss + self.vq_beta * commit_loss
        
        # Optimize uncontrollable queries in stage-2 (the codebook is frozen though)
        if "z_q_uncontrol" in outputs.keys():
            q_loss_uncontrol = ((outputs["emb_uncontrol"].detach() - outputs["z_uncontrol"]) ** 2).mean()
            commit_loss_uncontrol = ((outputs["emb_uncontrol"]- outputs["z_uncontrol"].detach()) ** 2).mean()
            loss = loss + q_loss_uncontrol + self.vq_beta * commit_loss_uncontrol

        code_usage = self._code_usage(outputs["indices"], self.lam_num_latents)

        loss_logs = (
            ("mse_loss", mse_loss),
            ("q_loss", q_loss),
            ("commit_loss", commit_loss),
            ("code_usage", code_usage),
        )
        loss_logs = loss_logs + self._standard_token_quality_metrics(outputs, batch)

        if "indices_uncontrol" in outputs.keys():
            uncontrol_code_usage = self._code_usage(outputs["indices_uncontrol"], self.lam.vq.num_latents)

            loss_logs = (
                ("mse_loss", mse_loss),
                ("q_loss", q_loss),
                ("commit_loss", commit_loss),
                ("q_loss_uncontrol", q_loss_uncontrol),
                ("commit_loss_uncontrol", commit_loss_uncontrol),
                ("code_usage", code_usage),
                ("code_usage_uncontrol", uncontrol_code_usage),
            )
            loss_logs = (
                loss_logs
                + self._standard_token_quality_metrics(outputs, batch)
                + self._standard_token_quality_metrics(
                    outputs,
                    batch,
                    key="indices_uncontrol",
                    prefix="uncontrol_code",
                    num_latents=self.lam.vq.num_latents,
                )
            )

        return outputs, loss, loss_logs



    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
        # Compute the training loss
        if self._profile_steps and batch_idx < self._profile_steps and torch.cuda.is_available():
            torch.cuda.synchronize()
            shared_start = time.perf_counter()
        outputs, loss, aux_losses = self.shared_step(batch)
        if self._profile_steps and batch_idx < self._profile_steps and torch.cuda.is_available():
            torch.cuda.synchronize()
            self._profile_shared_step = time.perf_counter() - shared_start


        # Log the training loss
        logs = {**{"train_loss": loss}, **{f"train/{k}": v for k, v in aux_losses}}
        self.log_dict(
            logs,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=False
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
        # Compute the test loss
        outputs, loss, aux_losses = self.shared_step(batch)

        # Log the test loss
        self.log_dict(
            {**{"test_loss": loss}, **{f"test/{k}": v for k, v in aux_losses}},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True
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
