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
