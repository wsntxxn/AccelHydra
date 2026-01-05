from dataclasses import dataclass
import gc
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils._pytree import tree_map
from omegaconf import OmegaConf

from accel_hydra.trainer import Trainer
from accel_hydra.utils.logging import LoggingLogger


@dataclass(kw_only=True)
class F5TTSTrainer(Trainer):

    logging_file: str | Path

    def on_train_start(self):
        super().on_train_start()
        self.train_loss = 0
        self.train_batch_num = 0
        if self.accelerator.is_main_process:
            self.logger = LoggingLogger(self.logging_file).create_instance()
            with open(self.project_dir / "config.yaml", "w") as writer:
                OmegaConf.save(self.config_dict, writer)

        if isinstance(self.model, DistributedDataParallel):
            num_params, trainable_params = self.model.module.count_params()
        else:
            num_params, trainable_params = self.model.count_params()

        if self.accelerator.is_main_process:
            self.logger.info(
                f"parameter number: {num_params}, trainable parameter number: {trainable_params}"
            )

    def on_train_epoch_start(self):
        self.train_loss = 0
        self.train_batch_num = 0

    def training_step(self, batch, batch_idx):
        output = self.model(**batch)
        lr = self.optimizer.param_groups[0]["lr"]
        self.accelerator.log({"train/lr": lr}, step=self.step)
        loss_dict = self.loss_fn(output)
        log_dict = {}
        for loss_name in loss_dict:
            if loss_name != "loss":
                log_dict[f"train/{loss_name}"] = loss_dict[loss_name].item()
        self.accelerator.log(
            log_dict,
            step=self.step,
        )
        loss = loss_dict["loss"]

        self.train_loss += loss.item()
        self.train_batch_num += 1
        return loss

    def on_validation_start(self):
        self.val_loss_dict = defaultdict(float)
        self.val_batch_num = 0
        self.val_time_aligned_batch_num = 0

    def validation_step(self, batch, batch_idx):
        output = self.model(**batch)
        output = self.accelerator.gather_for_metrics(output)
        output = tree_map(lambda x: x.mean(), output)
        loss_dict = self.loss_fn(output)
        for loss_name in loss_dict:
            self.val_loss_dict[loss_name] += loss_dict[loss_name].item()
        self.val_batch_num += 1

    def on_train_epoch_end(self):
        train_loss = self.train_loss / self.train_batch_num
        logging_msg = f"epoch[{self.epoch}], train loss: {train_loss:.3f}"
        self.accelerator.print(logging_msg)
        if self.accelerator.is_main_process:
            self.logger.info(logging_msg)

        torch.cuda.empty_cache()
        gc.collect()
