from abc import abstractmethod, ABC
from enum import Enum
from typing import Literal, Any
import shutil
from pathlib import Path
from dataclasses import dataclass
from contextlib import contextmanager, nullcontext

import numpy as np
import torch.distributed as dist
from tqdm import trange, tqdm
# from torchdata.stateful_dataloader import StatefulDataLoader
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from accelerate.utils import set_seed, broadcast, DataLoaderConfiguration
from accelerate import DistributedDataParallelKwargs

from ..utils.accelerate import AcceleratorSaveTrainableParams
from ..utils import is_package_available


@dataclass(kw_only=True)
class LoggingConfig:
    report_to: str | None = "wandb"  # "wandb" | "swanlab" | "tensorboard"
    project: str
    save_dir: str | Path
    name: str
    resume_id: str | None = None
    workspace: str | None = None  # organization name in SwanLab

    def __post_init__(self):
        self.supported_loggers = ("wandb", "swanlab", "tensorboard")
        if self.report_to not in self.supported_loggers:
            raise ValueError(
                f"Unsupported logger: {self.report_to}. Supported loggers are {self.supported_loggers}."
            )

        if not is_package_available(self.report_to):
            raise ValueError(
                f"{self.report_to} is not installed. Please install {self.report_to} using `pip install {self.report_to}`."
            )


class LRSchedulerInterval(str, Enum):
    EPOCH = "epoch"
    STEP = "step"


class CheckpointMixin(ABC):
    @abstractmethod
    def state_dict(self) -> dict:
        ...

    @abstractmethod
    def load_state_dict(self, state_dict: dict) -> None:
        ...


@dataclass(kw_only=True)
class MetricMonitor(CheckpointMixin):

    metric_name: str = "loss"
    mode: Literal["min", "max"] = "min"

    def __post_init__(self):
        if self.mode not in ("min", "max"):
            raise ValueError("Mode must be 'min' or 'max'.")
        self.best_value = np.inf if self.mode == "min" else -np.inf
        self.worse_count = 0

    def compare(self, x: float, best_x: float) -> bool:
        """Compares the current value with the best value based on mode."""
        return x < best_x if self.mode == "min" else x > best_x

    def __call__(self, metric_dict: dict[str, Any]) -> bool:
        """Checks if the new value is better and updates best_value if so."""
        metric_value = metric_dict[self.metric_name]
        if isinstance(metric_value, torch.Tensor):
            metric_value = metric_value.item()
        if self.compare(metric_value, self.best_value):
            self.best_value = metric_value
            self.worse_count = 0
            return True
        self.worse_count += 1
        return False

    def state_dict(self) -> dict:
        """Returns the state of the object as a dictionary."""
        return {
            "mode": self.mode,
            "best_value": self.best_value,
            "worse_count": self.worse_count
        }

    def load_state_dict(self, state_dict: dict):
        """Loads the state from a dictionary."""
        self.mode = state_dict["mode"]
        self.best_value = state_dict["best_value"]
        self.worse_count = state_dict["worse_count"]


@dataclass(kw_only=True)
class Trainer(CheckpointMixin):
    """Base trainer class providing training workflow management.

    Usage:
        This is an abstract base class. Subclasses must implement `training_step()` and
        `validation_step()` methods to define the specific training and validation logic.

    Attributes:
        config_dict: Configuration dictionary for storing training configuration information.
        project_dir: Project root directory path for saving training-related files.
        checkpoint_dir: Checkpoint save directory. If None, uses project_dir/checkpoints.
        logging_config: Logging configuration object for experiment logging (wandb/swanlab/tensorboard).
        train_dataloader: Training data loader.
        val_dataloader: Validation data loader. Can be None to skip validation.
        model: PyTorch model to be trained.
        optimizer: Optimizer for training.
        lr_scheduler: Learning rate scheduler.
        loss_fn: Loss function.
        epochs: Total number of training epochs.
        epoch_length: Number of steps per epoch. If None, uses the length of train_dataloader.
        lr_scheduler_interval: Learning rate scheduler update interval. STEP means update every step,
            EPOCH means update every epoch.
        gradient_accumulation_steps: Number of gradient accumulation steps to simulate larger batch size.
        max_grad_norm: Maximum gradient norm for gradient clipping. If None, no gradient clipping is performed.
        resume_from_checkpoint: Path to checkpoint for resuming training. None by default, meaning training from scratch.
        save_every_n_steps: Save checkpoint every N steps. If None, no step-based saving.
        permanent_save_every_n_steps: Permanently save checkpoint to project_dir every N steps.
            If None, no permanent saving. Checkpoints saved by `save_every_n_steps` and `save_every_n_epochs`
            will be automatically deleted based on cleaning strategies but these checkpoints will not be deleted.
        save_every_n_epochs: Save checkpoint every N epochs. Default is 1 (save every epoch).
        save_last_k: Keep the last K checkpoints and delete older ones. Default is 1 (keep only the latest checkpoint).
        metric_monitor: `MetricMonitor` instance for tracking validation metrics and saving best model.
        early_stop: Early stopping patience. Stop training if validation metric doesn't improve
            for N consecutive epochs.
        even_batches: Whether to use even batches for handling inconsistent data amounts across processes.
            Must set to False when batch_sampler does not have batch_size.
    """
    config_dict: dict | None = None
    project_dir: str | Path
    checkpoint_dir: str | Path = None
    logging_config: LoggingConfig | None = None

    train_dataloader: DataLoader
    val_dataloader: DataLoader | None
    model: nn.Module
    optimizer: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler
    loss_fn: nn.Module

    epochs: int
    epoch_length: int | None = None
    lr_scheduler_interval: LRSchedulerInterval = LRSchedulerInterval.STEP
    gradient_accumulation_steps: int = 1
    max_grad_norm: float | None = 2.0
    resume_from_checkpoint: str | Path | None = None
    save_every_n_steps: int | None = None
    permanent_save_every_n_steps: int | None = None
    save_every_n_epochs: int | None = 1
    save_last_k: int | None = 1
    metric_monitor: MetricMonitor | None = None
    early_stop: int | None = None

    # use_stateful_dataloader: bool = False
    even_batches: bool = True

    def wrap_and_broadcast_value(self, value: Any) -> torch.Tensor:
        value = torch.tensor(value, device=self.accelerator.device)
        broadcast(value, from_process=0)
        return value

    def setup_accelerator(self) -> None:
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        tracker = None
        if self.logging_config is not None:
            assert self.logging_config.report_to in (
                "wandb", "swanlab", "tensorboard"
            ), (
                f"Unsupported logger: {self.logging_config.report_to}. "
                "Supported loggers are 'wandb', 'swanlab', and 'tensorboard'."
            )
            if self.logging_config.report_to == "swanlab":
                from swanlab.integration.accelerate import SwanLabTracker
                tracker = SwanLabTracker(
                    run_name=self.logging_config.project,
                    experiment_name=self.logging_config.name,
                    logdir=self.logging_config.save_dir,
                    workspace=self.logging_config.workspace,
                    resume=True,
                    id=self.logging_config.resume_id
                )
            else:
                tracker = self.logging_config.report_to

        # dataloader_config = DataLoaderConfiguration(
        # use_stateful_dataloader=self.use_stateful_dataloader
        # even_batches=self.even_batches
        # )
        self.accelerator = AcceleratorSaveTrainableParams(
            log_with=tracker,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            project_dir=self.project_dir,
            step_scheduler_with_optimizer=(
                self.lr_scheduler_interval == LRSchedulerInterval.STEP
            ),
            # dataloader_config=dataloader_config,
            kwargs_handlers=[ddp_kwargs]
        )

        train_batch_sampler = self.train_dataloader.batch_sampler
        if not hasattr(train_batch_sampler, "batch_size"):
            assert self.even_batches is False, "even_batches must be False when batch_sampler does not have batch_size"

            # due to this line: https://github.com/huggingface/accelerate/blob/main/src/accelerate/data_loader.py#L246
            assert getattr(
                train_batch_sampler, "drop_last", False
            ) is True, "drop_last must be True when batch_sampler does not have batch_size"

        self.accelerator.even_batches = self.even_batches
        # TODO when `loss_fn` does not have named_parameters/buffers, loading will raise error
        (
            self.train_dataloader,
            self.model,
            self.optimizer,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.train_dataloader,
            self.model,
            self.optimizer,
            self.lr_scheduler,
        )
        if self.val_dataloader is not None:
            self.val_dataloader = self.accelerator.prepare(self.val_dataloader)
        self.accelerator.register_for_checkpointing(self)
        for checkpoint_object in self.checkpoint_objects:
            self.accelerator.register_for_checkpointing(checkpoint_object)
        if self.resume_from_checkpoint is not None:
            self.accelerator.print(
                f"resume from checkpoint: {self.resume_from_checkpoint}"
            )
            self.accelerator.load_state(
                self.resume_from_checkpoint, strict=False
            )

    @abstractmethod
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Performs a single training step, like `training_step()` in Pytorch-Lightning.
        
        This method is called for each batch during training. Subclasses must implement
        this method to define the forward pass, loss computation, and other optional operations.
        The returned loss will be automatically used for backpropagation.
        
        Args:
            batch: A batch of data from the training DataLoader.
            batch_idx: The index of the current batch within the current epoch (0-indexed).
                This can be useful for logging or conditional logic based on batch position.
        
        Returns:
            torch.Tensor: The computed loss tensor. It will be used for `loss.backward()`.
                The tensor should be a 0-dimensional.
                The loss will be automatically logged as "train/loss" by the Trainer.
        
        Example:
            ```python
            >>> def training_step(self, batch, batch_idx):
            ...     features, labels = batch
            ...     preds = self.model(features)
            ...     loss = self.loss_fn(preds, labels)
            ...     
            ...     # Optional: Log additional metrics
            ...     lr = self.optimizer.param_groups[0]["lr"]
            ...     self.accelerator.log({"train/lr": lr}, step=self.step)
            ...     
            ...     return loss
            ```
        
        Note:
            - You should NOT call `loss.backward()` manually - the Trainer handles this automatically.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def validation_step(self, batch: Any, batch_idx: int) -> None:
        """
        Performs a single validation step, like `validation_step()` in Pytorch-Lightning.
        
        This method is called for each batch during validation. Subclasses must implement
        this method to define the prediction operation and the potential metric calculation.
        You can specify the metric calculation logic to use the metric for learning rate scheduling
        or early stopping later.
        
        Args:
            batch: A batch of data from the validation DataLoader.
            batch_idx: The index of the current batch within the validation loop (0-indexed).
                This can be useful for logging or conditional logic based on batch position.
        
        Returns:
            None: This method should not return anything. Store validation results in instance
                variables for later use in `get_val_metrics()`.
        
        Example:
            ```python
            >>> def validation_step(self, batch, batch_idx):
            ...     features, labels = batch
            ...     preds = self.model(features)
            ...     predictions = preds.argmax(dim=-1)
            ...     
            ...     # Gather predictions from all processes (important for distributed training)
            ...     output = {"predictions": predictions, "labels": labels}
            ...     output = self.accelerator.gather_for_metrics(output)
            ...     
            ...     # Accumulate metrics
            ...     accurate_preds = (output["predictions"] == output["labels"])
            ...     self.validation_stats["accurate"] += accurate_preds.long().sum()
            ...     self.validation_stats["num_elems"] += accurate_preds.shape[0]
            ```
        
        Note:
            - Use `self.accelerator.gather_for_metrics()` to collect predictions from all processes 
              before computing metrics, otherwise discrepancies between processes may result in deadlocks.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def get_context(self) -> contextmanager:
        """
        FIXME: why does it not work?
        """
        if self.even_batches:
            return nullcontext()
        else:
            return self.accelerator.join_uneven_inputs([self.model])

    def gather_min_length(self, length: int) -> int:
        length_tensor = torch.tensor(length, device=self.accelerator.device)
        dist.all_reduce(length_tensor, op=dist.ReduceOp.MIN)
        return length_tensor.item()

    def val_loop(self) -> None:
        self.model.eval()
        torch.set_grad_enabled(False)

        self.on_validation_start()

        if dist.is_initialized():
            dataloader_len = self.gather_min_length(len(self.val_dataloader))
        else:
            dataloader_len = len(self.val_dataloader)
        self.val_data_iterator = iter(self.val_dataloader)
        if self.accelerator.is_main_process:
            range_iterator = trange(
                dataloader_len,
                desc="Validation",
            )
        else:
            range_iterator = range(dataloader_len)

        for batch_idx in range_iterator:
            batch = next(self.val_data_iterator)
            self.validation_step(batch, batch_idx)

        self.on_validation_end()
        self.model.train()
        torch.set_grad_enabled(True)

    def on_validation_start(self) -> None:
        pass

    def on_validation_end(self) -> None:
        pass

    def get_val_metrics(self) -> dict[str, Any]:
        return {}

    def on_train_epoch_start(self) -> None:
        pass

    def on_train_epoch_end(self) -> None:
        pass

    @property
    def checkpoint_objects(self) -> list[CheckpointMixin]:
        """Returns a list of additional objects to be included in checkpoints.

        This property allows subclasses to specify additional objects (beyond the Trainer itself)
        that should be saved and restored during checkpointing. All objects in the returned list
        must implement the `CheckpointMixin` interface (i.e., have `state_dict()` and
        `load_state_dict()` methods). The customized checkpointing is achieved by registering these 
        objects with the Accelerate framework during `setup_accelerator()`.

        Returns:
            list[CheckpointMixin]: A list of objects to include in checkpoints. Default is an
                empty list. Subclasses can override this property to return custom objects.

        Example:
            ```python
            import torch
            from accel_hydra.trainer import CheckpointMixin

            class VersionTracker(CheckpointMixin):
                def __init__(self):
                    self.version = torch.__version__

                def state_dict(self) -> dict:
                    return {"version": self.version}
                
                def load_state_dict(self, state_dict: dict) -> None:
                    self.version = state_dict["version"]

            class MyTrainer(Trainer):
                @property
                def checkpoint_objects(self) -> list[CheckpointMixin]:
                    return [VersionTracker()]
            ```
        """
        return []

    def state_dict(self) -> dict:
        state_dict = {"epoch": self.epoch, "step": self.step}
        # state_dict = {"step": self.step}

        # if isinstance(self.train_dataloader, StatefulDataLoader):
        #     # FIXME: after `accelerator.prepare`, how to determine if `train_dataloader` is `StatefulDataLoader`?
        #     state_dict["train_dataloader"] = self.train_dataloader.state_dict()

        if self.metric_monitor is not None:
            state_dict["metric_monitor"] = self.metric_monitor.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        self.epoch = state_dict["epoch"]
        self.step = state_dict["step"]

        # self.step = state_dict["step"]
        # self.epoch = self.step // self.epoch_length

        # if "train_dataloader" in state_dict:
        #     self.train_dataloader.load_state_dict(
        #         state_dict["train_dataloader"]
        #     )
        if "metric_monitor" in state_dict:
            self.metric_monitor.load_state_dict(state_dict["metric_monitor"])

    def clean_checkpoints_to_k(
        self, checkpoints_dir: Path | str, k: int
    ) -> None:
        checkpoints_dir = Path(checkpoints_dir)
        checkpoints = (
            list(checkpoints_dir.glob("epoch_*")) +
            list(checkpoints_dir.glob("step_*"))
        )
        # sort `checkpoints` by their last modified timestamp (ascending order)
        checkpoints.sort(key=lambda x: x.stat().st_mtime)
        if k > 0:
            to_delete = checkpoints[:-k] if len(checkpoints) > k else []
        elif k == 0:
            to_delete = checkpoints
        for checkpoint in to_delete:
            shutil.rmtree(checkpoint)

    def save_checkpoint(
        self,
        save_dir: Path | str,
        clean_old_checkpoints: bool = True
    ) -> None:
        """
        Note: since `wait_for_everyone` is called, user must be responsible for making sure 
        all processes call or not call this function at the same time!!!
        """
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            save_dir = Path(save_dir)

            if clean_old_checkpoints:
                checkpoints_dir = save_dir.parent
                if self.save_last_k:
                    self.clean_checkpoints_to_k(
                        checkpoints_dir, self.save_last_k - 1
                    )

            self.accelerator.save_state(save_dir)
        self.accelerator.wait_for_everyone()

    def train_loop(self) -> None:
        torch.set_grad_enabled(True)
        self.model.train()
        self.on_train_epoch_start()

        epoch_steps = (self.epoch + 1) * self.epoch_length - self.step

        if dist.is_initialized():
            epoch_steps = self.gather_min_length(epoch_steps)
        else:
            epoch_steps = epoch_steps

        if self.accelerator.is_main_process:
            range_iterator = trange(
                epoch_steps, desc=f"Epoch {self.epoch + 1}/{self.epochs}"
            )
        else:
            range_iterator = range(epoch_steps)

        for batch_idx in range_iterator:
            try:
                batch = next(self.train_data_iterator)
            except StopIteration:
                self.train_data_iterator = iter(self.train_dataloader)
                batch = next(self.train_data_iterator)

            with self.accelerator.accumulate(self.model):
                loss = self.training_step(batch, batch_idx)
                self.accelerator.log({"train/loss": loss.item()},
                                     step=self.step)

                self.accelerator.backward(loss)

                # gradient clipping and logging
                if self.accelerator.sync_gradients:
                    if self.max_grad_norm:
                        grad_norm = self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )
                    else:
                        grad_norm = nn.utils.clip_grad_norm_(
                            self.model.parameters(), float('inf')
                        )
                    self.accelerator.log({"train/grad_norm": grad_norm},
                                         step=self.step)

                self.optimizer.step()
                if self.lr_scheduler_interval == LRSchedulerInterval.STEP:
                    self.lr_scheduler.step()
                self.optimizer.zero_grad()

            self.step += 1

            if self.save_every_n_steps:
                should_save_checkpoint = self.step % self.save_every_n_steps == 0
                if should_save_checkpoint:
                    self.save_checkpoint(
                        self.checkpoint_dir / f"step_{self.step}"
                    )

            # FIXME `self.epoch` may be not set properly at this step
            if self.permanent_save_every_n_steps:
                should_save_checkpoint = self.step % self.permanent_save_every_n_steps == 0
                if should_save_checkpoint:
                    # if self.step % self.epoch_length == 0:
                    #     self.epoch += 1
                    self.save_checkpoint(
                        self.project_dir / f"ckpt_step_{self.step}",
                        clean_old_checkpoints=False
                    )
                    # if self.step % self.epoch_length == 0:
                    #     self.epoch -= 1

        if self.val_dataloader is not None:
            self.val_loop()
        else:
            self.accelerator.print("No validation data, skipping validation")

        self.epoch += 1

        if self.lr_scheduler_interval == LRSchedulerInterval.EPOCH:
            self.lr_scheduler.step()

        if self.save_every_n_epochs:
            should_save_checkpoint = self.wrap_and_broadcast_value(
                self.epoch % self.save_every_n_epochs == 0
            )
            if should_save_checkpoint:
                self.accelerator.print("\n Saving latest checkpoint...")
                self.save_checkpoint(
                    self.checkpoint_dir / f"epoch_{self.epoch}"
                )

        if self.val_dataloader is not None:
            metric_dict: dict = self.get_val_metrics()
            if self.metric_monitor is not None:
                # save checkpoint if the monitored metric improves
                should_save_checkpoint = self.wrap_and_broadcast_value(
                    self.metric_monitor(metric_dict)
                )
                if should_save_checkpoint:
                    self.accelerator.print("\n Saving best checkpoint...")
                    self.save_checkpoint(self.checkpoint_dir / "best")

                if self.early_stop is not None and self.metric_monitor.worse_count >= self.early_stop:
                    self.should_stop_training = True

        # on start of train epoch end func
        self.on_train_epoch_end()

    def on_train_start(self) -> None:
        self.project_dir = Path(self.project_dir)
        self.project_dir.mkdir(parents=True, exist_ok=True)
        if not self.checkpoint_dir:
            self.checkpoint_dir = self.project_dir / "checkpoints"
        else:
            self.checkpoint_dir = Path(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.accelerator.print(
            f"{self.accelerator.state.num_processes} devices are used in training"
        )

        # if load from previous checkpoint, `epoch` and `step` have been set
        if not hasattr(self, "epoch"):
            self.epoch = 0
        if not hasattr(self, "step"):
            self.step = 0
        self.should_stop_training = False

        # set up `epoch_length` and training data iterator
        if self.epoch_length is None:
            self.epoch_length = len(self.train_dataloader)
        self.train_data_iterator = iter(self.train_dataloader)

        self.accelerator.print("training start ............")
        if self.logging_config is not None:
            self.accelerator.init_trackers(
                self.logging_config.project,
                init_kwargs={
                    "wandb": {
                        "name": self.logging_config.name,
                        "dir": self.logging_config.save_dir,
                        "id": self.logging_config.resume_id,
                        "resume": "allow",
                    }
                }
            )

        if self.val_dataloader is not None and self.metric_monitor is None:
            assert self.early_stop is None, "early stop does not have metrics to monitor!"

    def on_train_end(self) -> None:
        self.accelerator.print("training end ............")
        self.accelerator.end_training()
        # wandb sometimes stuck in finishing
        if is_package_available("wandb"):
            import wandb
            if wandb.run is not None:
                wandb.finish()

    def train(self, seed: int) -> None:
        set_seed(seed)
        self.setup_accelerator()

        self.on_train_start()

        for _ in range(self.epoch, self.epochs):
            self.train_loop()
            if self.should_stop_training:
                break

        self.on_train_end()
