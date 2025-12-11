# Getting Started

This guide will walk you through creating a complete training setup using AccelHydra, using the MNIST classification task as an example.

## Project Structure

The example project has the following structure:

```
my_project/
├── train.py              # Main training script
├── model.py              # Model definition
├── trainer.py            # Custom Trainer class
├── data.py               # Dataset definitions
└── configs/
    ├── train.yaml        # Main configuration file
    └── basic.yaml        # Base configuration
```

Only `train.py` and `configs/train.yaml` are necessary to launch training. For other components like models 
and datasets, the definition location is not important. The only requirement is that their configuration in
`configs/train.yaml` can be instantiated by `hydra.utils.instantiate()`.

## Step 1: Define Your Model

Create `model.py` and define your PyTorch model:

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=0.1)
        self.adaptive_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.dropout(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
```

## Step 2: Define Your Datasets

Create `data.py` to define datasets. Here we directly use `torchvision.datasets.MNIST` for convenience:

```python
import torchvision
from torchvision import transforms

def train_dataset(data_root: str):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.MNIST(
        root=data_root, train=True, download=True, transform=transform
    )
    return dataset

def val_dataset(data_root: str):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.MNIST(
        root=data_root, train=False, download=True, transform=transform
    )
    return dataset
```

## Step 3: Define Your Trainer

Create `trainer.py` and define your Trainer class. You must implement `training_step` and `validation_step`:

```python
from dataclasses import dataclass
from accel_hydra import Trainer

@dataclass(kw_only=True)
class MnistTrainer(Trainer):
    def training_step(self, batch, batch_idx):
        features, labels = batch
        preds = self.model(features)
        loss = self.loss_fn(preds, labels)
        
        # Log learning rate
        lr = self.optimizer.param_groups[0]["lr"]
        self.accelerator.log({"train/lr": lr}, step=self.step)
        
        return loss

    def on_validation_start(self):
        # Initialize metric accumulators
        self.validation_stats = {"accurate": 0, "num_elems": 0}

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        preds = self.model(features)
        predictions = preds.argmax(dim=-1)
        
        # Gather predictions from all processes (important for distributed training)
        output = {"predictions": predictions, "labels": labels}
        output = self.accelerator.gather_for_metrics(output)
        
        # Accumulate metrics
        accurate_preds = (output["predictions"] == output["labels"])
        self.validation_stats["accurate"] += accurate_preds.long().sum()
        self.validation_stats["num_elems"] += accurate_preds.shape[0]

    def get_val_metrics(self):
        return {
            "accuracy": self.validation_stats["accurate"].item() / 
                       self.validation_stats["num_elems"]
        }

    def on_validation_end(self):
        eval_metric = self.validation_stats["accurate"].item() / \
                     self.validation_stats["num_elems"]
        self.accelerator.print(
            f"epoch[{self.epoch}] --> eval_metric= {100 * eval_metric:.2f}%"
        )
        self.accelerator.log({"val/accuracy": eval_metric}, step=self.step)
```

## Step 4: Write Configuration YAML(s)

Create `configs/basic.yaml` for base configuration:

```yaml
defaults:
  - _self_
  - override hydra/hydra_logging: disabled
  - override hydra/job_logging: disabled

data_root: YOUR_DATA_ROOT

hydra:
  output_subdir: null
  run:
    dir: .
```

This is optional for better disentanglement. You can also put everything in the following `configs/train.yaml`.

Create `configs/train.yaml` for the main training configuration:

```yaml
defaults:
  - basic
  - _self_

seed: 42
exp_name: mnist
exp_dir: experiments/${exp_name}
epochs: 10
epoch_length: Null
gradient_accumulation_steps: 4

train_dataloader:
  _target_: torch.utils.data.DataLoader
  dataset:
    _target_: data.train_dataset
    data_root: ${data_root}
  batch_size: 128
  shuffle: true
  num_workers: 4

val_dataloader:
  _target_: torch.utils.data.DataLoader
  dataset:
    _target_: data.val_dataset
    data_root: ${data_root}
  batch_size: 128

optimizer:
  _target_: torch.optim.AdamW
  lr: 0.0001

lr_scheduler:
  _target_: torch.optim.lr_scheduler.OneCycleLR
  max_lr: 0.0025
  epochs: ${epochs}

model:
  _target_: model.Model

loss_fn:
  _target_: torch.nn.CrossEntropyLoss

trainer:
  _target_: trainer.MnistTrainer
  project_dir: ${exp_dir}
  logging_config:
    _target_: accel_hydra.trainer.LoggingConfig
    report_to: tensorboard
    project: runs
    save_dir: ${exp_dir}
    name: ${exp_name}
  gradient_accumulation_steps: ${gradient_accumulation_steps}
  epochs: ${epochs}
  epoch_length: ${epoch_length}
  save_every_n_steps: 500
  save_every_n_epochs: 1
  save_last_k: 2
  metric_monitor:
    _target_: accel_hydra.MetricMonitor
    metric_name: accuracy
    mode: max
```

The configuration uses Hydra's `_target_` pattern to specify class instantiation. This allows flexible configuration without hardcoding imports.
However, this also makes it difficult to figure out returned classes in the code, so we recommend using typing annotations to improve the 
readability of the code.

## Step 5: Write the Training Script

Create `train.py` as the main entry point:

```python
from pathlib import Path
import multiprocessing as mp

mp.set_start_method("spawn", force=True)

import hydra
from omegaconf import OmegaConf
from accelerate.state import PartialState

from accel_hydra import Trainer
from accel_hydra.models import CountParamsBase
from accel_hydra.utils import (
    register_omegaconf_resolvers,
    setup_resume_cfg,
    init_dataloader_from_config,
)
from accel_hydra.utils.lr_scheduler import (
    get_warmup_steps,
    get_dataloader_one_pass_outside_steps,
    get_total_training_steps,
    get_steps_inside_accelerator_from_outside_steps,
    get_dataloader_one_pass_steps_inside_accelerator,
    lr_scheduler_param_adapter,
)

register_omegaconf_resolvers()

def main():
    configs = []

    @hydra.main(version_base=None, config_path="configs", config_name="train")
    def parse_config_from_command_line(config):
        config = OmegaConf.to_container(config, resolve=True)
        configs.append(config)

    parse_config_from_command_line()
    config = configs[0]

    # Helper state for accessing information about the current training environment
    state = PartialState()

    # Optional: dump config to file
    if config.get("dump_config", None) is not None:
        if state.is_main_process:
            with open(config["dump_config"], "w") as f:
                OmegaConf.save(config, f)
                print(f'config.yaml saved to {f.name}')
        return

    # Setup resume configuration if needed
    config = setup_resume_cfg(config, do_print=state.is_main_process)

    # Instantiate model
    model: CountParamsBase = hydra.utils.instantiate(
        config["model"], _convert_="all"
    )
    
    # Initialize dataloaders
    train_dataloader = init_dataloader_from_config(config["train_dataloader"])
    if "val_dataloader" in config and config["val_dataloader"] is not None:
        val_dataloader = init_dataloader_from_config(config["val_dataloader"])
    else:
        val_dataloader = None
    
    # Instantiate optimizer
    optimizer = hydra.utils.instantiate(
        config["optimizer"], params=model.parameters(), _convert_="all"
    )

    # Calculate training steps (important for distributed training + gradient accumulation)
    dataloader_one_pass_outside_steps = get_dataloader_one_pass_outside_steps(
        train_dataloader, state.num_processes
    )
    total_training_steps = get_total_training_steps(
        train_dataloader, config["epochs"], state.num_processes,
        config["epoch_length"]
    )
    dataloader_one_pass_steps_inside_accelerator = (
        get_dataloader_one_pass_steps_inside_accelerator(
            dataloader_one_pass_outside_steps,
            config["gradient_accumulation_steps"], state.num_processes
        )
    )
    num_training_updates = get_steps_inside_accelerator_from_outside_steps(
        total_training_steps, dataloader_one_pass_outside_steps,
        dataloader_one_pass_steps_inside_accelerator,
        config["gradient_accumulation_steps"], state.num_processes
    )

    # Setup warmup if configured
    if "warmup_params" in config:
        num_warmup_steps = get_warmup_steps(
            **config["warmup_params"],
            dataloader_one_pass_outside_steps=dataloader_one_pass_outside_steps
        )
        num_warmup_updates = get_steps_inside_accelerator_from_outside_steps(
            num_warmup_steps, dataloader_one_pass_outside_steps,
            dataloader_one_pass_steps_inside_accelerator,
            config["gradient_accumulation_steps"], state.num_processes
        )
    else:
        num_warmup_updates = None

    # Adapt LR scheduler parameters
    lr_scheduler_config = lr_scheduler_param_adapter(
        config_dict=config["lr_scheduler"],
        num_training_steps=num_training_updates,
        num_warmup_steps=num_warmup_updates,
    )

    # Instantiate LR scheduler
    lr_scheduler = hydra.utils.instantiate(
        lr_scheduler_config, optimizer=optimizer, _convert_="all"
    )
    
    # Instantiate loss function
    loss_fn = hydra.utils.instantiate(config["loss_fn"], _convert_="all")
    
    # Instantiate trainer
    trainer: Trainer = hydra.utils.instantiate(
        config["trainer"],
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss_fn=loss_fn,
        _convert_="all"
    )
    trainer.config_dict = config  # Assign config for potential use in hooks
    
    # Start training
    trainer.train(seed=config["seed"])

if __name__ == "__main__":
    main()
```

This script can be used as a template for your own training script. It implements a common training launch workflow
so it may be usable in many projects.

## Step 6: Launch Training

```bash
accelerate launch \
  --num_processes 2 \
  --num_machines 1 \
  --mixed_precision fp16 \
  train.py
```

### Distributed Training

Distributed training is easy using accelerate launch parameters. For example, to train on 2 nodes with 8 GPUs each and 
mixed precison of fp16:
```bash
accelerate launch \
  --num_processes 16 \
  --num_machines 2 \
  --mixed_precision fp16 \
  train.py
```

### Command Line Overrides

You can override any configuration value from the command line, following Hydra syntax:

```bash
python train.py epochs=20 optimizer.lr=0.0005 trainer.save_every_n_steps=1000
```

For more details on Hydra syntax and overrides, see the [Configuration Guide](config_guide.md).


## Next Steps

- Learn about [Trainer hooks and customization](trainer.md)
- Understand [Hydra configuration patterns](config_guide.md)
- Explore [helper functions and utilities](api_reference.md)