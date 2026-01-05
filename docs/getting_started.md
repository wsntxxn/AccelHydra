# Getting Started

This guide will walk you through creating a complete training setup using AccelHydra, using the MNIST classification task as an example.

## Project Structure

The example project has the following structure:

```
my_project/
├── train_launcher.py     # Optional: Custom TrainLauncher (if needed)
├── model.py              # Model definition
├── trainer.py            # Custom Trainer class
├── data.py               # Dataset definitions
└── configs/
    ├── train.yaml        # Main configuration file
    └── basic.yaml        # Base configuration
```

Only `configs/train.yaml` is necessary to launch training. You can use the built-in `TrainLauncher` directly, 
or create a custom one if you need to customize the training setup. For other components like models 
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
  gradient_accumulation_steps: 4
  epochs: ${epochs}
  epoch_length: Null
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

## Step 5: Launch Training (Optional: Customize with TrainLauncher)

AccelHydra provides a common training launcher that handles the standard training launch workflow (reading configuration, instantiating datasets and models, start training). You can directly use it for basic projects.

### Option 1: Use the Default TrainLauncher (Recommended for Simple Cases)

If your project follows the standard training workflow, you can directly use the built-in `TrainLauncher`:

```bash
accelerate launch \
  --num_processes 2 \
  --num_machines 1 \
  --mixed_precision fp16 \
  -m accel_hydra.train_entry \
  -c configs/train.yaml
```

### Option 2: Create a Custom TrainLauncher (For Custom Requirements)

If you need to customize the training setup (e.g., custom dataloader initialization, custom resolver registration), you can create your own launcher by inheriting from `TrainLauncher`:

Create `train_launcher.py`:

```python
from typing import Callable
from accel_hydra import TrainLauncher

def register_my_resolvers():
    # Your custom OmegaConf resolver register function
    ...

class MyCustomLauncher(TrainLauncher):
    @staticmethod
    def get_register_resolver_fn() -> Callable:
        """Override to register custom OmegaConf resolvers."""
        return register_my_resolvers
    
    def get_dataloaders(self):
        """Override if you need custom dataloader initialization logic."""
        # You can customize dataloader creation here
        ...
    
```

Then launch training with your custom launcher:

```bash
accelerate launch \
  --num_processes 2 \
  --num_machines 1 \
  --mixed_precision fp16 \
  -m accel_hydra.train_entry \
  --launcher train_launcher.MyCustomLauncher \
  -c configs/train.yaml
```

The `TrainLauncher` base class handles:
- Configuration loading from command line
- Model, optimizer, LR scheduler, and loss function instantiation
- Trainer setup and training launch
- Resume from checkpoint support

By inheriting from `TrainLauncher`, you can override specific methods to customize the parts you need.

### Command Line Overrides

You can override any configuration value from the command line, following Hydra syntax:

```bash
accelerate launch \
  --num_processes 2 \
  --num_machines 1 \
  --mixed_precision fp16 \
  -m accel_hydra.train_entry \
  -c configs/train.yaml \
  -o data_root=/path/to/custom/data/root epochs=20 optimizer.lr=0.0005
```

For more details on Hydra syntax and overrides, see the [Configuration Guide](config_guide.md).


## Next Steps

- Learn about [Trainer hooks and customization](trainer.md)
- Understand [Hydra configuration patterns](config_guide.md)
- Explore [helper functions and utilities](api_reference.md)