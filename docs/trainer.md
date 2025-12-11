# Trainer

The `Trainer` class is the core component of AccelHydra. It provides a structured training loop with customizable hooks, automatic checkpoint management, logging, and distributed training support.

## Overview

The `Trainer` class wraps the standard PyTorch training workflow and integrates with Accelerate for distributed training and Hydra for configuration management. To use it, you need to inherit from `Trainer` and implement the required abstract methods.

## Training Loop Flow

The training process follows this high-level flow:

```
+---------------------------------------------------------+
|                      train(seed)                        |
|  - Sets random seed                                     |
|  - Calls setup_accelerator()                            |
|  - on_train_start()                                     |
+---------------------------+-----------------------------+
                            |
                            v
+---------------------------------------------------------+
|              Inside each epoch:                         |
|                                                         |
|  1. on_train_epoch_start()                              |
|  2. Training Loop:                                      |
|     - For each batch:                                   |
|       * training_step(batch, batch_idx)                 |
|       * Backward pass                                   |
|       * Gradient clipping                               |
|       * Optimizer step                                  |
|       * LR scheduler step (if step-based)               |
|       * Checkpoint saving (if triggered)                |
|  3. Validation Loop (if val_dataloader provided):       |
|     - on_validation_start()                             |
|     - For each batch:                                   |
|       * validation_step(batch, batch_idx)               |
|     - on_validation_end()                               |
|  4. LR scheduler step (if epoch-based)                  |
|  5. Checkpoint saving (if epoch-based trigger)          |
|  6. Best model saving (if metric_monitor improved)      |
|  7. Early stopping check                                |
|  8. on_train_epoch_end()                                |
+---------------------------+-----------------------------+
                            |
                            v
+---------------------------------------------------------+
|  - on_train_end()                                       |
+---------------------------------------------------------+
```

## Core Methods

### Abstract Methods (Must Implement)

---

#### `training_step(batch, batch_idx)`

::: accel_hydra.Trainer.training_step
    options:
      show_source: false
      heading_level: 5

---

#### `validation_step(batch, batch_idx)`

::: accel_hydra.Trainer.validation_step
    options:
      show_source: false
      heading_level: 5
---

### Hooks (Optional Override)

Hooks are optional methods you can override to customize behavior at specific points in the training process. All hooks have empty default implementations, so you only need to override the ones you need.

#### `on_train_start()`

::: accel_hydra.Trainer.on_train_start
    options:
      show_source: false
      heading_level: 5

---

#### `on_train_end()`

<!-- Called once at the end of training, after all epochs complete. Useful for cleanup tasks. -->
::: accel_hydra.Trainer.on_train_end
    options:
      show_source: false
      heading_level: 5

---

#### `on_train_epoch_start()`

<!-- Called at the beginning of each training epoch, before the training loop for that epoch starts. -->
::: accel_hydra.Trainer.on_train_epoch_start
    options:
      show_source: false
      heading_level: 5

---

#### `on_train_epoch_end()`

<!-- Called at the end of each training epoch, after validation (if applicable) completes. -->
::: accel_hydra.Trainer.on_train_epoch_end
    options:
      show_source: false
      heading_level: 5

---

#### `on_validation_start()`

<!-- Called at the beginning of validation, before the validation loop starts. Typically used to initialize or reset metric accumulators. -->

::: accel_hydra.Trainer.on_validation_start
    options:
      show_source: false
      heading_level: 5

---

#### `on_validation_end()`

<!-- Called at the end of validation, after all validation batches have been processed. Can be used for logging or custom logic. -->

::: accel_hydra.Trainer.on_validation_end
    options:
      show_source: false
      heading_level: 5

---

#### `get_val_metrics()`

<!-- Called automatically during validation after `on_validation_end()`. Should return a dictionary of validation metrics that will be used by `MetricMonitor` for best model saving. -->

::: accel_hydra.Trainer.get_val_metrics
    options:
      show_source: false
      heading_level: 5

---

### Example: Complete Trainer Implementation

For a complete example, see [Step 3: Define Your Trainer](getting_started.md#step-3-define-your-trainer) in the Getting Started guide.


## Full Trainer API

::: accel_hydra.Trainer
    options:
      show_source: false
      heading_level: 3
      members: [checkpoint_objects]
      show_root_heading: true
      show_root_toc_entry: true
