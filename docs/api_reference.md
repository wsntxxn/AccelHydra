# API Reference

This document describes the helper functions and classes provided by AccelHydra.

## Models

::: accel_hydra.models.common
    options:
      show_source: false
      heading_level: 3
      show_root_heading: true
      show_root_toc_entry: false
      members:
        - CountParamsBase
        - LoadPretrainedBase
        - SaveTrainableParamsBase

## Trainer Components

::: accel_hydra.trainer.base
    options:
      show_source: false
      heading_level: 3
      show_root_heading: true
      show_root_toc_entry: false
      members:
        - MetricMonitor
        - LoggingConfig

## Training Launcher

::: accel_hydra.train_launcher
    options:
      show_source: false
      heading_level: 3
      show_root_heading: true
      show_root_toc_entry: false
      members:
        - TrainLauncher

## Utilities

### Data Utilities

::: accel_hydra.utils.data
    options:
      show_root_heading: true
      show_root_toc_entry: false
      show_source: false
      heading_level: 4

### Configuration Utilities

::: accel_hydra.utils.config
    options:
      show_root_heading: true
      show_root_toc_entry: false
      show_source: false
      heading_level: 4

::: accel_hydra.utils.general
    options:
      show_source: false
      heading_level: 4
      show_root_heading: true
      show_root_toc_entry: false
      members:
        - setup_resume_cfg

### Learning Rate Scheduler Utilities

::: accel_hydra.utils.lr_scheduler
    options:
      show_root_heading: true
      show_root_toc_entry: false
      show_source: false
      heading_level: 4

### PyTorch Utilities

::: accel_hydra.utils.torch
    options:
      show_root_heading: true
      show_root_toc_entry: false
      show_source: false
      heading_level: 4

### Accelerate Extensions

::: accel_hydra.utils.accelerate
    options:
      show_root_heading: true
      show_root_toc_entry: false
      show_source: false
      heading_level: 4
      members:
        - AcceleratorSaveTrainableParams
