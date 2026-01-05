# AccelHydra

[![python](https://img.shields.io/badge/Python_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![accelerate](https://img.shields.io/badge/Accelerate-yellow?logo=huggingface)](https://github.com/huggingface/accelerate)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![license](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/wsntxxn/AccelHydra#license)

*A lightweight, configurable and modular training framework based on [Accelerate](https://github.com/huggingface/accelerate) and [Hydra](https://github.com/facebookresearch/hydra).*

## What is AccelHydra?

AccelHydra is a **lightweight, configurable** and **modular** training framework that combines the power of:

- **[Accelerate](https://github.com/huggingface/accelerate)**: For seamless distributed training across various hardware configurations
- **[Hydra](https://github.com/facebookresearch/hydra)**: For flexible, composable configuration management

### It **IS**:

- A trainer wrapping PyTorch, providing basic utility functions to improve the reusability of PyTorch training code
- Built on `accelerate` to support various distributed training / inference environments
- Built on `hydra` to support modular training configurations and command line overrides, with potential extended features like parameter sweeping

### It **IS NOT**:

- A training framework designed for specific tasks (e.g., LLMs, image/audio-related tasks)
- A package including various state-of-the-art model implementations
- An inference-time accelerating or memory-reducing toolkit

## Why AccelHydra?

### Key Benefits:

1. **Avoid boilerplate code**: Common training loop and utility functions are extracted into a reusable library
2. **Simplified distributed training**: Accelerate handles the complexity of multi-GPU/multi-node setups
3. **Flexible configuration**: Hydra manages config loading with support for command-line overrides and parameter sweeps
4. **Moderate abstraction level**: Unlike deep frameworks with many inheritance layers, AccelHydra keeps code simple and understandable
5. **Separation of concerns**: Generic training logic lives in the library, while task-specific code stays in your project

### When to Use AccelHydra:

- ✅ You want to avoid rewriting the same training loop for every project
- ✅ You need support for distributed training across multiple GPUs/nodes
- ✅ You want flexible, configurable training setups
- ✅ You prefer moderate abstraction over deep framework layers
- ✅ You're doing academic research and need to quickly prototype ideas

### When Not to Use AccelHydra:

- ❌ You need task-specific frameworks (use domain-specific libraries instead)
- ❌ Efficiency is critical (this library focuses on research usability, not production optimization)
- ❌ You prefer frameworks with a higher-level abstraction and extensive built-in model implementations

## Installation

AccelHydra is tested on Python 3.10+. We recommend creating a new virtual environment before installing:

```bash
pip install accel_hydra
```

Or install directly from GitHub:

```bash
pip install git+https://github.com/wsntxxn/AccelHydra
```

## Quick Start

1. **Create your model**
2. **Create your trainer**: Inherit from `accel_hydra.Trainer` and implement `training_step` (and optionally `validation_step`)
3. **Write your config**: Create a Hydra-style YAML configuration file
4. **Launch training**: Use the built-in `accel_hydra.train_entry` with `accelerate launch` to start distributed training
   - Optionally create a custom `TrainLauncher` if you need to customize the training setup

For a detailed walkthrough, see the [Getting Started Guide](getting_started.md).

## Citation

If you found this library useful, please consider citing:

```bibtex
@misc{
  title={AccelHydra: A Lightweight, Configurable and Modular Training Framework based on Accelerate and Hydra.},
  author={Xuenan Xu and Yixuan Li and Jiahao Mei},
  howpublished={\url{https://github.com/wsntxxn/AccelHydra}},
  year={2025}
}
```
