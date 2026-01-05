<div>

# AccelHydra

[![python](https://img.shields.io/badge/Python_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![accelerate](https://img.shields.io/badge/Accelerate-yellow?logo=huggingface)](https://github.com/huggingface/accelerate)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![docs](https://img.shields.io/badge/docs-latest-brightgreen)](https://wsntxxn.github.io/AccelHydra)
[![license](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/wsntxxn/AccelHydra#license)

*It is still developing, with potential errors. Welcome bug report and PR!*

</div>

## :sparkles: Introduction


### :thinking: What is AccelHydra?

A **lightweight, configurable** and **modular** training framework based on [Accelerate](https://github.com/huggingface/accelerate) and [Hydra](https://github.com/facebookresearch/hydra).

It **IS**:
* a trainer wrapping PyTorch, providing some basic utility functions to improve the reusability of PyTorch training code. 
* built on [*accelerate*](https://github.com/huggingface/accelerate) to support various distributed training / inference environments.
* built on [*hydra*](https://github.com/facebookresearch/hydra) to support modular training configurations and command line overrides, with potential extended features like parameter sweeping.

It **IS NOT**:
* a training framework designed for specific tasks (e.g., LLMs, image/audio-related tasks...).
* a package including various state-of-the-art model implementations.
* an inference-time accelerating or memory-reducing toolkit.

### :bulb: Why you might want to use AccelHydra?

* Avoid writing boilerplates every time. The training loop and some utility functions remain almost the same across different projects, so we take them out as a basic library.
* The functionality of config loading is managed by Hydra, while distributed training is managed by Accelerate, so you don't need to worry about these details.
* Maintain **a moderate level of abstraction**. Great libraries like [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) and [Transformers](https://huggingface.co/docs/transformers/index) are powerful, but their codebases are too deep for newcomers to understand, or lack convenient interface to modify. We don't want a `Trainer` with dozens of inheritence layers, nor a single `train.py` with all logics in thousands of lines. 
* Similar codebases can be found: [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template/), [lightning-accelerate](https://github.com/hoang1007/lightning-accelerate). However, task-specific codes and base codes (base classes and utility functions) should be separated to continuously fix bugs in base codes. Therefore the principle here is to only integrate generic codes into the library, instead of codes designed for specific tasks (CV, NLP, RL, ...)

### :warning: Why you might not want to use AccelHydra?

* Overriding may sometimes become complicated for Hydra. Breaking training configs into different components make things clear, but it may also fail from time to time.
* Efficiency consideration. This library is suitable for acamedic research to implement an idea and test its applicability, but the efficiency for data loading, training, and inference are not involved in this library.


## :package: Installation

This repositiry is tested on Python 3.10+. We recommend creating a new virtual envrionment before installing `AccelHydra` to avoid breaking existing environments:
```bash
pip install accel_hydra
# or
pip install git+https://github.com/wsntxxn/AccelHydra
```

## :computer: Usage

Check out the [documentation](https://wsntxxn.github.io/AccelHydra) or have a look at [examples](https://github.com/wsntxxn/AccelHydra/tree/main/examples).

Basically, to use AccelHydra for training, you need to implement your own datasets, models, loss functions, and trainer:
* For basic functions and classes provided by AccelHydra, you don't need to implement again.
* The trainer should inherit `accel_hydra.Trainer` and implements necessary function `training_step` (and `validation_step` if validation is used).
* Write Hydra-style YAML configs with the top-level `train.yaml`.
* (Optional) Write a custom `TrainLauncher` if you need to customize the training setup.
* (Optional) Write a custom training entry script `train.py`.
* Launch training using the built-in entry point (for example, 8 GPUs on 2 nodes, fp16):
```bash
accelerate launch \
  --num_processes 8 \
  --num_machines 2 \
  --mixed_precision fp16 \
  -m accel_hydra.train_entry \
  -c configs/train.yaml
```

## :memo: Note

Currently AccelHydra is only tested on GPU nodes. Welcome to test this library on more machines and create corresponding PRs.

## :book: Citation

If you found this repository useful, please consider citing
```bibtex
@misc{
  title={AccelHydra: A Lightweight, Configurable and Modular Training Framework based on Accelerate and Hydra.},
  author={Xuenan Xu and Yixuan Li and Jiahao Mei},
  howpublished={\url{https://github.com/wsntxxn/AccelHydra}},
  year={2025}
}
```
