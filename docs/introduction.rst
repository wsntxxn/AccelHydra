Introduction
============

What is AccelHydra?
-------------------

AccelHydra is a **lightweight, configurable** and **modular** training framework that combines the power of:

- **Accelerate**: For seamless distributed training across various hardware configurations
- **Hydra**: For flexible, composable configuration management

It IS
-----

- A trainer wrapping PyTorch, providing basic utility functions to improve the reusability of PyTorch training code
- Built on `accelerate` to support various distributed training / inference environments
- Built on `hydra` to support modular training configurations and command line overrides, with potential extended features like parameter sweeping

It IS NOT
---------

- A training framework designed for specific tasks (e.g., LLMs, image/audio-related tasks)
- A package including various state-of-the-art model implementations
- An inference-time accelerating or memory-reducing toolkit

Why AccelHydra?
---------------

Key Benefits
~~~~~~~~~~~~

1. **Avoid boilerplate code**: Common training loop and utility functions are extracted into a reusable library
2. **Simplified distributed training**: Accelerate handles the complexity of multi-GPU/multi-node setups
3. **Flexible configuration**: Hydra manages config loading with support for command-line overrides and parameter sweeps
4. **Moderate abstraction level**: Unlike deep frameworks with many inheritance layers, AccelHydra keeps code simple and understandable
5. **Separation of concerns**: Generic training logic lives in the library, while task-specific code stays in your project

When to Use AccelHydra
~~~~~~~~~~~~~~~~~~~~~~

- ✅ You want to avoid rewriting the same training loop for every project
- ✅ You need support for distributed training across multiple GPUs/nodes
- ✅ You want flexible, configurable training setups
- ✅ You prefer moderate abstraction over deep framework layers
- ✅ You're doing academic research and need to quickly prototype ideas

When Not to Use AccelHydra
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ❌ You need task-specific frameworks (use domain-specific libraries instead)
- ❌ Efficiency is critical (this library focuses on research usability, not production optimization)
- ❌ You prefer frameworks with a higher-level abstraction and extensive built-in model implementations

Installation
------------

AccelHydra is tested on Python 3.10+. We recommend creating a new virtual environment before installing:

.. code-block:: bash

   pip install accel_hydra

Or install directly from GitHub:

.. code-block:: bash

   pip install git+https://github.com/wsntxxn/AccelHydra

Quick Start
-----------

1. **Create your model**
2. **Create your trainer**: Inherit from :class:`accel_hydra.Trainer` and implement :meth:`training_step` (and optionally :meth:`validation_step`)
3. **Write your config**: Create a Hydra-style YAML configuration file
4. **Launch training**: Use the built-in :mod:`accel_hydra.train_entry` with ``accelerate launch`` to start distributed training
   - Optionally create a custom :class:`accel_hydra.TrainLauncher` if you need to customize the training setup

For a detailed walkthrough, see the :doc:`Getting Started Guide <getting_started>`.

Citation
--------

If you found this library useful, please consider citing:

.. code-block:: text

   @misc{
     title={AccelHydra: A Lightweight, Configurable and Modular Training Framework based on Accelerate and Hydra.},
     author={Xuenan Xu and Yixuan Li and Jiahao Mei},
     howpublished={\url{https://github.com/wsntxxn/AccelHydra}},
     year={2025}
   }
