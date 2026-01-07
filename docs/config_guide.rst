Configuration Guide
===================

AccelHydra uses `Hydra <https://hydra.cc/>`_ for configuration management, which provides a powerful and flexible way to organize training configurations. This guide covers the essential Hydra patterns used in AccelHydra.

Basic Concepts
--------------

``_target_`` Pattern
~~~~~~~~~~~~~~~~~~~~

The ``_target_`` key specifies the fully qualified path to a class or function that should be instantiated. Hydra's ``hydra.utils.instantiate()`` uses this to create objects from the configuration.

**Example:**

.. code-block:: yaml

   loss_fn:
     _target_: torch.nn.CrossEntropyLoss
     ignore_index: -100

Loading this and instantiate it by:

.. code-block:: python

   loss_fn = hydra.utils.instantiate(config["loss_fn"], _convert_="all")

is equivalent to:

.. code-block:: python

   import torch.nn

   loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

Parameter Passing with ``_convert_``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When instantiating objects, use ``_convert_="all"`` to ensure OmegaConf types are properly converted to Python native types:

.. code-block:: python

   model = hydra.utils.instantiate(config["model"], _convert_="all")

Configuration Structure
-----------------------

Single File Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

You can put everything in a single ``train.yaml``:

.. code-block:: yaml

   seed: 42
   exp_name: my_experiment
   epochs: 10

   model:
     _target_: model.MyModel
     hidden_dim: 512

   optimizer:
     _target_: torch.optim.AdamW
     lr: 0.001

   trainer:
     _target_: trainer.MyTrainer
     epochs: ${epochs}
     project_dir: experiments/${exp_name}

Modular Configuration with ``defaults``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For better organization, you can split configurations into multiple files using Hydra's ``defaults`` list:

.. code-block:: yaml

   # configs/train.yaml
   defaults:
     - basic
     - override hydra/hydra_logging: disabled
     - override hydra/job_logging: disabled
     - _self_

   seed: 42
   exp_name: my_experiment
   epochs: 10

   model:
     _target_: model.MyModel

   optimizer:
     _target_: torch.optim.AdamW
     lr: 0.001

   trainer:
     _target_: trainer.MyTrainer
     epochs: ${epochs}

.. code-block:: yaml

   # configs/basic.yaml
   defaults:
     - _self_
     - override hydra/hydra_logging: disabled
     - override hydra/job_logging: disabled

   data_root: /path/to/data

   hydra:
     output_subdir: null
     run:
       dir: .

The ``defaults`` list:

- ``basic``: Loads ``configs/basic.yaml`` and merges it into the current config
- ``override hydra/hydra_logging: disabled``: Disables Hydra's default logging
- ``_self_``: Includes the current file's configuration

Variable Interpolation
~~~~~~~~~~~~~~~~~~~~~~

Use ``${variable_name}`` to reference other configuration values:

.. code-block:: yaml

   exp_name: mnist
   exp_dir: experiments/${exp_name}  # Results in: experiments/mnist

   trainer:
     epochs: ${epochs}  # References the epochs value defined above

Command Line Overrides
----------------------

Hydra allows you to override any configuration value from the command line:

Basic Overrides
~~~~~~~~~~~~~~~

.. code-block:: bash

   python train.py epochs=20
   python train.py optimizer.lr=0.0005
   python train.py model.hidden_dim=1024

Multiple Overrides
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python train.py epochs=20 optimizer.lr=0.0005 trainer.save_every_n_steps=1000

Adding New Fields
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python train.py +new_param=value

Removing Fields
~~~~~~~~~~~~~~~

.. code-block:: bash

   python train.py ~val_dataloader  # Remove val_dataloader

Changing Configuration Groups
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python train.py model=resnet50  # Switch to a different model config

OmegaConf Resolvers
-------------------

AccelHydra registers custom OmegaConf resolvers that can be used in YAML files:

``len()`` Resolver
~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   vocab_size: ${len:${vocab_list}}  # Get length of a list

``multiply()`` Resolver
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   total_params: ${multiply:${num_layers},${hidden_dim}}  # Multiply values

These are registered via ``register_omegaconf_resolvers()`` which is typically called at the start of ``train.py``.

Advanced Patterns
-----------------

Conditional Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use variable interpolation to create conditional logic:

.. code-block:: yaml

   epochs: 10
   use_warmup: true
   warmup_epochs: ${epochs:2}  # Uses epochs value (10) if exists, otherwise defaults to 2

   warmup_params:
     warmup_epochs: ${warmup_epochs}  # Only used if configured

Config Composition
~~~~~~~~~~~~~~~~~~~

You can compose multiple config files:

.. code-block:: yaml

   # configs/model/resnet18.yaml
   _target_: model.ResNet18

   # configs/model/resnet50.yaml
   _target_: model.ResNet50

   # configs/train.yaml
   defaults:
     - model: resnet18
     - basic
     - _self_

Then switch models:

.. code-block:: bash

   python train.py model=resnet50

Limitations
-----------

AccelHydra does not use all of Hydra's features. Specifically:

- **Config groups**: Limited support, basic usage works but complex config groups may not
- **Multi-run (sweeps)**: Not directly supported, use external tools or Hydra's sweep CLI
- **Advanced plugins**: May require additional setup

For complete Hydra functionality, refer to the `official Hydra documentation <https://hydra.cc/docs/intro/>`_.

Examples
--------

See the `examples <https://github.com/wsntxxn/AccelHydra/tree/main/examples>`_ for complete configuration examples.
