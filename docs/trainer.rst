Trainer
=======

The :class:`accel_hydra.Trainer` class is the core component of AccelHydra. It provides a structured training loop with customizable hooks, automatic checkpoint management, logging, and distributed training support.

Overview
--------

The :class:`accel_hydra.Trainer` class wraps the standard PyTorch training workflow and integrates with Accelerate for distributed training and Hydra for configuration management. To use it, you need to inherit from :class:`accel_hydra.Trainer` and implement the required abstract methods.

Training Loop Flow
------------------

The training process follows this high-level flow:

.. code-block:: text

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

Core Methods
------------

Abstract Methods (Must Implement)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: accel_hydra.trainer.Trainer.training_step
   :noindex:

.. automethod:: accel_hydra.trainer.Trainer.validation_step
   :noindex:

Hooks (Optional Override)
~~~~~~~~~~~~~~~~~~~~~~~~~

Hooks are optional methods you can override to customize behavior at specific points in the training process. All hooks have empty default implementations, so you only need to override the ones you need.

.. automethod:: accel_hydra.trainer.Trainer.on_train_start
   :noindex:

.. automethod:: accel_hydra.trainer.Trainer.on_train_end
   :noindex:

.. automethod:: accel_hydra.trainer.Trainer.on_train_epoch_start
   :noindex:

.. automethod:: accel_hydra.trainer.Trainer.on_train_epoch_end
   :noindex:

.. automethod:: accel_hydra.trainer.Trainer.on_validation_start
   :noindex:

.. automethod:: accel_hydra.trainer.Trainer.on_validation_end
   :noindex:

.. automethod:: accel_hydra.trainer.Trainer.get_val_metrics
   :noindex:

Example: Complete Trainer Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a complete example, see :ref:`getting-started-step-3` in the Getting Started guide.

Full Trainer API
----------------

.. autoclass:: accel_hydra.trainer.Trainer
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :noindex:
