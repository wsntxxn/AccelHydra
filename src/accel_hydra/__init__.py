# Copyright (c) 2025 Xuenan Xu, Yixuan Li, Jiahao Mei. All rights reserved.
#
# Licensed under the MIT License. See LICENSE file for details.

from importlib.metadata import version

__version__ = version("accel_hydra")
__license__ = "MIT"

from .train_launcher import TrainLauncher
from .trainer import MetricMonitor, Trainer

__all__ = ["Trainer", "MetricMonitor", "TrainLauncher"]
