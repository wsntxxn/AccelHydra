# Copyright (c) 2025 Xuenan Xu, Yixuan Li, Jiahao Mei. All rights reserved.
#
# Licensed under the MIT License. See LICENSE file for details.

from importlib.metadata import version

__version__ = version("accel_hydra")
__license__ = "MIT"

from .trainer import Trainer, MetricMonitor
from .train_launcher import TrainLauncher
