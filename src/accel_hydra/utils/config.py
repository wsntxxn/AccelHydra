from pathlib import Path
import sys
import os
from typing import Callable

import hydra
import omegaconf
from omegaconf import OmegaConf


def multiply(*args):
    result = 1
    for arg in args:
        result *= arg
    return result


def register_omegaconf_resolvers(clear_resolvers: bool = True) -> None:
    """
    Register custom resolver for hydra configs, which can be used in YAML
    files for dynamically setting values.
    
    Args:
        clear_resolvers: If True, clear all existing resolvers before registering.
                        Set to False if you want to extend existing resolvers.
    """
    if clear_resolvers:
        OmegaConf.clear_resolvers()
    OmegaConf.register_new_resolver("len", len, replace=True)
    OmegaConf.register_new_resolver("multiply", multiply, replace=True)


def generate_config_from_command_line_overrides(
    config_file: str | Path,
    register_resolver_fn: Callable = register_omegaconf_resolvers,
) -> omegaconf.DictConfig:
    register_resolver_fn()

    config_file = Path(config_file).resolve()
    config_name = config_file.name.__str__()
    config_path = config_file.parent.__str__()
    config_path = os.path.relpath(config_path, Path(__file__).resolve().parent)

    overrides = sys.argv[1:]
    with hydra.initialize(version_base=None, config_path=config_path):
        config = hydra.compose(config_name=config_name, overrides=overrides)
    omegaconf.OmegaConf.resolve(config)

    return config
