from pathlib import Path
import argparse
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


def load_config_with_overrides(
    config_file: str | Path,
    overrides: list[str],
    register_resolver_fn: Callable = register_omegaconf_resolvers,
) -> omegaconf.DictConfig:
    register_resolver_fn()

    config_file = Path(config_file).resolve()
    config_name = config_file.name.__str__()
    config_dir = config_file.parent.resolve().__str__()

    with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
        config = hydra.compose(config_name=config_name, overrides=overrides)

    config = OmegaConf.to_container(config, resolve=True)

    return config


def load_config_from_cli(
    return_config: bool = True,
    register_resolver_fn: Callable = register_omegaconf_resolvers,
):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        "-c",
        default="configs/train.yaml",
        type=str,
        help="Path to the config file",
    )
    parser.add_argument(
        "--overrides",
        "-o",
        default=[],
        nargs="*",
        help="Overrides to the config",
    )
    args, _ = parser.parse_known_args()

    if return_config:
        config = load_config_with_overrides(
            args.config_file, args.overrides, register_resolver_fn
        )
        return config
    else:
        return args.config_file, args.overrides


def parse_launch_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--launcher",
        "-l",
        default="accel_hydra.train_launcher.TrainLauncher",
        type=str,
        help="The entrypoint of the training script to use"
    )
    args, _ = parser.parse_known_args()
    return args.launcher
