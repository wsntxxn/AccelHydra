import json
from typing import Union, Dict
from pathlib import Path
import os

from omegaconf import OmegaConf


def is_package_available(package_name: str) -> bool:
    try:
        import importlib

        package_exists = importlib.util.find_spec(package_name) is not None
        return package_exists
    except Exception:
        return False


def read_jsonl_to_mapping(
    jsonl_file: Union[str, Path],
    key_col: str,
    value_col: str,
    base_path=None
) -> Dict[str, str]:
    """
    Read two columns, indicated by `key_col` and `value_col`, from the
    given jsonl file to return the mapping dict
    TODO handle duplicate keys
    """
    mapping = {}
    with open(jsonl_file, 'r') as file:
        for line in file.readlines():
            data = json.loads(line.strip())
            key = data[key_col]
            value = data[value_col]
            if base_path:
                value = os.path.join(base_path, value)
            mapping[key] = value
    return mapping


def setup_resume_cfg(config: dict, do_print: bool = True):
    if "resume_from_checkpoint" in config["trainer"]:
        ckpt_dir = Path(config["trainer"]["resume_from_checkpoint"])
        if "resume_from_config" in config["trainer"]:
            resumed_config = config["trainer"]["resume_from_config"]
        else:
            exp_dir = ckpt_dir.parent.parent
            resumed_config = exp_dir / "config.yaml"
        resumed_config = OmegaConf.load(resumed_config)
        resumed_config["trainer"].update({
            "resume_from_checkpoint": ckpt_dir.__str__(),
            "logging_config": config["trainer"]
                              ["logging_config"],  # for resume wandb runs
        })
    elif config.get("auto_reusme_from_latest_ckpt", False):
        exp_dir = Path(config["exp_dir"])
        ckpt_root = exp_dir / "checkpoints"
        if ckpt_root.is_dir() and any(p.is_dir() for p in ckpt_root.iterdir()):
            # use last ckpt
            ckpt_dir: Path = sorted((exp_dir / "checkpoints").iterdir())[-1]
            resumed_config = OmegaConf.load(exp_dir / "config.yaml")
            resumed_config["trainer"].update({
                "resume_from_checkpoint": ckpt_dir.__str__(),
                "logging_config": config["trainer"]
                                  ["logging_config"],  # for resume wandb runs
            })
        else:
            resumed_config = config
    else:
        resumed_config = config

    if do_print:
        if "resume_from_checkpoint" in resumed_config["trainer"]:
            print(
                f'\n train will resume from checkpoint: {resumed_config["trainer"]["resume_from_checkpoint"]}\n '
            )
        else:
            print('\n train will start from scratch\n ')

    return resumed_config
