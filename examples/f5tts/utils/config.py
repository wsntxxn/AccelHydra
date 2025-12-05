from omegaconf import OmegaConf
from accel_hydra.utils.config import register_omegaconf_resolvers as register_base_resolvers

from utils.tokenize import get_vocab_size


def add(*x):
    return sum(x)


def register_omegaconf_resolvers() -> None:
    """
    Register custom resolvers.
    This function first calls the base resolvers from accel_hydra, then registers additional resolvers specific to UniFlow-Audio.
    """
    register_base_resolvers(clear_resolvers=True)

    OmegaConf.register_new_resolver(
        "get_vocab_size", get_vocab_size, replace=True
    )
    OmegaConf.register_new_resolver("add", add, replace=True)
