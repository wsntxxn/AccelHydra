from copy import deepcopy

import hydra


def init_dataloader_from_config(config: dict):
    """
    A helper function to initialize a dataloader from a config.
    Args:
        config: A dictionary or DictConfig containing the dataloader configuration, with the format:
        '''
        _target_: torch.utils.data.DataLoader
        dataset:
          _target_: ...
          ...
        (sampler:
          _target_: ...
          ...)
        (batch_sampler:
          _target_: ...
          ...)
        batch_size: int
        num_workers: int
        '''
    Returns:
        instantiated dataloader object.
    """
    config = deepcopy(config)
    kwargs = {}
    if "sampler" in config:
        data_source = hydra.utils.instantiate(
            config["dataset"], _convert_="all"
        )
        sampler = hydra.utils.instantiate(
            config["sampler"], data_source=data_source, _convert_="all"
        )
        kwargs["sampler"] = sampler
        config.pop("sampler")
    elif "batch_sampler" in config:
        data_source = hydra.utils.instantiate(
            config["dataset"], _convert_="all"
        )
        batch_sampler = hydra.utils.instantiate(
            config["batch_sampler"], data_source=data_source, _convert_="all"
        )
        kwargs["batch_sampler"] = batch_sampler
        config.pop("batch_sampler")

    dataloader = hydra.utils.instantiate(config, **kwargs, _convert_="all")
    return dataloader
