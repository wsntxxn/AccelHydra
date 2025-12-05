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
        config["sampler"]["data_source"] = config["dataset"]
        sampler = hydra.utils.instantiate(config["sampler"], _convert_="all")
        kwargs["sampler"] = sampler
        config.pop("sampler")

        # if hasattr(sampler, "data_source"):
        #     kwargs["dataset"] = sampler.data_source
        #     config.pop("dataset")

    elif "batch_sampler" in config:
        config["batch_sampler"]["data_source"] = config["dataset"]
        batch_sampler = hydra.utils.instantiate(
            config["batch_sampler"], _convert_="all"
        )
        kwargs["batch_sampler"] = batch_sampler
        config.pop("batch_sampler")

        # if hasattr(batch_sampler, "sampler"
        #           ) and hasattr(batch_sampler.sampler, "data_source"):
        #     kwargs["dataset"] = batch_sampler.sampler.data_source
        #     config.pop("dataset")
        # elif hasattr(batch_sampler, "data_source"):
        #     kwargs["dataset"] = batch_sampler.data_source
        #     config.pop("dataset")

    dataloader = hydra.utils.instantiate(config, **kwargs, _convert_="all")
    return dataloader
