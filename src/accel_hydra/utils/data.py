from copy import deepcopy

import hydra


def init_dataloader_from_config(config: dict):
    """
    A helper function to initialize a dataloader from a config.

    Args:
        config: A dictionary or DictConfig containing the dataloader configuration.
    
    Returns:
        instantiated dataloader object.

    Example:
        ```python
        config = '''
        train_dataloader:
        _target_: torch.utils.data.DataLoader
        dataset:
            _target_: data.train_dataset
            data_root: /path/to/data
        # sampler:
        #   _target_: torch.utils.data.Sampler
        #   ...
        # batch_sampler:
        #   _target_: torch.utils.data.BatchSampler
        #   ...
        '''
        config = OmegaConf.create(config)
        train_dataloader = init_dataloader_from_config(config["train_dataloader"])
        ```
    """
    config = deepcopy(config)
    kwargs = {}
    # NOTE: Here we put "dataset" unchanged in the config dict instead of using instantiated dataset
    # because of this bug: https://github.com/omry/omegaconf/issues/731, and standard
    # PyTorch `Dataset` class inherits `Generic`
    #
    # You can modify this to use the exact same `dataset` to instantiate the sampler/batch_sampler and
    # dataloader, as long as your dataset does not inherit `Generic`
    if "sampler" in config:
        config["sampler"]["data_source"] = config["dataset"]
        sampler = hydra.utils.instantiate(config["sampler"], _convert_="all")
        kwargs["sampler"] = sampler
        config.pop("sampler")

    elif "batch_sampler" in config:
        config["batch_sampler"]["data_source"] = config["dataset"]
        batch_sampler = hydra.utils.instantiate(
            config["batch_sampler"], _convert_="all"
        )
        kwargs["batch_sampler"] = batch_sampler
        config.pop("batch_sampler")

    dataloader = hydra.utils.instantiate(config, **kwargs, _convert_="all")
    return dataloader
