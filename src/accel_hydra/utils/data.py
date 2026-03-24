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
        .. code-block:: python

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
    """
    config = deepcopy(config)
    kwargs = {}
    # NOTE: Here we manually take class and instantiating arguments to avoid the dataset class being
    # converted between instance and OmegaConf.DictConfig, especially when there is randomness in
    # the initialization process of the dataset instance
    dataset = hydra.utils.instantiate(config.pop("dataset"), _convert_="all")
    kwargs["dataset"] = dataset
    if "sampler" in config:
        sampler_cfg = config.pop("sampler")
        sampler_cls = hydra.utils.get_class(sampler_cfg.pop("_target_"))
        sampler = sampler_cls(data_source=dataset, **sampler_cfg)
        kwargs["sampler"] = sampler
    elif "batch_sampler" in config:
        batch_sampler_cfg = config.pop("batch_sampler")
        batch_sampler_cls = hydra.utils.get_class(
            batch_sampler_cfg.pop("_target_")
        )
        batch_sampler = batch_sampler_cls(
            data_source=dataset, **batch_sampler_cfg
        )
        kwargs["batch_sampler"] = batch_sampler

    if "collate_fn" in config:
        collate_fn = hydra.utils.instantiate(
            config.pop("collate_fn"), _convert_="all"
        )
    else:
        collate_fn = None
    kwargs["collate_fn"] = collate_fn
    dataloader_cls = hydra.utils.get_class(config.pop("_target_"))
    dataloader = dataloader_cls(**kwargs, **config)
    return dataloader
