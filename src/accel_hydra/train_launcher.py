import hydra
from omegaconf import OmegaConf
from accelerate.state import PartialState
from torch.utils.data import DataLoader
from typing import Callable

from .trainer import Trainer
from .models import CountParamsBase
from .utils.config import load_config_from_cli, register_omegaconf_resolvers
from .utils.general import setup_resume_cfg
from .utils.data import init_dataloader_from_config
from .utils.lr_scheduler import (
    get_warmup_steps,
    get_dataloader_one_pass_outside_steps,
    get_total_training_steps,
    get_steps_inside_accelerator_from_outside_steps,
    get_dataloader_one_pass_steps_inside_accelerator,
    lr_scheduler_param_adapter,
)


class TrainLauncher:
    """Base class for training launchers that handle configuration and training setup.
    
    This class provides a structured way to launch training with Hydra configuration.
    Subclasses can override specific methods to customize the training process.
    """
    @staticmethod
    def get_register_resolver_fn() -> Callable:
        """Get the function to register custom OmegaConf resolvers.
        
        Subclasses can override this staticmethod to return a custom resolver registration function.
        The returned function should be callable without arguments.
        
        Returns:
            Callable: A function that registers custom OmegaConf resolvers
        """
        return register_omegaconf_resolvers

    def get_steps_for_lr_scheduler(self, train_dataloader: DataLoader):
        """Calculate steps for LR scheduler.
        
        This method handles the complexity of step counting in distributed training
        with gradient accumulation.
        
        Args:
            train_dataloader: The training dataloader
            
        Returns:
            tuple: (num_training_updates, num_warmup_updates) where num_warmup_updates can be None
        """
        # `accelerator.prepare` is very confusing for multi-gpu, gradient accumulation scenario:
        # For more information: see https://github.com/huggingface/diffusers/issues/4387,
        # https://github.com/huggingface/diffusers/issues/9633, and
        # https://github.com/huggingface/diffusers/issues/3954
        dataloader_one_pass_outside_steps = get_dataloader_one_pass_outside_steps(
            train_dataloader, self.state.num_processes
        )
        total_training_steps = get_total_training_steps(
            train_dataloader,
            self.config["trainer"]["epochs"],
            self.state.num_processes,
            self.config["trainer"]["epoch_length"],
        )
        dataloader_one_pass_steps_inside_accelerator = (
            get_dataloader_one_pass_steps_inside_accelerator(
                dataloader_one_pass_outside_steps,
                self.config["trainer"]["gradient_accumulation_steps"],
                self.state.num_processes
            )
        )
        num_training_updates = get_steps_inside_accelerator_from_outside_steps(
            total_training_steps, dataloader_one_pass_outside_steps,
            dataloader_one_pass_steps_inside_accelerator,
            self.config["trainer"]["gradient_accumulation_steps"],
            self.state.num_processes
        )

        if "warmup_params" in self.config:
            num_warmup_steps = get_warmup_steps(
                **self.config["warmup_params"],
                dataloader_one_pass_outside_steps=
                dataloader_one_pass_outside_steps
            )
            num_warmup_updates = get_steps_inside_accelerator_from_outside_steps(
                num_warmup_steps, dataloader_one_pass_outside_steps,
                dataloader_one_pass_steps_inside_accelerator,
                self.config["gradient_accumulation_steps"],
                self.state.num_processes
            )
        else:
            num_warmup_updates = None

        return num_training_updates, num_warmup_updates

    def get_dataloaders(self, ):
        train_dataloader = init_dataloader_from_config(
            self.config["train_dataloader"]
        )
        if "val_dataloader" in self.config and self.config["val_dataloader"
                                                          ] is not None:
            val_dataloader = init_dataloader_from_config(
                self.config["val_dataloader"]
            )
        else:
            val_dataloader = None
        return train_dataloader, val_dataloader

    def run(self, ):
        """Main entry point that orchestrates the training setup and launch.
        
        This method follows the standard training setup flow:
        1. Load configuration
        2. Setup resume if needed
        3. Create model, dataloaders, optimizer, LR scheduler, loss function
        4. Create trainer and start training
        
        Subclasses can override this method to customize the entire flow, or
        override individual methods to customize specific steps.
        """

        register_resolver_fn = self.get_register_resolver_fn()
        config = load_config_from_cli(
            register_resolver_fn=register_resolver_fn
        )

        state = PartialState()
        self.config, self.state = config, state

        if config.get("dump_config", None) is not None:
            if state.is_main_process:
                with open(config["dump_config"], "w") as f:
                    OmegaConf.save(config, f)
                    print(f"config.yaml saved to {config['dump_config']}")
            return

        config = setup_resume_cfg(config, do_print=state.is_main_process)

        model: CountParamsBase = hydra.utils.instantiate(
            config["model"], _convert_="all"
        )

        train_dataloader, val_dataloader = self.get_dataloaders()

        optimizer = hydra.utils.instantiate(
            config["optimizer"], params=model.parameters(), _convert_="all"
        )

        num_training_updates, num_warmup_updates = self.get_steps_for_lr_scheduler(
            train_dataloader
        )

        lr_scheduler_config = lr_scheduler_param_adapter(
            config_dict=config["lr_scheduler"],
            num_training_steps=num_training_updates,
            num_warmup_steps=num_warmup_updates,
        )

        lr_scheduler = hydra.utils.instantiate(
            lr_scheduler_config, optimizer=optimizer, _convert_="all"
        )
        loss_fn = hydra.utils.instantiate(config["loss_fn"], _convert_="all")

        trainer: Trainer = hydra.utils.instantiate(
            config["trainer"],
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            loss_fn=loss_fn,
            _convert_="all"
        )
        trainer.config_dict = config  # assign here, don't instantiate it

        trainer.train(seed=config["seed"])
