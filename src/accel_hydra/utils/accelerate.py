from accelerate import Accelerator


class AcceleratorSaveTrainableParams(Accelerator):
    """Extended Accelerator that only saves trainable parameters and buffers.

    This class extends the base :class:`accelerate.Accelerator` class to support
    selective state dict saving. When a model has the `param_names_to_save` attribute
    (typically :class:`accel_hydra.models.common.SaveTrainableParamsBase`),
    only the parameters and buffers specified in that attribute will be saved.

    This is particularly useful for models with frozen pre-trained components,
    where you only want to save trainable parameters to save space.

    Args:
        *args: Positional arguments passed to the base :class:`accelerate.Accelerator` class.
        **kwargs: Keyword arguments passed to the base :class:`accelerate.Accelerator` class.

    Example:
        .. code-block:: python

            from accel_hydra.utils.accelerate import AcceleratorSaveTrainableParams
            from accel_hydra.models.common import SaveTrainableParamsBase
            import torch.nn as nn

            class MyModel(SaveTrainableParamsBase):
                def __init__(self):
                    super().__init__()
                    self.frozen_layer = nn.Linear(10, 10)  # Frozen pre-trained layer
                    self.trainable_layer = nn.Linear(10, 5)  # Trainable layer
                    self.frozen_layer.requires_grad_(False)

            model = MyModel()
            accelerator = AcceleratorSaveTrainableParams()
            model = accelerator.prepare(model)

            # When saving, only trainable parameters and buffers are saved
            state_dict = accelerator.get_state_dict(model)
    """
    def get_state_dict(self, model, unwrap=True):
        """Get the state dict of the model, filtering to only trainable parameters.

        Args:
            model: The model to get the state dict from.
            unwrap: Whether to unwrap the model before getting the state dict.
                Defaults to True.

        Returns:
            dict: The trainable state dict of the model. The filtering works when
                the model has the `param_names_to_save` attribute.
        """
        state_dict = super().get_state_dict(model, unwrap)
        if hasattr(model, "param_names_to_save"):
            param_names_to_save = model.param_names_to_save
            return {
                k: v
                for k, v in state_dict.items() if k in param_names_to_save
            }
        return state_dict
