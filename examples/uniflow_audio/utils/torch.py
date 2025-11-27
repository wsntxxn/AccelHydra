import torch

from accel_hydra.utils.torch import create_mask_from_length


def convert_pad_shape(pad_shape: list[list[int]]):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def create_alignment_path(duration: torch.Tensor, mask: torch.Tensor):
    b, t_x, t_y = mask.shape
    cum_duration = torch.cumsum(duration, 1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = create_mask_from_length(cum_duration_flat, t_y).float()
    path = path.view(b, t_x, t_y)
    # take the diff on the `t_x` axis
    path = path - torch.nn.functional.pad(
        path, convert_pad_shape([[0, 0], [1, 0], [0, 0]])
    )[:, :-1]
    path = path * mask
    return path
