import torch
import torch.nn as nn


try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    DEVICE_TYPE = "npu"
except ModuleNotFoundError:
    DEVICE_TYPE = "cuda"

from .text_encoder import T5TextEncoder
        
class SketchT5TextEncoder(T5TextEncoder):
    def __init__(
        self, f0_dim: int , energy_dim: int, latent_dim: int,
        embed_dim: int, model_name: str = "google/flan-t5-large",
    ):
        super().__init__(
            embed_dim = embed_dim,
            model_name = model_name,
        )
        self.f0_proj = nn.Linear(f0_dim, latent_dim)
        self.f0_norm = nn.LayerNorm(f0_dim)
        self.energy_proj = nn.Linear(energy_dim, latent_dim)

    def encode(
        self,
        text: list[str],
    ):
        with torch.no_grad(), torch.amp.autocast(
            device_type=DEVICE_TYPE, enabled=False
        ):
            return super().encode(text)
        
    def encode_sketch(
        self,
        f0,
        energy,
    ):
        f0_embed = self.f0_proj(self.f0_norm(f0)).unsqueeze(-1)
        energy_embed = self.energy_proj(energy).unsqueeze(-1)
        sketch_embed = torch.cat([f0_embed, energy_embed], dim=-1)
        return {"output": sketch_embed}


if __name__ == "__main__":
    text_encoder = T5TextEncoder(embed_dim=512)
    text = ["a man is speaking", "a woman is singing while a dog is barking"]

    output = text_encoder(text)
