from omegaconf import OmegaConf
from accel_hydra.utils.config import register_omegaconf_resolvers as register_base_resolvers


def get_pitch_downsample_ratio(
    autoencoder_config: dict, pitch_frame_resolution: float
):
    latent_frame_resolution = autoencoder_config[
        "downsampling_ratio"] / autoencoder_config["sample_rate"]
    return round(latent_frame_resolution / pitch_frame_resolution)


def register_omegaconf_resolvers() -> None:
    """
    Register custom resolvers.
    This function first calls the base resolvers from accel_hydra, then registers additional resolvers specific to UniFlow-Audio.
    """
    register_base_resolvers(clear_resolvers=True)

    OmegaConf.register_new_resolver(
        "get_pitch_downsample_ratio", get_pitch_downsample_ratio, replace=True
    )
