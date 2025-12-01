import sys
import os

os.environ[
    "BIGVGAN_MODULE_PATH"
] = "/mnt/shared-storage-user/xuxuenan/workspace/f5tts/src/third_party/BigVGAN"
sys.path.insert(1, os.environ["BIGVGAN_MODULE_PATH"])

import torch
from huggingface_hub import hf_hub_download
from vocos import Vocos

import bigvgan

device = (
    "cuda" if torch.cuda.is_available() else "xpu" if torch.xpu.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


# load vocoder
def load_vocoder(
    vocoder_name="vocos",
    is_local=False,
    local_path="",
    device=device,
    hf_cache_dir=None
):
    if vocoder_name == "vocos":
        # vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
        if is_local:
            print(f"Load vocos from local path {local_path}")
            config_path = f"{local_path}/config.yaml"
            model_path = f"{local_path}/pytorch_model.bin"
        else:
            print("Download Vocos from huggingface charactr/vocos-mel-24khz")
            repo_id = "charactr/vocos-mel-24khz"
            config_path = hf_hub_download(
                repo_id=repo_id,
                cache_dir=hf_cache_dir,
                filename="config.yaml"
            )
            model_path = hf_hub_download(
                repo_id=repo_id,
                cache_dir=hf_cache_dir,
                filename="pytorch_model.bin"
            )
        vocoder = Vocos.from_hparams(config_path)
        state_dict = torch.load(
            model_path, map_location="cpu", weights_only=True
        )
        from vocos.feature_extractors import EncodecFeatures

        if isinstance(vocoder.feature_extractor, EncodecFeatures):
            encodec_parameters = {
                "feature_extractor.encodec." + key: value
                for key, value in
                vocoder.feature_extractor.encodec.state_dict().items()
            }
            state_dict.update(encodec_parameters)
        vocoder.load_state_dict(state_dict)
        vocoder = vocoder.eval().to(device)
    elif vocoder_name == "bigvgan":
        if is_local:
            # download generator from https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x/tree/main
            vocoder = bigvgan.BigVGAN.from_pretrained(
                local_path, use_cuda_kernel=False
            )
        else:
            vocoder = bigvgan.BigVGAN.from_pretrained(
                "nvidia/bigvgan_v2_24khz_100band_256x",
                use_cuda_kernel=False,
                cache_dir=hf_cache_dir
            )

        vocoder.remove_weight_norm()
        vocoder = vocoder.eval().to(device)
    return vocoder
