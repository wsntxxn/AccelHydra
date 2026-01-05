from pathlib import Path
import os

import torch
import torchaudio
import hydra
from accelerate import Accelerator

from omegaconf import OmegaConf
from safetensors.torch import load_file
from tqdm import tqdm
from accel_hydra.models.common import LoadPretrainedBase
from accel_hydra.utils import load_config_from_cli

from utils.config import register_omegaconf_resolvers
from utils.vocoder import load_vocoder

try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except:
    pass


def main():

    accelerator = Accelerator(mixed_precision="no")
    config = load_config_from_cli(
        register_resolver_fn=register_omegaconf_resolvers
    )

    exp_dir, ckpt_dir = None, None
    if os.path.isdir(config["exp_dir"]):
        exp_dir = Path(config["exp_dir"])
    if os.path.isdir(config["ckpt_dir_or_file"]):
        ckpt_dir = Path(config["ckpt_dir_or_file"])
        ckpt_path = ckpt_dir / "model.safetensors"
    elif os.path.isfile(config["ckpt_dir_or_file"]):
        ckpt_path = config["ckpt_dir_or_file"]
        ckpt_dir = Path(ckpt_path).parent

    if ckpt_dir is None and exp_dir is None:
        if not os.path.exists(config["ckpt_dir_or_file"]):
            raise ValueError(
                f"ckpt_dir {config['ckpt_dir_or_file']} does not exist."
            )
        raise ValueError(
            "Either exp_dir or ckpt_dir_or_file should be provided."
        )

    if ckpt_dir is None:
        use_best = config.get("use_best", True)
        if use_best:  # use best ckpt
            ckpt_path: Path = exp_dir / "checkpoints/best/model.safetensors"
        else:  # use last ckpt
            ckpt_path: Path = sorted((exp_dir / "checkpoints").iterdir()
                                    )[-1] / "model.safetensors"
    if exp_dir is None:
        exp_dir = ckpt_dir.parent.parent

    if "exp_config_path" in config:
        exp_config_path = config['exp_config_path']
        exp_config = OmegaConf.load(config['exp_config_path'])
    else:
        exp_config_path = exp_dir / "config.yaml"

    exp_config = OmegaConf.load(exp_config_path)
    accelerator.print(
        f'\n ckpt path: {ckpt_path}, model config path: {exp_config_path}\n '
    )

    model: LoadPretrainedBase = hydra.utils.instantiate(
        exp_config["model"], _convert_="all"
    )
    state_dict = load_file(ckpt_path)
    model.load_pretrained(state_dict)
    model.eval()

    mel_spec_kwargs = {
        "target_sample_rate":
            exp_config["train_dataloader"]["dataset"]["target_sample_rate"],
        "n_mel_channels":
            exp_config["train_dataloader"]["dataset"]["n_mel_channels"],
        "hop_length":
            exp_config["train_dataloader"]["dataset"]["hop_length"],
        "win_length":
            exp_config["train_dataloader"]["dataset"]["win_length"],
        "n_fft":
            exp_config["train_dataloader"]["dataset"]["n_fft"],
        "mel_spec_type":
            exp_config["train_dataloader"]["dataset"]["mel_spec_type"],
    }
    config["test_dataloader"]["dataset"].update(mel_spec_kwargs)
    test_dataloader = hydra.utils.instantiate(
        config["test_dataloader"], _convert_="all"
    )

    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    vocoder = load_vocoder(device=accelerator.device, **config["vocoder"])

    if config["wav_dir_root"] is not None:
        wav_dir_root = Path(config["wav_dir_root"])
    else:
        wav_dir_root = exp_dir
    audio_output_dir = wav_dir_root / config["wav_dir"]
    if accelerator.is_main_process:
        audio_output_dir.mkdir(parents=True, exist_ok=True)
    unwrapped_model = accelerator.unwrap_model(model)
    pbar_disable = not accelerator.is_main_process

    with torch.no_grad():
        for batch in tqdm(test_dataloader, disable=pbar_disable):
            # Text
            gen_text = batch["text"][0]
            prompt_text = batch["prompt_text"][0]
            if prompt_text.encode("utf-8") == 1:
                prompt_text = prompt_text + " "
            text = [prompt_text + gen_text]

            ref_mel = batch["prompt_mel_spec"][0]
            # Duration, mel frame length
            ref_mel_len = ref_mel.shape[0]
            ref_text_len = len(prompt_text.encode("utf-8"))
            gen_text_len = len(gen_text.encode("utf-8"))
            gen_mel_len = int(ref_mel_len / ref_text_len * gen_text_len)
            total_mel_len = ref_mel_len + gen_mel_len

            result = unwrapped_model.sample(
                cond=ref_mel.unsqueeze(0),
                text=text,
                duration=torch.as_tensor([total_mel_len]).to(
                    accelerator.device
                ),
                lens=torch.as_tensor([ref_mel_len]).to(accelerator.device),
                **config["infer_args"],
            )
            mel_spec = result[0]

            mel_spec = mel_spec[0]
            mel_spec = mel_spec[ref_mel_len:total_mel_len, :].unsqueeze(0)
            mel_spec = mel_spec.permute(0, 2, 1).to(torch.float32)
            if mel_spec_kwargs["mel_spec_type"] == "vocos":
                waveform = vocoder.decode(mel_spec).cpu()
            elif mel_spec_kwargs["mel_spec_type"] == "bigvgan":
                waveform = vocoder(mel_spec).squeeze(0).cpu()

            torchaudio.save(
                audio_output_dir / f"{batch['audio_id'][0]}.wav", waveform,
                mel_spec_kwargs["target_sample_rate"]
            )

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
