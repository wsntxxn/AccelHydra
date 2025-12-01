from pathlib import Path
import os

import soundfile as sf
import torch
import torchaudio
import hydra
from accelerate import Accelerator

from omegaconf import OmegaConf
from safetensors.torch import load_file
from tqdm import tqdm
from accel_hydra.models.common import LoadPretrainedBase

from utils.config import register_omegaconf_resolvers
from utils.vocoder import load_vocoder
from utils.tokenize import convert_char_to_pinyin
from utils.audio import MelSpec

try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except:
    pass

register_omegaconf_resolvers()


def main():

    accelerator = Accelerator(mixed_precision="no")
    configs = []

    @hydra.main(config_path="configs", config_name="inference")
    def parse_config_from_command_line(config):
        config = OmegaConf.to_container(config, resolve=True)
        configs.append(config)

    parse_config_from_command_line()
    config = configs[0]

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

    model: LoadPretrainedBase = hydra.utils.instantiate(exp_config["model"])
    state_dict = load_file(ckpt_path)
    model.load_pretrained(state_dict)
    model.eval()

    # if "sampler" in config["test_dataloader"]:
    #     data_source = hydra.utils.instantiate(
    #         config["test_dataloader"]["dataset"], _convert_="all"
    #     )
    #     sampler = hydra.utils.instantiate(
    #         config["test_dataloader"]["sampler"],
    #         data_source=data_source,
    #         _convert_="all"
    #     )
    #     test_dataloader = hydra.utils.instantiate(
    #         config["test_dataloader"], sampler=sampler, _convert_="all"
    #     )
    # else:
    #     test_dataloader = hydra.utils.instantiate(
    #         config["test_dataloader"], _convert_="all"
    #     )

    # model, test_dataloader = accelerator.prepare(model, test_dataloader)

    model = accelerator.prepare(model)
    vocoder = load_vocoder(device=accelerator.device, **config["vocoder"])

    if config["wav_dir_root"] is not None:
        wav_dir_root = Path(config["wav_dir_root"])
    else:
        wav_dir_root = exp_dir
    audio_output_dir = wav_dir_root / config["wav_dir"]
    if accelerator.is_main_process:
        audio_output_dir.mkdir(parents=True, exist_ok=True)
    unwrapped_model = accelerator.unwrap_model(model)
    # pbar_disable = not accelerator.is_main_process

    mel_spectrogram_fn = MelSpec(**exp_config["model"]["mel_spec_kwargs"])
    mel_spec_type = exp_config["model"]["mel_spec_kwargs"]["mel_spec_type"]

    # Prompt speech
    target_sample_rate = exp_config["model"]["mel_spec_kwargs"][
        "target_sample_rate"]
    target_rms = config["target_rms"]
    ref_audio, ref_sr = torchaudio.load(config["prompt_speech"])
    ref_rms = torch.sqrt(torch.mean(torch.square(ref_audio)))
    if ref_rms < target_rms:
        ref_audio = ref_audio * target_rms / ref_rms
    assert ref_audio.shape[
        -1
    ] > 5000, f"Empty prompt wav: {config['prompt_speech']}, or torchaudio backend issue."
    if ref_sr != target_sample_rate:
        ref_audio = torchaudio.functional.resample(
            ref_audio, ref_sr, target_sample_rate
        )

    # Text
    gen_text = config["text"]
    prompt_text = config["prompt_text"]
    if prompt_text.encode("utf-8") == 1:
        prompt_text = prompt_text + " "
    text = [prompt_text + gen_text]
    tokenizer = exp_config["model"]["tokenizer"]
    if tokenizer == "pinyin":
        text_list = convert_char_to_pinyin(text, polyphone=True)
    else:
        text_list = text

    # to mel spectrogram
    ref_mel = mel_spectrogram_fn(ref_audio)
    ref_mel = ref_mel.squeeze(0)
    ref_mel = ref_mel.to(accelerator.device)

    # Duration, mel frame length
    ref_mel_len = ref_mel.shape[-1]
    ref_text_len = len(prompt_text.encode("utf-8"))
    gen_text_len = len(gen_text.encode("utf-8"))
    total_mel_len = ref_mel_len + int(
        ref_mel_len / ref_text_len * gen_text_len
    )

    with torch.no_grad():
        mel_spec, _ = unwrapped_model.sample(
            cond=ref_mel.unsqueeze(0).permute(0, 2, 1),
            text=text_list,
            duration=torch.as_tensor([total_mel_len]).to(accelerator.device),
            lens=torch.as_tensor([ref_mel_len]).to(accelerator.device),
            **config["infer_args"],
        )

        mel_spec = mel_spec[0]
        mel_spec = mel_spec[ref_mel_len:total_mel_len, :].unsqueeze(0)
        mel_spec = mel_spec.permute(0, 2, 1).to(torch.float32)
        if mel_spec_type == "vocos":
            waveform = vocoder.decode(mel_spec).cpu()
        elif mel_spec_type == "bigvgan":
            waveform = vocoder(mel_spec).squeeze(0).cpu()

        if ref_rms < target_rms:
            waveform = waveform * ref_rms / target_rms

        torchaudio.save(
            audio_output_dir / config["output_fname"], waveform,
            target_sample_rate
        )

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
