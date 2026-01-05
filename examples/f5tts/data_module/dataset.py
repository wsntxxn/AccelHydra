import json
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from datasets import Dataset
from accel_hydra.utils.general import read_jsonl_to_mapping

from utils.general import default
from utils.audio import MelSpec


@dataclass(kw_only=True)
class AudioLoadingMixin:
    def load_audio(
        self, audio_path: str, target_sample_rate: int
    ) -> torch.Tensor:
        audio, source_sample_rate = torchaudio.load(audio_path)
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        if source_sample_rate != target_sample_rate:
            audio = torchaudio.functional.resample(
                audio, source_sample_rate, target_sample_rate
            )
        return audio


@dataclass(kw_only=True)
class CustomDataset(AudioLoadingMixin):

    hf_data_raw_path: str
    duration_path: str
    min_duration: float = field(default=0.3)
    max_duration: float = field(default=30.0)
    target_sample_rate: int = field(default=24_000)

    hop_length: int = field(default=256)
    n_mel_channels: int = field(default=100)
    n_fft: int = field(default=1024)
    win_length: int = field(default=1024)
    mel_spec_type: str = field(default="vocos")
    preprocessed_mel: bool = field(default=False)
    mel_spec_module: nn.Module | None = field(default=None)
    max_samples: int | None = field(default=None)

    def __post_init__(self):
        self.data = Dataset.from_file(self.hf_data_raw_path)
        if self.max_samples is not None:
            slice_indexes = np.random.randint(
                0, len(self.data), self.max_samples
            )
            self.data = self.data.select(slice_indexes)
        with open(self.duration_path, "r", encoding="utf-8") as f:
            self.durations = json.load(f)["duration"]

        if not self.preprocessed_mel:
            self.mel_spectrogram = default(
                self.mel_spec_module,
                MelSpec(
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    win_length=self.win_length,
                    n_mel_channels=self.n_mel_channels,
                    target_sample_rate=self.target_sample_rate,
                    mel_spec_type=self.mel_spec_type,
                ),
            )

    def get_length(self, index):
        if (self.durations is not None):
            # Please make sure the separately provided durations are correct, otherwise 99.99% OOM
            return self.durations[index
                                 ] * self.target_sample_rate / self.hop_length
        return self.data[index]["duration"
                               ] * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        while True:
            row = self.data[index]
            audio_path = row["audio_path"]
            text = row["text"]
            duration = row["duration"]

            # filter by given length
            if self.min_duration <= duration <= self.max_duration:
                break  # valid

            index = (index + 1) % len(self.data)

        if self.preprocessed_mel:
            mel_spec = torch.tensor(row["mel_spec"])
        else:
            read_success = False
            while not read_success:
                try:
                    audio = self.load_audio(
                        audio_path, self.target_sample_rate
                    )
                    read_success = True
                except Exception as e:
                    print(f"Error loading audio from {audio_path}: {e}")
                    index = (index + 1) % len(self.data)
                    row = self.data[index]
                    audio_path = row["audio_path"]
                    text = row["text"]

            # to mel spectrogram
            mel_spec = self.mel_spectrogram(audio)
            mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t'

        return {
            "mel_spec": mel_spec,
            "text": text,
        }


@dataclass(kw_only=True)
class CrossSentenceTTSDataset(torch.utils.data.Dataset, AudioLoadingMixin):

    cross_sentence_meta: str
    audio_mapping: str

    target_sample_rate: int = field(default=24_000)
    hop_length: int = field(default=256)
    n_mel_channels: int = field(default=100)
    n_fft: int = field(default=1024)
    win_length: int = field(default=1024)
    mel_spec_type: str = field(default="vocos")
    preprocessed_mel: bool = field(default=False)
    mel_spec_module: nn.Module | None = field(default=None)

    def __post_init__(self):
        self.aid_to_audio = read_jsonl_to_mapping(
            self.audio_mapping, "audio_id", "audio"
        )
        self.data = []
        with open(self.cross_sentence_meta, "r") as reader:
            for line in reader.readlines():
                prompt_audio_id, prompt_duration, prompt_text, \
                    audio_id, duration, text = line.strip().split("\t")
                self.data.append({
                    "prompt_audio_id": prompt_audio_id,
                    "audio_id": audio_id,
                    "prompt_text": prompt_text,
                    "text": text,
                })

        self.mel_spectrogram = default(
            self.mel_spec_module,
            MelSpec(
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_mel_channels=self.n_mel_channels,
                target_sample_rate=self.target_sample_rate,
                mel_spec_type=self.mel_spec_type,
            ),
        )

    def __getitem__(self, index):
        item = self.data[index]
        prompt_audio_id = item["prompt_audio_id"]
        audio_id = item["audio_id"]
        prompt_text = item["prompt_text"]
        text = item["text"]

        audio_path = self.aid_to_audio[audio_id]
        audio = self.load_audio(audio_path, self.target_sample_rate)
        # to mel spectrogram
        mel_spec = self.mel_spectrogram(audio)
        mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t'

        prompt_audio_path = self.aid_to_audio[prompt_audio_id]
        prompt_audio = self.load_audio(
            prompt_audio_path, self.target_sample_rate
        )
        prompt_mel_spec = self.mel_spectrogram(prompt_audio)
        prompt_mel_spec = prompt_mel_spec.squeeze(0).transpose(
            0, 1
        )  # '1 d t -> d t'

        return {
            "audio_id": audio_id,
            "prompt_mel_spec": prompt_mel_spec,
            "mel_spec": mel_spec,
            "text": text,
            "prompt_text": prompt_text,
        }

    def __len__(self):
        return len(self.data)
