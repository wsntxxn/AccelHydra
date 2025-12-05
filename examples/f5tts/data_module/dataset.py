import json
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from datasets import Dataset

from utils.general import default
from utils.audio import MelSpec


@dataclass
class CustomDataset(torch.utils.data.Dataset):

    # TODO minimum duration grouping transform
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
            audio, source_sample_rate = torchaudio.load(audio_path)

            # make sure mono input
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            # resample if necessary
            if source_sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(
                    source_sample_rate, self.target_sample_rate
                )
                audio = resampler(audio)

            # to mel spectrogram
            mel_spec = self.mel_spectrogram(audio)
            mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t'

        return {
            "mel_spec": mel_spec,
            "text": text,
        }


class MinimumDurationGroupingDataset(CustomDataset):

    group_duration: float = field(default=12.0)

    def __post_init__(self):
        super().__post_init__()
        groups = []
        group_durations = []
        current_group = []
        current_sum = 0.0

        for idx in range(len(self.durations)):
            duration = self.durations[idx]
            current_group.append(idx)
            current_sum += duration
            if current_sum >= self.group_duration:
                groups.append(current_group)
                current_group = []
                current_sum = 0.0
                group_durations.append(current_sum)

        if current_group:
            groups.append(current_group)
            group_durations.append(current_sum)

        self.durations = group_durations
        self.groups = groups
