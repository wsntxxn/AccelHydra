from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class F5Collate:
    def __call__(self, batch: dict) -> dict:
        mel_specs = [item["mel_spec"].squeeze(0) for item in batch]
        mel_lengths = torch.LongTensor([spec.shape[-1] for spec in mel_specs])
        max_mel_length = mel_lengths.amax()

        padded_mel_specs = []
        for spec in mel_specs:
            padding = (0, max_mel_length - spec.size(-1))
            padded_spec = F.pad(spec, padding, value=0)
            padded_mel_specs.append(padded_spec)

        mel_specs = torch.stack(padded_mel_specs)

        text = [item["text"] for item in batch]
        text_lengths = torch.LongTensor([len(item) for item in text])

        # return dict(
        #     mel=mel_specs,
        #     mel_lengths=mel_lengths,  # records for padding mask
        #     text=text,
        #     text_lengths=text_lengths,
        # )
        return dict(
            inp=mel_specs.permute(0, 2, 1),
            lens=mel_lengths,  # records for padding mask
            text=text,
            text_lengths=text_lengths,
        )


@dataclass
class F5CollateWithoutRename(F5Collate):
    def __call__(self, batch: dict) -> dict:
        output = super().__call__(batch)
        return {
            "mel_spec": output["inp"],
            "mel_spec_lengths": output["lens"],
            "text": output["text"],
            "text_lengths": output["text_lengths"]
        }
