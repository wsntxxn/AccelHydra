from typing import Any
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn


@dataclass
class PaddingCollate:
    pad_keys: list[str] = field(default_factory=list)
    torchify_keys: list[str] = field(default_factory=list)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        collate_samples: dict[str, list[Any]] = {
            k: [dic[k] for dic in batch]
            for k in batch[0]
        }
        batch_keys = list(collate_samples.keys())

        for key in batch_keys:
            if key in self.pad_keys:
                torchified_batch = [
                    torch.as_tensor(d) for d in collate_samples[key]
                ]
                data_batch = nn.utils.rnn.pad_sequence(
                    torchified_batch, batch_first=True
                )
                data_lengths = torch.as_tensor(
                    [len(d) for d in torchified_batch],
                    dtype=torch.int32,
                )

                collate_samples.update({
                    key: data_batch,
                    f"{key}_lengths": data_lengths
                })
            elif key in self.torchify_keys:
                if isinstance(collate_samples[key][0], np.ndarray):
                    collate_samples[key] = np.array(collate_samples[key])
                collate_samples[key] = torch.as_tensor(collate_samples[key])

        return collate_samples
