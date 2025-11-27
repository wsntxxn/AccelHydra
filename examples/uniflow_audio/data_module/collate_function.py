from typing import Any
from dataclasses import dataclass, field

from accel_hydra.data.collate_function import PaddingCollate


@dataclass
class PaddingCollateWithAnyContent(PaddingCollate):

    content_pad_keys: list[str] = field(default_factory=list)
    content_torchify_keys: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.content_collate_fn = PaddingCollate(
            self.content_pad_keys, self.content_torchify_keys
        )

    def __call__(self, batch):
        batch = super().__call__(batch)
        content = batch["content"]
        if isinstance(content[0], dict):
            content = self.content_collate_fn(content)
        elif isinstance(content[0],
                        torch.Tensor) or isinstance(content[0], np.ndarray):
            content = [torch.as_tensor(d) for d in content]
            padded_content = nn.utils.rnn.pad_sequence(
                content, batch_first=True
            )
            content_lengths = torch.as_tensor(
                [len(d) for d in content],
                dtype=torch.int32,
            )
            content = {
                "content": padded_content,
                "content_lengths": content_lengths,
            }
        batch.update({"content": content})
        return batch
