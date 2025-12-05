from torch.utils.data import Sampler, BatchSampler, Dataset
from tqdm import trange
import torch


class DataSourceGetLengthMixin(Dataset):
    def get_length(self, idx: int) -> float | int:
        raise NotImplementedError("Subclasses must implement this method")


class DynamicBatchSampler(BatchSampler):
    """Base class for dynamic batching based on frame length threshold.
    
    Creates batches dynamically to ensure total frames per batch does not exceed
    the threshold, improving padding efficiency.

    Args:
        data_source (Dataset): The dataset to sample from. 
            Must implement a `get_length(idx)` method returning the length of a data point.
        batch_length_threshold (int): The maximum total length per batch.
        max_samples (int, optional): The maximum number of samples per batch, 0 means no limit. Default is 0.
        random_seed (int, optional): Seed for randomization. Default is None.
        drop_last (bool, optional): If True, drops the last batch if it's smaller than the threshold. Default is False.
    """
    def __init__(
        self,
        data_source: DataSourceGetLengthMixin,
        batch_length_threshold: int,
        max_samples: int = 0,
        random_seed: int = None,
        drop_last: bool = False
    ):
        self.batch_length_threshold = batch_length_threshold
        self.max_samples = max_samples
        self.random_seed = random_seed
        self.epoch = 0
        self.data_source = data_source

        batches = []
        batch = []
        batch_length = 0

        indices_with_lengths = self.get_indices_with_lengths()

        for idx, sample_length in indices_with_lengths:
            if batch_length + sample_length <= self.batch_length_threshold and (
                max_samples == 0 or len(batch) < max_samples
            ):
                batch.append(idx)
                batch_length += sample_length
            else:
                if len(batch) > 0:
                    batches.append(batch)
                if sample_length <= self.batch_length_threshold:
                    batch = [idx]
                    batch_length = sample_length
                else:
                    batch = []
                    batch_length = 0

        if not drop_last and len(batch) > 0:
            batches.append(batch)

        self.batches = batches
        self.drop_last = drop_last

    def get_indices_with_lengths(self, ) -> list[tuple[int, float | int]]:
        result = []
        for idx in trange(len(self.data_source)):
            result.append((idx, self.data_source.get_length(idx)))
        return result

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler."""
        self.epoch = epoch

    def __iter__(self):
        if self.random_seed is not None:
            g = torch.Generator()
            g.manual_seed(self.random_seed + self.epoch)
            indices = torch.randperm(len(self.batches), generator=g).tolist()
            batches = [self.batches[i] for i in indices]
        else:
            batches = self.batches
        return iter(batches)

    def __len__(self):
        return len(self.batches)


class SortedDynamicBatchSampler(DynamicBatchSampler):
    """Dynamic batch sampler with length-based sorting.
    
    First sorts samples by frame length, then creates dynamic batches.
    This improves padding efficiency by grouping similar-length samples together.
    """
    def get_indices_with_lengths(self) -> list[tuple[int, float | int]]:
        return sorted(
            super().get_indices_with_lengths(), key=lambda elem: elem[1]
        )
