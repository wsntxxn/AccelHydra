import numpy as np
import torch
from torch.utils.data import Sampler


class BucketSampler(Sampler[int]):
    """
    A sampler that groups samples into buckets based on their duration values.
    Each batch is sampled from the same bucket, ensuring that samples within
    a batch have similar lengths, which facilitates efficient batch processing.

    **Requirement:**
    The `data_source` (dataset) must have a `durations` attribute that is
    a sequence (list, array, etc.) where `data_source.durations[i]` corresponds
    to the duration of `data_source[i]`. This allows the sampler to access
    duration information for each sample by index.

    Args:
        data_source: The dataset object. Must have a `durations` attribute
            such that `data_source.durations[i]` gives the duration of
            `data_source[i]`.
        num_buckets: Number of buckets to divide samples into.
        batch_size: Size of each batch.
        shuffle: Whether to shuffle the order of samples within each bucket
            at the start of each epoch. Defaults to True.
        drop_last: Whether to drop the last incomplete batch from each bucket.
            Defaults to False.
        generator: Optional torch.Generator for reproducible shuffling.
            If None, a new generator will be created with a random seed.

    Example:
        >>> class MyDataset:
        ...     def __init__(self):
        ...         self.data = [...]
        ...         self.durations = [1.5, 2.3, 1.8, ...]  # duration for each sample
        ...     def __getitem__(self, idx):
        ...         return self.data[idx]
        ...     def __len__(self):
        ...         return len(self.data)
        >>> dataset = MyDataset()
        >>> sampler = BucketSampler(dataset, num_buckets=5, batch_size=32)
    """
    def __init__(
        self,
        data_source,
        num_buckets: int,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        generator=None,
    ):
        self.data_source = data_source
        self.num_buckets = num_buckets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator

        # Access durations from the dataset
        durations = np.array(data_source.durations)

        # Assign samples to buckets based on duration
        # Use percentile-based method to ensure roughly equal sample counts per bucket
        if num_buckets == 1:
            # If only one bucket, all samples go into this bucket
            bucket_assignments = np.zeros(len(durations), dtype=np.int32)
        else:
            # Use percentiles to divide duration range into num_buckets intervals
            percentiles = np.linspace(0, 100, num_buckets + 1)
            thresholds = np.percentile(durations, percentiles)
            # Assign each sample to its corresponding bucket
            bucket_assignments = np.digitize(
                durations, thresholds[1:], right=False
            )
            # digitize returns values from 0 to num_buckets, clip to [0, num_buckets-1]
            bucket_assignments = np.clip(
                bucket_assignments, 0, num_buckets - 1
            )

        # Group sample indices by bucket
        self.buckets = [[] for _ in range(num_buckets)]
        for idx, bucket_id in enumerate(bucket_assignments):
            self.buckets[bucket_id].append(idx)

        # Record the size of each bucket
        self.bucket_sizes = [len(bucket) for bucket in self.buckets]

        # Calculate total number of samples (accounting for drop_last)
        self.num_samples = 0
        if self.drop_last:
            # Count complete batches from each bucket, plus cross-bucket batches from remaining samples
            total_remaining = 0
            for bucket in self.buckets:
                # Complete batches from this bucket
                self.num_samples += len(bucket) // batch_size * batch_size
                # Remaining samples from this bucket
                total_remaining += len(bucket) % batch_size
            # Cross-bucket batches from remaining samples
            self.num_samples += total_remaining // batch_size * batch_size
        else:
            # Include all samples, including incomplete batches
            self.num_samples = sum(len(bucket) for bucket in self.buckets)

    def __iter__(self):
        # Initialize generator for reproducible shuffling
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        # Shuffle samples within each bucket at the start of each epoch
        if self.shuffle:
            buckets = [
                np.random.permutation(bucket).tolist()
                for bucket in self.buckets
            ]
        else:
            buckets = self.buckets

        # print(f'buckets: {buckets}')

        batches = []
        remaining_samples = []

        # Generate batches from each bucket
        for bucket in buckets:
            # Always take complete batches from each bucket first
            num_complete_batches = len(bucket) // self.batch_size

            for i in range(num_complete_batches):
                start_idx = i * self.batch_size
                end_idx = start_idx + self.batch_size
                batches.append(bucket[start_idx:end_idx])

            # Collect remaining samples from this bucket
            remaining_start = num_complete_batches * self.batch_size
            if remaining_start < len(bucket):
                remaining_samples.extend(bucket[remaining_start:])

        # Form batches from remaining samples across buckets
        if remaining_samples:
            if self.drop_last:
                # Only form complete batches from remaining samples
                num_remaining_batches = len(
                    remaining_samples
                ) // self.batch_size
                for i in range(num_remaining_batches):
                    start_idx = i * self.batch_size
                    end_idx = start_idx + self.batch_size
                    batches.append(remaining_samples[start_idx:end_idx])
            else:
                # Include the last incomplete batch
                num_remaining_batches = (
                    len(remaining_samples) + self.batch_size - 1
                ) // self.batch_size
                for i in range(num_remaining_batches):
                    start_idx = i * self.batch_size
                    end_idx = start_idx + self.batch_size
                    batches.append(remaining_samples[start_idx:end_idx])
        # print(f'batches: {batches}')

        # Shuffle batch order if enabled
        if self.shuffle:
            batch_ids = torch.randperm(len(batches),
                                       generator=generator).tolist()
        else:
            batch_ids = list(range(len(batches)))

        # print(f'batch_ids: {batch_ids}')

        # Yield sample indices in batch order
        for batch_id in batch_ids:
            batch = batches[batch_id]
            for idx in batch:
                yield idx

    def __len__(self):
        return self.num_samples
