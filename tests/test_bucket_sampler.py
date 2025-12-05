import numpy as np

from accel_hydra.data_module.sampler import BucketSampler


class DummyDataset:
    def __init__(self, durations) -> None:
        self.durations = durations


data_source = DummyDataset(durations=np.arange(22))
sampler = BucketSampler(
    data_source, num_buckets=3, batch_size=3, drop_last=True, shuffle=True
)
print(list(sampler))
