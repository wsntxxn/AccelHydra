import numpy as np
import unittest

from accel_hydra.data_module.sampler import BucketSampler


class DummyDataset:
    """Dummy dataset for testing BucketSampler"""
    def __init__(self, durations):
        self.durations = durations

    def __len__(self):
        return len(self.durations)


class TestBucketSampler(unittest.TestCase):
    """Test cases for BucketSampler"""
    def test_basic_bucketing(self):
        """Test that samples are correctly assigned to buckets"""
        # Create dataset with durations from 0 to 21
        durations = np.arange(22)
        dataset = DummyDataset(durations)

        sampler = BucketSampler(
            dataset,
            num_buckets=3,
            batch_size=3,
            drop_last=False,
            shuffle=False
        )

        # Check that all buckets are populated
        self.assertEqual(len(sampler.buckets), 3)
        self.assertEqual(sum(len(bucket) for bucket in sampler.buckets), 22)

        # Check that samples are distributed across buckets
        bucket_sizes = [len(bucket) for bucket in sampler.buckets]
        self.assertEqual(bucket_sizes, [8, 7, 7])

    def test_drop_last(self):
        """Test drop_last functionality"""
        durations = np.arange(22)
        dataset = DummyDataset(durations)

        # With drop_last=True, incomplete batches should be dropped
        sampler_drop = BucketSampler(
            dataset,
            num_buckets=3,
            batch_size=3,
            drop_last=True,
            shuffle=False
        )

        # With drop_last=False, incomplete batches should be included
        sampler_no_drop = BucketSampler(
            dataset,
            num_buckets=3,
            batch_size=3,
            drop_last=False,
            shuffle=False
        )

        # drop_last should have fewer or equal samples
        self.assertLessEqual(len(sampler_drop), len(sampler_no_drop))

        # Verify that all samples in drop_last are complete batches
        samples_drop = list(sampler_drop)
        self.assertEqual(len(samples_drop), 21)

        samples_no_drop = list(sampler_no_drop)
        self.assertEqual(len(samples_no_drop), 22)


if __name__ == "__main__":
    unittest.main()
