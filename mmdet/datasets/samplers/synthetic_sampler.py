import itertools
import math
from typing import Iterator, Optional, Sized

import torch
from torch.utils.data import Sampler

from mmengine.dist import get_dist_info, sync_random_seed
from mmengine.registry import DATA_SAMPLERS

@DATA_SAMPLERS.register_module()
class SyntheticDataSampler(Sampler):
    """A sampler for synthetic data.

    Args:
        dataset (Sized): The dataset.
        shuffle (bool): Whether shuffle the dataset or not. Defaults to True.
        seed (int, optional): Random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Defaults to None.
        round_up (bool): Whether to add extra samples to make the number of
            samples evenly divisible by the world size. Defaults to True.
    """

    def __init__(self,
                 dataset: Sized,
                 shuffle: bool = True,
                 seed: int = None,
                 synthetic_ratio: float = 0.2,
                 round_up: bool = True) -> None:
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        self.shuffle = shuffle
        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
        self.epoch = 0
        self.round_up = round_up
        self.synthetic_ratio = synthetic_ratio
        
        self.real_data_inds = []
        self.syn_data_inds = []
        for idx in range(len(self.dataset)):
            data_info = self.dataset.get_data_info(idx)
            if not data_info.get('is_syn', False):
                self.real_data_inds.append(idx)
            else:
                self.syn_data_inds.append(idx)

        if self.round_up:
            self.num_samples = math.ceil(len(self.real_data_inds) / world_size)
            self.total_size = self.num_samples * self.world_size
        else:
            self.num_samples = math.ceil(
                (len(self.real_data_inds) - rank) / world_size)
            self.total_size = len(self.real_data_inds)

    def __iter__(self) -> Iterator[int]:
        """Iterate the indices."""
        # deterministically shuffle based on epoch and seed
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            num_syn_samples = int(self.synthetic_ratio * len(self.real_data_inds))
            num_real_samples = self.total_size - num_syn_samples

            syn_indices = torch.randperm(len(self.syn_data_inds), generator=g)[:num_syn_samples]
            real_indices = torch.randperm(len(self.real_data_inds), generator=g)[:num_real_samples]
            
            real_syn_indices = [self.real_data_inds[i] for i in real_indices] + [self.syn_data_inds[i] for i in syn_indices]
            perm = torch.randperm(len(real_syn_indices), generator=g).tolist()
            indices = [real_syn_indices[i] for i in perm]
        else:
            indices = torch.arange(len(self.real_data_inds)).tolist()

        # add extra samples to make it evenly divisible
        if self.round_up:
            indices = (
                indices *
                int(self.total_size / len(indices) + 1))[:self.total_size]

        # subsample
        indices = indices[self.rank:self.total_size:self.world_size]

        return iter(indices)

    def __len__(self) -> int:
        """The number of samples in this rank."""
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas use a different
        random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch