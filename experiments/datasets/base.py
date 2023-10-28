import abc

import numpy as np
from torch.utils import data
from torch.utils.data import DataLoader, random_split


class Dataset(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def train_data(self) -> data.Dataset:
        """Return the entire train data for this dataset"""
        raise NotImplemented()

    @abc.abstractmethod
    def test_data(self) -> data.Dataset:
        """Return the entire test data for this dataset"""
        raise NotImplemented()


def create_dataloader(data, batch_size: int):
    return DataLoader(data, batch_size=batch_size, shuffle=True)


def split_dataset(dataset, n):
    """Split a dataset into n parts of equal length"""
    return random_split(dataset=dataset, lengths=np.repeat(int(len(dataset) / n), n))
