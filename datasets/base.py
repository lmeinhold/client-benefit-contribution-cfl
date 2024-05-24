import abc

import numpy as np
from torch.utils import data
from torch.utils.data import DataLoader, random_split


class Dataset(metaclass=abc.ABCMeta):
    """Common base class for all datasets"""

    @abc.abstractmethod
    def train_data(self) -> data.Dataset:
        """Return the entire train data for this dataset"""
        raise NotImplemented()

    @abc.abstractmethod
    def test_data(self) -> data.Dataset:
        """Return the entire test data for this dataset"""
        raise NotImplemented()

    @abc.abstractmethod
    def get_name(self) -> str:
        """Returns a short name for the dataset, e.q. 'MNIST'"""
        raise NotImplemented()

    @abc.abstractmethod
    def num_classes(self):
        """Return the number of different class labels that the dataset contains"""
        raise NotImplemented()


def create_dataloader(data: data.Dataset, batch_size: int) -> data.DataLoader:
    """
    Create a dataloader from the given dataset with shuffling and memory pinning

        Parameters:
            data (Dataset): the dataset to create the dataloader from
            batch_size (int): the batch size to use for the dataloader

        Returns:
            a dataloader
    """
    return DataLoader(data, batch_size=batch_size, shuffle=True, pin_memory=True)


def split_dataset(dataset, n) -> list[data.Dataset]:
    """
    Split a dataset into n parts of equal length randomly

        Parameters:
            dataset (Dataset): dataset to split
            n (int): number of new subsets

        Returns:
            a list of subsets
    """
    return random_split(dataset=dataset, lengths=np.repeat(int(len(dataset) / n), n))


class CachingDataset(data.Dataset):
    """
    A wrapper for torch Datasets that caches transformations.
    Avoids recomputing transformations every epoch"""

    def __init__(self, source_dataset):
        self.source_dataset = source_dataset
        self.cache = {}
        self.targets = source_dataset.targets
        self.data = source_dataset.data

    def __len__(self):
        return self.source_dataset.__len__()

    def __getitem__(self, index):
        if index not in self.cache:
            self.cache[index] = self.source_dataset.__getitem__(index)
        return self.cache[index]
