import torch
from torch.nn.functional import one_hot
from torch.utils import data
from torchvision import datasets
from torchvision.transforms import ToTensor

from datasets.base import Dataset, CachingDataset


class MNIST(Dataset):
    """
    A wrapper for the MNIST dataset.
    http://yann.lecun.com/exdb/mnist/
    """

    def __init__(self, save_dir: str):
        self.save_dir = save_dir

    def train_data(self) -> data.Dataset:
        return CachingDataset(datasets.MNIST(
            root=self.save_dir,
            train=True,
            transform=ToTensor(),
            target_transform=lambda y: one_hot(torch.tensor(y), 10).type(torch.FloatTensor),
            download=True,
        ))

    def test_data(self) -> data.Dataset:
        return CachingDataset(datasets.MNIST(
            root=self.save_dir,
            train=False,
            transform=ToTensor(),
            target_transform=lambda y: one_hot(torch.tensor(y), 10).type(torch.FloatTensor),
            download=True
        ))

    def get_name(self) -> str:
        return "MNIST"

    def num_classes(self):
        return 10
