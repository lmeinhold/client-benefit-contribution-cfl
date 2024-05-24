import torch
from torch.nn.functional import one_hot
from torch.utils import data
from torchvision import datasets
from torchvision.transforms import transforms

from datasets.base import Dataset, CachingDataset


def _transform_onehot(a):
    return one_hot(torch.as_tensor(a), num_classes=10).type(torch.FloatTensor)


class CIFAR10(Dataset):
    """
    A wrapper for the CIFAR10 dataset.
    https://www.cs.toronto.edu/~kriz/cifar.html
    """

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def train_data(self) -> data.Dataset:
        return CachingDataset(datasets.CIFAR10(
            root=self.save_dir,
            train=True,
            transform=self.transforms,
            target_transform=_transform_onehot,
            download=True,
        ))

    def test_data(self) -> data.Dataset:
        return CachingDataset(datasets.CIFAR10(
            root=self.save_dir,
            train=False,
            transform=self.transforms,
            target_transform=_transform_onehot,
            download=True
        ))

    def get_name(self) -> str:
        return "CIFAR10"

    def num_classes(self):
        return 10
