import torch
from torch.nn.functional import one_hot
from torch.utils import data
from torchvision import datasets
from torchvision.transforms import ToTensor, transforms

from experiments.datasets.base import Dataset


class CIFAR10(Dataset):
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def train_data(self) -> data.Dataset:
        return datasets.CIFAR10(
            root=self.save_dir,
            train=True,
            transform=self.transforms,
            target_transform=lambda y: one_hot(torch.tensor(y), 10).type(torch.FloatTensor),
            download=True,
        )

    def test_data(self) -> data.Dataset:
        return datasets.CIFAR10(
            root=self.save_dir,
            train=False,
            transform=self.transforms,
            target_transform=lambda y: one_hot(torch.tensor(y), 10).type(torch.FloatTensor),
            download=True
        )

    def get_name(self) -> str:
        return "CIFAR10"

    def num_classes(self):
        return 10
