import torch
from torch.nn.functional import one_hot
from torch.utils import data
from torchvision import datasets
from torchvision.transforms import ToTensor

from experiments.datasets.base import Dataset


class EMNIST(Dataset):
    """Extended MNIST Dataset"""
    SPLIT = "balanced"
    N_CLASSES = 47

    def __init__(self, save_dir: str):
        self.save_dir = save_dir

    def train_data(self) -> data.Dataset:
        return datasets.EMNIST(
            root=self.save_dir,
            train=True,
            transform=ToTensor(),
            target_transform=lambda y: one_hot(torch.tensor(y), self.num_classes()).type(torch.FloatTensor),
            download=True,
            split=self.SPLIT
        )

    def test_data(self) -> data.Dataset:
        return datasets.EMNIST(
            root=self.save_dir,
            train=False,
            transform=ToTensor(),
            target_transform=lambda y: one_hot(torch.tensor(y), self.num_classes()).type(torch.FloatTensor),
            download=True,
            split=self.SPLIT
        )

    def get_name(self) -> str:
        return "EMNIST"

    def num_classes(self):
        return self.N_CLASSES
