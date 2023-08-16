import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.nn.functional import one_hot
from torch.optim import SGD
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor

from src.federated_learning.fedavg import FedAvg
from src.federated_learning.torchutils import get_device
from src.models.mnist import CNN

BATCH_SIZE = 32
N_CLIENTS = 100
LR = 2e-3
EPOCHS = 5
ROUNDS = 50
ALPHA = 0.6


def load_data():
    train_data = datasets.MNIST(
        root='../../data',
        train=True,
        transform=ToTensor(),
        target_transform=lambda y: one_hot(torch.tensor(y), 10).type(torch.FloatTensor),
        download=True,
    )
    test_data = datasets.MNIST(
        root='../../data',
        train=False,
        transform=ToTensor(),
        target_transform=lambda y: one_hot(torch.tensor(y), 10).type(torch.FloatTensor),
        download=True
    )

    return train_data, test_data


def create_dataloader(data):
    return DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)


def split_dataset(dataset, n):
    """Split a dataset into n parts of equal length"""
    return random_split(dataset=dataset, lengths=np.repeat(int(len(dataset) / n), n))


def create_model():
    return CNN()


def create_optimizer(params):
    return SGD(params, LR)


def main():
    train_data, test_data = load_data()
    client_data_loaders = [create_dataloader(d) for d in split_dataset(train_data, N_CLIENTS)]

    fa = FedAvg(
        client_data_loaders,
        model_fn=create_model,
        optimizer_fn=create_optimizer,
        loss_fn=CrossEntropyLoss(),
        rounds=ROUNDS,
        epochs=EPOCHS,
        alpha=ALPHA,
        device=get_device(),
        test_data=create_dataloader(test_data)
    )

    fa.fit()


if __name__ == "__main__":
    main()
