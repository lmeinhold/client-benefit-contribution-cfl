from pathlib import Path

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.nn.functional import one_hot
from torch.optim import SGD
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor

from federated_learning.fedavg import FedAvg
from utils.metrics_logging import Logger, JsonAdapter
from utils.torchutils import get_device
from models.mnist import CNN

BATCH_SIZE = 64
N_CLIENTS = 100
LR = 2e-3
EPOCHS = 5
ROUNDS = 30
ALPHA = 0.8


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


def create_optimizer(params, lr):
    return SGD(params, lr)


def main():
    logger = Logger(algorithm="FedAvg", dataset="MNIST", model="CNN", rounds=ROUNDS, epochs=EPOCHS)
    logger_adapter = JsonAdapter(Path("../output"), logger.run_id)
    logger.attach(logger_adapter)
    logger.log_run_data()

    train_data, test_data = load_data()
    client_data_loaders = [create_dataloader(d) for d in split_dataset(train_data, N_CLIENTS)]

    fa = FedAvg(
        client_data_loaders,
        model_fn=create_model,
        optimizer_fn=lambda p: create_optimizer(p, LR),
        loss_fn=CrossEntropyLoss(),
        rounds=ROUNDS,
        epochs=EPOCHS,
        alpha=ALPHA,
        logger=logger,
        device=get_device(),
        test_data=create_dataloader(test_data)
    )

    fa.fit()


if __name__ == "__main__":
    main()
