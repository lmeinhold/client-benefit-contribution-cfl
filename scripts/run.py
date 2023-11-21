from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from experiments.datasets.base import create_dataloader
from experiments.datasets.cifar import CIFAR10
from experiments.datasets.imbalancing import split_with_label_distribution_skew
from federated_learning.fedprox_new import FedProx
from models.cifar import CNN
from utils.torchutils import get_device

N_CLIENTS = 100
GAMMA = 1
BATCH_SIZE = 32
LR = 2e-3
EPOCHS = 5
ROUNDS = 5
ALPHA = 0.9
MU = 0
MODEL = CNN


def create_optimizer(params):
    return SGD(params, LR)


if __name__ == "__main__":
    dataset = CIFAR10("../data").train_data()
    client_dataloaders = [create_dataloader(ds, BATCH_SIZE) for ds in
                          split_with_label_distribution_skew(dataset, GAMMA, N_CLIENTS)]

    fedprox = FedProx(
        model_class=MODEL,
        loss=CrossEntropyLoss(),
        optimizer=create_optimizer,
        rounds=ROUNDS,
        epochs=EPOCHS,
        gamma=GAMMA,
        mu=MU,
        device=get_device()
    )

    fedprox.fit(client_dataloaders, client_dataloaders)
