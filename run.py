from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

from experiments.datasets.base import create_dataloader
from experiments.datasets.cifar import CIFAR10
from experiments.datasets.imbalancing import split_with_label_distribution_skew
from federated_learning.fedprox import FedProx
from models.cifar import CNN
from utils.torchutils import get_device
from datetime import datetime

OUTPUT_DIR = "output/"

N_CLIENTS = 100
GAMMA = [0.1, 0.5, 1, 5, 10]
BATCH_SIZE = 64
LR = 2e-3
EPOCHS = 3
ROUNDS = 5
ALPHA = 0.9
MU = [0.1, 1, 10]
MODEL = CNN


def create_optimizer(params):
    return AdamW(params, LR)


def run_fedprox(model, rounds, epochs, gamma, mu, train_loader, test_loaders):
    fedprox = FedProx(
        model_class=model,
        loss=CrossEntropyLoss(),
        optimizer=create_optimizer,
        rounds=rounds,
        epochs=epochs,
        clients_per_round=gamma,
        mu=mu,
        device=get_device()
    )

    return fedprox.fit(train_loader, test_loaders)


if __name__ == "__main__":
    dataset = CIFAR10("data").train_data()
    client_dataloaders = [create_dataloader(ds, BATCH_SIZE) for ds in
                          split_with_label_distribution_skew(dataset, GAMMA, N_CLIENTS)]

    datestr = datetime.now().strftime("%Y%m%d_%H%M%S")

    for gamma in GAMMA:
        for mu in MU:
            print(f"Running with gamma={gamma} and mu={mu}")
            results = run_fedprox(MODEL, ROUNDS, EPOCHS, gamma, mu, client_dataloaders, client_dataloaders)
            results.as_dataframe()\
                   .to_csv(OUTPUT_DIR + f"{datestr}_mu{mu}_gamma{gamma}.csv", index=False)
