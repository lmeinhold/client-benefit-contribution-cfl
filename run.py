from datetime import datetime
from pathlib import Path

from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

from experiments.datasets.base import create_dataloader
from experiments.datasets.cifar import CIFAR10
from experiments.datasets.imbalancing import split_with_label_distribution_skew
from federated_learning.fedprox import FedProx
from federated_learning.flsc import FLSC
from models.cifar import CNN
from utils.torchutils import get_device

OUTPUT_DIR = "output/"

N_CLIENTS = 80
GAMMA = 0.8
BATCH_SIZE = 64
LR = 2e-3
EPOCHS = 5
ROUNDS = 80
ALPHA = [0.1]
MU = [0.1, 1, 10]  # 1 for FederatedAveraging
N_CLUSTERS = [3, 5]
CLUSTERS_PER_CLIENT = [1, 2, 3]  # 1 for IFCA
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


def run_flsc(model, rounds, epochs, gamma, n_clusters, clusters_per_client, train_loaders, test_loaders):
    flsc = FLSC(
        model_class=model,
        loss=CrossEntropyLoss(),
        optimizer=create_optimizer,
        rounds=rounds,
        epochs=epochs,
        clients_per_round=gamma,
        n_clusters=n_clusters,
        clusters_per_client=clusters_per_client,
        device=get_device()
    )

    return flsc.fit(train_loaders, test_loaders)


if __name__ == "__main__":
    dataset = CIFAR10("/tmp").train_data()
    client_dataloaders = [create_dataloader(ds, BATCH_SIZE) for ds in
                          split_with_label_distribution_skew(dataset, ALPHA[0], N_CLIENTS)]  # TODO

    datestr = datetime.now().strftime("%Y%m%d_%H%M%S")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    n_configs = len(ALPHA) * (len(MU) + len(N_CLUSTERS) * len(CLUSTERS_PER_CLIENT))
    cur_config = 0
    for alpha in ALPHA:
        for mu in MU:
            print(f"Running with mu={mu}")
            cur_config += 1
            print(f"...FedProx ({cur_config}/{n_configs})")
            run_fedprox(MODEL, ROUNDS, EPOCHS, GAMMA, mu, client_dataloaders, client_dataloaders) \
                .as_dataframe() \
                .to_csv(OUTPUT_DIR + f"{datestr}_fedprox_alpha{ALPHA}_mu{mu}_gamma{GAMMA}.csv", index=False)
        for nc in N_CLUSTERS:
            for cpc in CLUSTERS_PER_CLIENT:
                cur_config += 1
                print(f"...FLSC with {nc} clusters, {cpc} per client ({cur_config}/{n_configs})")
                run_flsc(MODEL, ROUNDS, EPOCHS, GAMMA, nc, cpc, client_dataloaders, client_dataloaders) \
                    .as_dataframe() \
                    .to_csv(OUTPUT_DIR + f"{datestr}_flsc_alpha{ALPHA[0]}_mu0_gamma{GAMMA}_nc{nc}_cpc{cpc}.csv", # TODO
                            index=False)
