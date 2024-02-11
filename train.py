#!/usr/bin/env python3
"""Model Training.

Usage:
    train.py [--datasets=<datasets> --imbalance-types=<imbalance-types> --imbalances=<imbalances> --algorithms=<algorithms> --rounds=<rounds> --epochs=<epochs> --penalty=<penalty> --n-clients=<n-clients> --clients-per-round=<clients-per-round> --clusters=<clusters> --clusters-per-client=<clusters-per-client> --resume=<run_id> --seed=<seed> --cpu --dry-run --verbose]
    train.py (--list-algorithms | --list-datasets)
    train.py (-h | --help)
    train.py --version

Options:
    --datasets=<datasets>                           List of datasets to train on. [default: all]
    --imbalance-types=<imbalance-types>             List of data imbalances to apply to each dataset. [default: all]
    --imbalances=<imbalances>                       List of severities of data imbalance to use per imbalance type. [default: 0.1,1,5,10]
    --algorithms=<algorithms>                       List of algorithms to run. [default: all]
    --rounds=<rounds>                               Number of federated learning rounds. [default: 100]
    --epochs=<epochs>                               Number of epochs per round. [default: 5]
    --penalty=<penalty>                             Factor µ for the proximal term. (FedProx) [default: 0.1]
    --clusters=<clusters>                           Number of clusters. [default: 3]
    --n-clients=<n-clients>                         Number of clients. [default: 100]
    --clients-per-round=<clients-per-round>         Fraction of clients selected for training per round. [default: 0.8]
    --clusters-per-client=<clusters-per-client>     Maximum number of clusters that a client is assigned to. [default: 2]

    --resume=<run_id>                               Resume training from a previous run. [default:]
    --seed=<seed>                                   A seed to make training reproducible. [default: 42]
    --cpu                                           Use CPU only. [default: False]
    --dry-run                                       Generate configs only, without training.
    --verbose                                       Print debug info

    --list-algorithms                               List all implemented algorithms.
    --list-datasets                                 List all implemented datasets.

    --version                                       Show version
    -h --help                                       Show this screen
"""
import functools
import logging
import os.path
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import jsonpickle
import numpy as np
import pandas as pd
import torch
from docopt import docopt
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import models.cifar as cifar_models
import models.mnist as mnist_models
from datasets.base import create_dataloader
from datasets.cifar import CIFAR10
from datasets.emnist import EMNIST
from datasets.imbalancing.imbalancing import split_dataset_equally, split_with_quantity_skew, \
    split_with_label_distribution_skew, train_test_split
from datasets.imbalancing.stats import li_ldi_qi
from datasets.mnist import MNIST
from federated_learning.fedprox import FedProx
from federated_learning.flsc import FLSC
from federated_learning.local import LocalModels
from utils.results_writer import ResultsWriter
from utils.torchutils import get_device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train.py")

ALL_ALGORITHMS = ["FedAvg", "FedProx", "IFCA", "FLSC", "Local"]
ALL_DATASETS = ["mnist", "cifar10"]
ALL_DATA_IMBALANCES = ["iid", "quantity_distribution", "label_distribution"]

DATASETS = {
    "mnist": MNIST,
    "cifar10": CIFAR10,
    "emnist": EMNIST,
}

MODELS = {
    "mnist": mnist_models.CNN,
    "emnist": mnist_models.CNN,
    "cifar10": cifar_models.CNN,
}

IMBALANCES = {
    "iid": split_dataset_equally,
    "quantity_distribution": split_with_quantity_skew,
    "label_distribution": split_with_label_distribution_skew,
}

LOSS_FN = CrossEntropyLoss
LR = 1e-2
BATCH_SIZE = 64
TEST_SIZE = 0.2

OUTPUT_DIR = "./output/"
DATA_DIR = "/var/tmp"


def create_optimizer(params):
    return SGD(params, LR)


def parse_list_arg(arg: str) -> list[str]:
    if arg == "":
        raise Exception("Empty argument!")

    if "," not in arg:
        return [arg]

    return arg.split(",")


def to_int_list(strings: list[str]) -> list[int]:
    return list(map(int, strings))


def to_float_list(strings: list[str]) -> list[float]:
    return list(map(float, strings))


@dataclass(eq=True, frozen=True)
class RunConfig:
    algorithm: str
    dataset: str
    rounds: int
    epochs: int
    n_clients: int
    clients_per_round: float
    imbalance_type: any
    imbalance_value: float


@dataclass(eq=True, frozen=True)
class FedProxConfig(RunConfig):
    penalty: float


@dataclass(eq=True, frozen=True)
class FlscConfig(RunConfig):
    clusters: int
    clusters_per_client: int


def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_config(algorithm: str, dataset: str, rounds: int, epochs: int, n_clients: int, clients_per_round: float,
                  penalty: float, clusters: int, clusters_per_client: int, imbalance_type,
                  imbalance_value: float) -> RunConfig:
    """Create a run config. Checks parameters for correct algorithm name"""
    actual_imbalance_value = imbalance_value if imbalance_type != "iid" else 1

    if algorithm == "fedavg" or algorithm == "fedprox" and penalty == 0:
        return FedProxConfig(
            algorithm="FedAvg",
            dataset=dataset,
            rounds=rounds,
            epochs=epochs,
            n_clients=n_clients,
            clients_per_round=clients_per_round,
            penalty=0,
            imbalance_type=imbalance_type,
            imbalance_value=actual_imbalance_value,
        )
    elif algorithm == "fedprox":
        return FedProxConfig(
            algorithm="FedProx",
            dataset=dataset,
            rounds=rounds,
            epochs=epochs,
            n_clients=n_clients,
            clients_per_round=clients_per_round,
            penalty=penalty,
            imbalance_type=imbalance_type,
            imbalance_value=actual_imbalance_value,
        )
    elif algorithm == "ifca" or algorithm == "flsc" and clusters_per_client == 1:
        return FlscConfig(
            algorithm="IFCA",
            dataset=dataset,
            rounds=rounds,
            epochs=epochs,
            n_clients=n_clients,
            clients_per_round=clients_per_round,
            clusters=clusters,
            clusters_per_client=1,
            imbalance_type=imbalance_type,
            imbalance_value=actual_imbalance_value,
        )
    elif algorithm == "flsc":
        return FlscConfig(
            algorithm="FLSC",
            dataset=dataset,
            rounds=rounds,
            epochs=epochs,
            n_clients=n_clients,
            clients_per_round=clients_per_round,
            clusters=clusters,
            clusters_per_client=clusters_per_client,
            imbalance_type=imbalance_type,
            imbalance_value=actual_imbalance_value,
        )
    elif algorithm == "local" or algorithm == "global":
        return RunConfig(
            algorithm=algorithm,
            dataset=dataset,
            rounds=rounds,
            epochs=epochs,
            n_clients=n_clients if algorithm == "local" else 1,
            clients_per_round=1,
            imbalance_type=imbalance_type,
            imbalance_value=actual_imbalance_value,
        )
    else:
        raise Exception(f"Unknown algorithm '{algorithm}'")


def run(run_config: RunConfig, train_data, test_data, device: str = "cpu") -> ResultsWriter:
    """Perform a single training run based on the given config"""
    alg = run_config.algorithm.lower()
    if alg in ["fedavg", "fedprox"]:
        assert isinstance(run_config, FedProxConfig)
        return run_fedprox(run_config, train_data, test_data, device=device)
    elif alg in ["ifca", "flsc"]:
        assert isinstance(run_config, FlscConfig)
        return run_flsc(run_config, train_data, test_data, device=device)
    elif alg == "local":
        return run_local(run_config, train_data, test_data, device=device)
    elif alg == "global":
        return run_global(run_config, train_data, test_data, device=device)
    else:
        raise Exception(f"Unknown algorithm")


def run_fedprox(run_config: FedProxConfig, train_data, test_data, device: str = "cpu") -> ResultsWriter:
    """Run a single FedAvg or FedProx training run"""
    fedprox = FedProx(
        model_class=MODELS[run_config.dataset],
        loss=LOSS_FN(),
        optimizer=create_optimizer,
        rounds=run_config.rounds,
        epochs=run_config.epochs,
        clients_per_round=run_config.clients_per_round,
        mu=run_config.penalty,
        device=device
    )
    return fedprox.fit(train_data, test_data)


def run_flsc(run_config: FlscConfig, train_data, test_data, device: str = "cpu") -> ResultsWriter:
    """Run a single IFCA or FLSC training run"""
    flsc = FLSC(
        model_class=MODELS[run_config.dataset],
        loss=LOSS_FN(),
        optimizer=create_optimizer,
        rounds=run_config.rounds,
        epochs=run_config.epochs,
        n_clusters=run_config.clusters,
        clusters_per_client=run_config.clusters_per_client,
        clients_per_round=run_config.clients_per_round,
        device=device
    )
    return flsc.fit(train_data, test_data)


def run_local(run_config: RunConfig, train_data, test_data, device: str = "cpu") -> ResultsWriter:
    """Train a local model for each client"""
    local = LocalModels(
        model_class=MODELS[run_config.dataset],
        loss=LOSS_FN(),
        optimizer=create_optimizer,
        rounds=run_config.rounds,
        epochs=run_config.epochs,
        device=device
    )
    return local.fit(train_data, test_data)


def run_global(run_config: RunConfig, train_data, test_data, device: str = "cpu") -> ResultsWriter:
    """Train a global model for each dataset"""
    raise NotImplementedError()


def main():
    arguments = docopt(__doc__, version="Model Training 1.0")
    verbose = arguments["--verbose"]
    if verbose:
        logger.setLevel(logging.DEBUG)

    logger.debug(arguments)

    seed = int(arguments["--seed"])
    logger.debug(f"Using seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    if arguments["--list-algorithms"]:
        print("\n".join(ALL_ALGORITHMS))
        sys.exit(0)

    if arguments["--list-datasets"]:
        print("\n".join(ALL_DATASETS))
        sys.exit(0)

    logger.debug(f"GPU available: {torch.cuda.is_available()}")
    device = "cpu" if arguments["--cpu"] else get_device()
    logger.info(f"Using device: '{device}'")

    algorithms = parse_list_arg(arguments["--algorithms"])
    clusters = to_int_list(parse_list_arg(arguments["--clusters"]))
    clusters_per_client = to_int_list(parse_list_arg(arguments["--clusters-per-client"]))
    datasets = parse_list_arg(arguments["--datasets"])
    penalty = to_float_list(parse_list_arg(arguments["--penalty"]))

    imbalances_types = parse_list_arg(arguments["--imbalance-types"])
    imbalance_values = to_float_list(parse_list_arg(arguments["--imbalances"]))

    rounds = int(arguments["--rounds"])
    epochs = int(arguments["--epochs"])
    clients_per_round = float(arguments["--clients-per-round"])
    n_clients = int(arguments["--n-clients"])

    # replace "all" arguments
    if "all" in algorithms:
        algorithms = ALL_ALGORITHMS

    if "all" in datasets:
        datasets = ALL_DATASETS

    if "all" in imbalances_types:
        imbalances_types = ALL_DATA_IMBALANCES

    run_id = arguments["--resume"]
    if run_id is None:
        run_id = new_run_id()

    logger.info(f"Generating configs for run '{run_id}'")
    configs = generate_configs(algorithms, n_clients, clients_per_round, clusters, clusters_per_client, datasets,
                               epochs, penalty,
                               rounds, imbalances_types, imbalance_values)

    logger.info(f"...generated {len(configs)} configs")

    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    sub_id = 0
    for config in tqdm(configs, "Run"):
        logger.debug(f"Running {config}")
        filename = f"{run_id}_{sub_id}"
        conf_filename = outdir / (filename + ".config.json")

        if not conf_filename.is_file():
            if not arguments["--dry-run"]:
                train_data, test_data = get_data_for_config(config.dataset, config.n_clients, config.imbalance_type,
                                                            config.imbalance_value, seed,
                                                            logfile=(outdir / (run_id + "_data.csv")))
                results = run(config, train_data, test_data, device=device)
                results_df = results.as_dataframe()
                results_df.to_csv(outdir / (filename + ".csv"), index=False)
                logger.debug(results_df.head())

            with open(conf_filename, "w") as config_file:
                config_file.write(jsonpickle.encode(config, unpicklable=False))
                config_file.write("\n")
        else:
            logger.info(f"Skipping {filename} because it already exists")

        sub_id += 1


def datasets_to_dataloaders(datasets, batch_size=BATCH_SIZE) -> list[DataLoader]:
    return [create_dataloader(d, batch_size) for d in datasets]


def generate_datasets(dataset, n=1, imbalance: str = "iid", alpha: float = 1, seed: int = 42, logfile: str = None):
    """Generate a set of train and test datasets from a given dataset, using the specified imbalance"""
    train = dataset(DATA_DIR).train_data()
    imbalance_fn = IMBALANCES[imbalance.lower()]
    train_datasets = imbalance_fn(train, n, alpha=alpha, seed=seed)

    if logfile is not None:
        log_imbalances(logfile, dataset.get_name().lower(), imbalance, alpha, train_datasets)

    train_datasets, test_datasets = train_test_split(train_datasets, TEST_SIZE, seed=seed)

    return datasets_to_dataloaders(train_datasets), datasets_to_dataloaders(test_datasets)


def log_imbalances(filename, dataset_name, imbalance_type, imbalance_value, datasets):
    li, ldi, qi = li_ldi_qi(datasets)
    df = pd.DataFrame({
        "dataset": dataset_name,
        "client": range(len(li)),
        "imbalance_type": imbalance_type,
        "imbalance_value": imbalance_value,
        "label_imbalance": li,
        "label_distribution_imbalance": ldi,
        "quantity_imbalance": qi,
    })
    df.to_csv(filename, index=False, mode="a", header=not os.path.exists(filename))


@functools.cache
def get_data_for_config(dataset_name: str, n_clients: int, imbalance_type: str, imbalance_value: float, seed: int,
                        logfile: str):
    return generate_datasets(DATASETS[dataset_name.lower()], n_clients, imbalance_type, imbalance_value, seed, logfile)


def generate_configs(algorithms, n_clients, clients_per_round, clusters, clusters_per_client, datasets, epochs, penalty,
                     rounds, imbalance_types, imbalance_values):
    """Generate all permutations of configs"""
    configs = set()
    for imbalance_type in imbalance_types:
        for imbalance_value in imbalance_values:
            for algorithm in algorithms:
                for n_clusters in clusters:
                    for n_clusters_per_client in clusters_per_client:
                        for dataset in datasets:
                            for mu in penalty:
                                configs.add(create_config(
                                    algorithm=algorithm.lower(),
                                    dataset=dataset.lower(),
                                    rounds=rounds,
                                    epochs=epochs,
                                    clients_per_round=clients_per_round,
                                    n_clients=n_clients,
                                    penalty=mu,
                                    clusters_per_client=n_clusters_per_client,
                                    clusters=n_clusters,
                                    imbalance_type=imbalance_type,
                                    imbalance_value=imbalance_value,
                                ))
    return configs


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted ^C")
        sys.exit(2)
