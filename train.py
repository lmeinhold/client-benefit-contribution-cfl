#!/usr/bin/env python3
"""Model Training.

Usage:
    train.py [--type=<type> --datasets=<datasets> --imbalance-types=<imbalance-types> --imbalances=<imbalances> --algorithms=<algorithms> --rounds=<rounds> --epochs=<epochs> --penalty=<penalty> --n-clients=<n-clients> --clients-per-round=<clients-per-round> --clusters=<clusters> --clusters-per-client=<clusters-per-client> --lxo=<x> --resume=<run_id> --seed=<seed> --cpu --dry-run --verbose]
    train.py (--list-algorithms | --list-datasets | --list-imbalances)
    train.py (-h | --help)
    train.py --version

Options:
    --type=<type>                                   Type of training to perform: for 'benefit' or 'contribution'. [default: benefit]
    --datasets=<datasets>                           List of datasets to train on. [default: all]
    --imbalance-types=<imbalance-types>             List of data imbalances to apply to each dataset. [default: all]
    --imbalances=<imbalances>                       List of severities of data imbalance to use per imbalance type. [default: 0.1,1,5,10]
    --algorithms=<algorithms>                       List of algorithms to run. [default: all]
    --rounds=<rounds>                               Number of federated learning rounds. [default: 100]
    --epochs=<epochs>                               Number of epochs per round. [default: 5]
    --penalty=<penalty>                             Factor µ for the proximal term. (FedProx) [default: 0.1]
    --clusters=<clusters>                           Number of clusters. [default: 3]
    --n-clients=<n-clients>                         Number of clients. [default: 100]
    --clients-per-round=<clients-per-round>         Fraction of clients selected for training per round. [default: 1.0]
    --clusters-per-client=<clusters-per-client>     Maximum number of clusters that a client is assigned to. [default: 2]

    --lxo=<x>                                       Leave x clients out for client contribution computation. [default: 10]

    --resume=<run_id>                               Resume training from a previous run. [default:]
    --seed=<seed>                                   A seed to make training reproducible. [default: 42]
    --cpu                                           Use CPU only. [default: False]
    --dry-run                                       Generate configs only, without training.
    --verbose                                       Print debug info

    --list-algorithms                               List all implemented algorithms.
    --list-datasets                                 List all implemented datasets.
    --list-imbalances                               List all implemented balances

    --version                                       Show version
    -h --help                                       Show this screen and exit
"""
import functools
import logging
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd
from docopt import docopt
from torch.nn import CrossEntropyLoss, BCELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import models.cifar as cifar_models
import models.diabetes as diabetes_models
import models.mnist as mnist_models
from datasets.base import create_dataloader
from datasets.cifar import CIFAR10
from datasets.diabetes import Diabetes
from datasets.emnist import EMNIST
from datasets.imbalancing.imbalancing import *
from datasets.imbalancing.stats import li_ldi_qi, fi_fdi, get_clusters_for_lxo
from datasets.mnist import MNIST
from federated_learning.fedprox import FedProx
from federated_learning.flsc import FLSC
from federated_learning.local import LocalModels
from utils.results_writer import ResultsWriter
from utils.torchutils import get_device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train.py")

ALL_ALGORITHMS = ["FedAvg", "FedProx", "IFCA", "FLSC", "Local"]

DATASETS = {
    "mnist": MNIST,
    "cifar10": CIFAR10,
    "emnist": EMNIST,
    "diabetes": Diabetes,
}

MODELS = {
    "mnist": mnist_models.CNN,
    "emnist": mnist_models.CNN,
    "cifar10": cifar_models.CNN,
    "diabetes": diabetes_models.MLP,
}

IMBALANCES = {
    "iid": split_dataset_equally,
    "quantity_distribution": split_with_quantity_skew,
    "label_distribution": split_with_label_distribution_skew,
    "feature_distribution_single_rotation": split_with_rotation,
    "feature_distribution_rotation_mix": split_with_feature_distribution_skew,
}

ALL_DATASETS = DATASETS.keys()
ALL_DATA_IMBALANCES = IMBALANCES.keys()

LOSS_FN_MULTI = CrossEntropyLoss
LOSS_FN_BINARY = BCELoss
LR = 1e-3
BATCH_SIZE = 256
TEST_SIZE = 0.2

OUTPUT_DIR = "./output/"
DATA_DIR = "/var/tmp/data"


def create_optimizer(params):
    return Adam(params, LR)


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
    ttype: str
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


def create_config(ttype: str, algorithm: str, dataset: str, rounds: int, epochs: int, n_clients: int,
                  clients_per_round: float, penalty: float, clusters: int, clusters_per_client: int, imbalance_type,
                  imbalance_value: float) -> RunConfig:
    """Create a run config. Checks parameters for correct algorithm name"""
    actual_imbalance_value = imbalance_value if imbalance_type != "iid" else 1

    if algorithm == "fedavg" or algorithm == "fedprox" and penalty == 0:
        return FedProxConfig(
            ttype=ttype,
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
            ttype=ttype,
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
            ttype=ttype,
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
            ttype=ttype,
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
            ttype=ttype,
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


def get_loss(run_config: RunConfig):
    if run_config.dataset.lower() == "diabetes":
        return LOSS_FN_BINARY()
    return LOSS_FN_MULTI()


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
    else:
        raise Exception(f"Unknown algorithm")


def run_fedprox(run_config: FedProxConfig, train_data, test_data, device: str = "cpu") -> ResultsWriter:
    """Run a single FedAvg or FedProx training run"""
    fedprox = FedProx(
        model_class=MODELS[run_config.dataset],
        loss_fn=get_loss(run_config),
        optimizer_fn=create_optimizer,
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
        loss_fn=get_loss(run_config),
        optimizer_fn=create_optimizer,
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
        loss=get_loss(run_config),
        optimizer_fn=create_optimizer,
        rounds=run_config.rounds,
        epochs=run_config.epochs,
        device=device
    )
    return local.fit(train_data, test_data)


def run_client_benefit(config, device, train_data, test_data) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform a training run for a single config/client benefit"""
    results = run(config, train_data, test_data, device=device)
    metrics_df, infos_df = results.as_dataframes()
    return infos_df, metrics_df


def run_client_contribution(config: RunConfig, device, client_labels: np.ndarray, train_data, test_data) -> tuple[
    list[np.ndarray], list[pd.DataFrame], list[pd.DataFrame]]:
    """Perform LXO client contribution calculation
    Returns a tuple containing the lists of the clients that were left out in each run,
    as well as the corresponding metrics and info dataframes"""
    unique_labels = np.unique(client_labels)
    left_out_clients = []
    metrics = []
    infos = []
    if config.algorithm.lower() == "local":  # Don't leave clients out when training local models
        results = run(config, train_data, test_data, device=device)
        metrics_df, infos_df = results.as_dataframes()

        return [], [metrics_df], [infos_df]

    for label in unique_labels:
        logger.info(f"Training without clients for label: {label}")
        selected_clients = np.argwhere(client_labels != label).flatten()
        left_out_clients.append(np.argwhere(client_labels == label).flatten())
        logger.debug(f"Clients left out: {left_out_clients[-1]}")

        train_subset = [train_data[i] for i in selected_clients]
        test_subset = [test_data[i] for i in selected_clients]

        results = run(config, train_subset, test_subset, device=device)

        metrics_df, infos_df = results.as_dataframes()
        metrics.append(metrics_df)
        infos.append(infos_df)

    logger.info(f"Training with all clients")
    left_out_clients.append(np.empty(dtype=int))

    results = run(config, train_data, test_data, device=device)

    metrics_df, infos_df = results.as_dataframes()
    metrics.append(metrics_df)
    infos.append(infos_df)

    return left_out_clients, metrics, infos


def datasets_to_dataloaders(datasets, batch_size=BATCH_SIZE) -> list[DataLoader]:
    return [create_dataloader(d, batch_size) for d in datasets]


def generate_datasets(dataset, n=1, imbalance: str = "iid", alpha: float = 1, seed: int = 42,
                      conn: duckdb.DuckDBPyConnection = duckdb.connect(), lxo: int = None):
    """Generate a set of train and test datasets from a given dataset, using the specified imbalance"""
    ds = dataset(DATA_DIR)
    train = ds.train_data()
    imbalance_fn = IMBALANCES[imbalance.lower()]
    train_datasets = imbalance_fn(train, n, alpha=alpha, seed=seed)

    qi, li, ldi, fi, fdi = log_imbalances(conn, ds.get_name().lower(), imbalance, alpha, train_datasets)

    train_datasets, test_datasets = train_test_split(train_datasets, TEST_SIZE, seed=seed)

    cluster_labels = None
    if lxo is not None:
        n_clusters = len(train_datasets) // lxo
        if "feature" in imbalance:
            cluster_labels = get_clusters_for_lxo(n_clusters, qi, li, ldi, fi, fdi)
        else:
            cluster_labels = get_clusters_for_lxo(n_clusters, qi, li, ldi)

    return datasets_to_dataloaders(train_datasets), datasets_to_dataloaders(test_datasets), cluster_labels


def get_tables(conn: duckdb.DuckDBPyConnection) -> list[str]:
    return [r[0] for r in conn.sql("SHOW TABLES").fetchall()]


def log_imbalances(conn: duckdb.DuckDBPyConnection, dataset_name: str, imbalance_type: str,
                   imbalance_value: int | float, datasets):
    li, ldi, qi = li_ldi_qi(datasets)
    labels = [ds.targets for ds in datasets]
    sizes = [len(ds) for ds in datasets]

    fi = np.nan
    fdi = np.nan
    features = None
    if "feature" in imbalance_type:
        fi, fdi = fi_fdi(datasets)
        features = [ds.features for ds in datasets]

    df = pd.DataFrame({
        "dataset": dataset_name,
        "client": range(len(li)),
        "imbalance_type": imbalance_type,
        "imbalance_value": imbalance_value,
        "label_imbalance": li,
        "label_distribution_imbalance": ldi,
        "quantity_imbalance": qi,
        "feature_imbalance": fi,
        "feature_distribution_imbalance": fdi,
        "client_labels": labels,
        "client_features": features,
        "client_size": sizes,
    })

    if "data_distributions" in get_tables(conn):
        sql_exists = """SELECT COUNT(1) > 0
        FROM data_distributions d
        WHERE
            d.dataset = ?
            AND d.imbalance_type = ?
            AND d.imbalance_value = ?"""
        combination_exists = conn.sql(sql_exists, params=[dataset_name, imbalance_type, imbalance_value]) \
            .fetchone()[0]
        if combination_exists:
            logger.debug(
                f"Not logging combination {dataset_name} {imbalance_type} {imbalance_value} because it already exists")
        else:
            conn.append("data_distributions", df)
    else:
        conn.sql("CREATE TABLE data_distributions AS SELECT * FROM df")

    return qi, li, ldi, fi, fdi


@functools.cache
def get_data_for_config(dataset_name: str, n_clients: int, imbalance_type: str, imbalance_value: float, seed: int,
                        conn: duckdb.DuckDBPyConnection, lxo: int | None) -> tuple[
    list[DataLoader], list[DataLoader], np.ndarray]:
    """Generate an imbalanced dataset with the specified parameters for the configuration"""
    return generate_datasets(DATASETS[dataset_name.lower()], n_clients, imbalance_type, imbalance_value, seed, conn,
                             lxo)


def generate_configs(ttype, algorithms, n_clients, clients_per_round, clusters, clusters_per_client, datasets, epochs,
                     penalty, rounds, imbalance_types, imbalance_values):
    """Generate all permutations of configs"""
    configs = set()
    for imbalance_type in imbalance_types:
        for imbalance_value in imbalance_values:
            for algorithm in algorithms:
                for n_clusters in clusters:
                    for n_clusters_per_client in clusters_per_client:
                        if n_clusters_per_client >= n_clusters:  # clusters per client must be lower than number of clusters, otherwise skip this config
                            continue
                        for dataset in datasets:
                            for mu in penalty:
                                configs.add(create_config(
                                    ttype=ttype,
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

    if arguments["--list-imbalances"]:
        print("\n".join(sorted(IMBALANCES.keys())))
        sys.exit(0)

    logger.debug(f"GPU available: {torch.cuda.is_available()}")
    device = "cpu" if arguments["--cpu"] else get_device()
    logger.info(f"Using device: '{device}'")

    ttype = arguments["--type"]
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

    lxo = int(arguments["--lxo"])

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
    configs = generate_configs(ttype, algorithms, n_clients, clients_per_round, clusters, clusters_per_client, datasets,
                               epochs, penalty,
                               rounds, imbalances_types, imbalance_values)

    logger.info(f"...generated {len(configs)} configs")

    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    dbpath = outdir / f"{run_id}.db"
    conn = duckdb.connect(str(dbpath))

    sub_id_n = 0
    for config in tqdm(configs, "Run"):
        sub_id = f"{run_id}_{sub_id_n}"
        logger.debug(f"Running {config}")

        tables = get_tables(conn)
        config_table_exists = "configurations" in tables
        run_exists = config_table_exists and conn.sql(
            f"SELECT COUNT(1) > 0 FROM configurations WHERE configurations.sub_id = '{sub_id_n}'").fetchone()[0]

        if run_exists:
            logger.info(f"Skipping {config.__dict__} because it already exists {config_table_exists} {run_exists}")
        else:
            if not arguments["--dry-run"]:
                train_data, test_data, client_labels = get_data_for_config(config.dataset, config.n_clients,
                                                                           config.imbalance_type,
                                                                           config.imbalance_value, seed, conn, lxo)

                infos, metrics = None, None
                if ttype == "benefit":
                    infos_df, metrics_df = run_client_benefit(config, device, train_data, test_data)
                    left_out_clients = np.empty(dtype=np.int64, shape=[1, ])
                    infos = [infos_df]
                    metrics = [metrics_df]
                elif ttype == "contribution":
                    left_out_clients, metrics, infos = run_client_contribution(config, device, client_labels,
                                                                               train_data, test_data)
                else:
                    raise Exception("Unknown type: {type}")

                for lo, metrics_df, infos_df in zip(left_out_clients, metrics, infos):
                    tables = get_tables(conn)
                    metrics_df["sub_id"] = sub_id
                    infos_df["sub_id"] = sub_id
                    infos_df["left_out_clients"] = [left_out_clients for _ in range(len(infos_df))]

                    if "metrics" in tables:
                        conn.append("metrics", metrics_df)
                    else:
                        conn.sql("CREATE TABLE metrics AS SELECT * FROM metrics_df")

                    if "infos" in tables:
                        conn.append("infos", infos_df)
                    else:
                        conn.sql("CREATE TABLE infos AS SELECT * FROM infos_df")

                config_df = pd.DataFrame(config.__dict__, index=[0])
                config_df["sub_id"] = sub_id

                conn.execute("""CREATE TABLE IF NOT EXISTS configurations (
                    sub_id VARCHAR NOT NULL,
                    ttype VARCHAR NOT NULL,
                    algorithm VARCHAR NOT NULL,
                    dataset VARCHAR NOT NULL,
                    rounds INTEGER NOT NULL,
                    epochs INTEGER NOT NULL,
                    n_clients INTEGER NOT NULL,
                    clients_per_round DOUBLE DEFAULT 1.0,
                    imbalance_type VARCHAR,
                    imbalance_value DOUBLE,
                    penalty DOUBLE DEFAULT 0,
                    clusters INTEGER DEFAULT NULL,
                    clusters_per_client INTEGER DEFAULT NULL
                )""")
                conn.append("configurations", config_df, by_name=True)

                conn.execute("CHECKPOINT")

        sub_id_n += 1

    conn.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted ^C")
        sys.exit(2)
