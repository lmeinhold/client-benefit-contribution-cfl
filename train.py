#!/usr/bin/env python3
"""Model Training.

Usage:
    train.py [--datasets=<datasets> --algorithms=<algorithms> --rounds=<rounds> --epochs=<epochs> --penalty=<penalty> --clients-per-round=<clients-per-round> --clusters=<clusters> --clusters-per-client=<clusters-per-client> --resume=<run_id> --verbose]
    train.py (--list-algorithms | --list-datasets)
    train.py (-h | --help)
    train.py --version

Options:
    --datasets=<datasets>                       List of datasets to train on. [default: all]
    --algorithms=<algorithms>                   List of algorithms to run. [default: all]
    --rounds=<rounds>                           Number of federated learning rounds. [default: 100]
    --epochs=<epochs>                           Number of epochs per round. [default: 5]
    --penalty=<penalty>                         Factor µ for the proximal term. (FedProx) [default: 0.1]
    --clusters=<clusters>                       Number of clusters. [default: 3]
    --clients-per-round=<clients-per-round>     Fraction of clients selected for training per round. [default: 0.8]
    --clusters-per-client=<clusters-per-client> Maximum number of clusters that a client is assigned to. (IFCA/FLSC) [default: 2]

    --resume=<run_id>                           Resume training from a previous run. [default:]
    --verbose                                   Print debug info

    --list-algorithms                           List all implemented algorithms.
    --list-datasets                             List all implemented datasets.

    --version                                   Show version
    -h --help                                   Show this screen
"""
import jsonpickle
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from docopt import docopt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train.py")

ALL_ALGORITHMS = ["FedAvg", "FedProx", "IFCA", "FLSC", "Local", "Global"]
ALL_DATASETS = ["mnist", "cifar10"]

OUTPUT_DIR = "./output/"


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
    clients_per_round: float


@dataclass(eq=True, frozen=True)
class FedProxConfig(RunConfig):
    penalty: float


@dataclass(eq=True, frozen=True)
class FlscConfig(RunConfig):
    clusters: int
    clusters_per_client: int


def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_config(algorithm: str, dataset: str, rounds: int, epochs: int, clients_per_round: float, penalty: float,
                  clusters: int, clusters_per_client: int) -> RunConfig:
    """Create a run config. Checks parameters for correct algorithm name"""
    if algorithm == "fedavg" or algorithm == "fedprox" and penalty == 0:
        return FedProxConfig(
            algorithm="FedAvg",
            dataset=dataset,
            rounds=rounds,
            epochs=epochs,
            clients_per_round=clients_per_round,
            penalty=0,
        )
    elif algorithm == "fedprox":
        return FedProxConfig(
            algorithm="FedProx",
            dataset=dataset,
            rounds=rounds,
            epochs=epochs,
            clients_per_round=clients_per_round,
            penalty=penalty,
        )
    elif algorithm == "ifca" or algorithm == "flsc" and clusters_per_client == 1:
        return FlscConfig(
            algorithm="IFCA",
            dataset=dataset,
            rounds=rounds,
            epochs=epochs,
            clients_per_round=clients_per_round,
            clusters=clusters,
            clusters_per_client=1,
        )
    elif algorithm == "flsc":
        return FlscConfig(
            algorithm="FLSC",
            dataset=dataset,
            rounds=rounds,
            epochs=epochs,
            clients_per_round=clients_per_round,
            clusters=clusters,
            clusters_per_client=clusters_per_client,
        )
    elif algorithm == "local" or algorithm == "global":
        return RunConfig(
            algorithm=algorithm,
            dataset=dataset,
            rounds=1,
            epochs=rounds,
            clients_per_round=1
        )
    else:
        raise Exception(f"Unknown algorithm '{algorithm}'")


def run(run_config: RunConfig, outfile: str | Path):
    """Perform a single training run based on the given config"""
    alg = run_config.algorithm.lower()
    if alg in ["fedavg", "fedprox"]:
        assert isinstance(run_config, FedProxConfig)
        run_fedprox(run_config, outfile)
    elif alg in ["ifca", "flsc"]:
        assert isinstance(run_config, FlscConfig)
        run_flsc(run_config, outfile)
    elif alg == "local":
        run_local(run_config, outfile)
    elif alg == "global":
        run_global(run_config, outfile)
    else:
        raise Exception(f"Unknown algorithm")


def run_fedprox(run_config: FedProxConfig, outfile: str | Path):
    """Run a single FedAvg or FedProx training run"""
    logger.debug(f"Running: {run_config}")
    pass


def run_flsc(run_config: FlscConfig, outfile: str | Path):
    """Run a single IFCA or FLSC training run"""
    logger.debug(f"Running: {run_config}")
    pass


def run_local(run_config: RunConfig, outfile: str | Path):
    """Train a local model for each client"""
    logger.debug(f"Running: {run_config}")
    pass


def run_global(run_config: RunConfig, outfile: str | Path):
    """Train a global model for each dataset"""
    logger.debug(f"Running: {run_config}")
    pass


def main():
    arguments = docopt(__doc__, version="Model Training 1.0")
    verbose = arguments["--verbose"]
    if verbose:
        logger.setLevel(logging.DEBUG)

    logger.debug(f"Arguments: {arguments}")

    if arguments["--list-algorithms"]:
        print("\n".join(ALL_ALGORITHMS))
        sys.exit(0)

    if arguments["--list-datasets"]:
        print("\n".join(ALL_DATASETS))
        sys.exit(0)

    algorithms = parse_list_arg(arguments["--algorithms"])
    clusters = to_int_list(parse_list_arg(arguments["--clusters"]))
    clusters_per_client = to_int_list(parse_list_arg(arguments["--clusters"]))
    datasets = parse_list_arg(arguments["--datasets"])
    penalty = to_float_list(parse_list_arg(arguments["--penalty"]))

    rounds = int(arguments["--rounds"])
    epochs = int(arguments["--epochs"])
    clients_per_round = float(arguments["--clients-per-round"])

    # replace "all" arguments
    if len(algorithms) == 1 and algorithms[0] == "all":
        algorithms = ALL_ALGORITHMS

    if len(datasets) == 1 and datasets[0] == "all":
        datasets = ALL_DATASETS

    run_id = arguments["--resume"]
    if run_id is None:
        run_id = new_run_id()

    logger.info(f"Generating configs for run '{run_id}'")
    configs = set()
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
                            penalty=mu,
                            clusters_per_client=n_clusters_per_client,
                            clusters=n_clusters
                        ))

    logger.info(f"...generated {len(configs)} configs")

    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    sub_id = 0
    for config in configs:
        filename = f"{run_id}_{sub_id}"
        with open(outdir / (filename + ".config.json"), "w") as config_file:
            config_file.write(jsonpickle.encode(config.__dict__, config_file))
            config_file.write("\n")
        run(config, outfile=outdir / filename)
        sub_id += 1


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
