#!/usr/bin/env python3

"""Run experiments

Usage:
  run_experiments.py [--experiments=<experiments>]
  run_experiments.py (-l | --list)
  run_experiments.py (-h | --help)

Options:
  -h --help                         Show this screen.
  -e --experiments=<experiments>    Select experiments to run [default: all].
  -l --list                         List available experiments
"""
import dataclasses
from typing import Callable

from docopt import docopt

from experiments.fedavg_mnist import main as run_fedavg_mnist
from experiments.fedavg_cifar import main as run_fedavg_cifar


@dataclasses.dataclass
class Experiment:
    key: str
    description: str
    run_fn: Callable


all_experiments = {
    "fedavg_mnist": Experiment("fedavg_mnist", "FedAvg on MNIST", run_fedavg_mnist),
    "fedavg_cifar": Experiment("fedavg", "FedAvg on CIFAR10", run_fedavg_cifar)
}


def list_experiments():
    """List all available experiments"""
    print("Implemented experiments:")
    for k, v in all_experiments.items():
        print(f"\t {k}: {v.description}")


def run_experiments(experiments: str | list[str]):
    for arg in experiments:
        e = all_experiments[arg]
        print(f"Runnning {e.description}")
        e.run_fn()


if __name__ == "__main__":
    arguments = docopt(__doc__)

    import torch

    print(f"CUDA available: {torch.cuda.is_available()}")

    if arguments["--list"]:
        list_experiments()
    else:
        if "--experiments" in arguments and not "all" in arguments["--experiments"]:
            experiments = arguments["--experiments"]
            run_experiments(experiments)
        else:
            run_experiments(all_experiments.keys())
