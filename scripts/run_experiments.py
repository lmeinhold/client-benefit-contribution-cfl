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

from docopt import docopt


def list_experiments():
    """List all available experiments"""
    print("Implemented experiments:")


def run_experiments(experiments: str | list[str]):
    pass


if __name__ == "__main__":
    arguments = docopt(__doc__)

    if arguments["--list"]:
        list_experiments()
    else:
        experiments = arguments["--experiments"]
        run_experiments(experiments)
