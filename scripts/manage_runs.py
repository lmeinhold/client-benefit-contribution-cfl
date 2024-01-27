#!/usr/bin/env python3
"""manage_runs.py

Usage:
    manage_runs.py list [--dir=<dir>]
    manage_runs.py archive [--dir=<dir>]
    manage_runs.py delete [--dir=<dir>]
    manage_runs.py (-h | --help)

Arguments:
    --dir=<dir>     Directory containing run files [default: ./output]
    -h --help       Print this message and exit.
"""
import subprocess
import sys
from pathlib import Path

import inquirer
from docopt import docopt


def get_runid_from_file(filename: str) -> str:
    name = filename.split(".")[0]
    parts = name.split("_")
    return "_".join(parts[:2])


def read_runs(directory: Path) -> set[str]:
    """Read run ids from specified directory"""
    files = directory.glob("*")
    return set(map(lambda f: get_runid_from_file(f.name), files))


def list_runs(run_ids: set[str]):
    for run_id in run_ids:
        print(run_id)


def select_runs(text: str, run_ids: set[str]) -> set[str]:
    select_list = sorted(list(run_ids))
    questions = [
        inquirer.Checkbox('run_ids', message=text, choices=select_list)
    ]
    return inquirer.prompt(questions)['run_ids']


def delete_runs(selected_runs: set[str], directory: Path):
    questions = [
        inquirer.Text('confirmation', message='Delete selected runs? (y/N)')
    ]
    answers = inquirer.prompt(questions)
    if answers['confirmation'] in ["y", "yes"]:
        for run in selected_runs:
            files = directory.glob(f"{run}_*")
            for f in files:
                f.unlink(missing_ok=True)
    else:
        print("Delete canceled")


def archive_runs(selected_runs: set[str], directory: Path):
    for run in selected_runs:
        files = directory.glob(f"{run}_*")
        subprocess.run(["tar", "-czf", f"{run}.tar.gz", *files])


def main():
    arguments = docopt(__doc__, version="manage_runs.py 1.0")

    run_dir = Path(arguments["--dir"])
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Directory {run_dir} does not exist or is not a directory")

    run_ids = read_runs(run_dir)
    if len(run_ids) == 0:
        print("No runs found")
        sys.exit(0)

    if arguments["list"]:
        list_runs(run_ids)
    elif arguments["delete"]:
        selected_runs = select_runs("Select runs to delete:", run_ids)
        if len(selected_runs) == 0:
            print("No runs to delete")
            sys.exit(0)
        delete_runs(selected_runs, run_dir)
    elif arguments["archive"]:
        selected_runs = select_runs("Select runs to archive:", run_ids)
        archive_runs(selected_runs, run_dir)
    else:
        raise Exception(f"Unknown subcommand")


if __name__ == "__main__":
    main()
