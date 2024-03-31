#!/usr/bin/env python3
"""Run evaluation.

Usage:
    eval.py <RUN_DB> [--output=<OUTPUT_DIR> --verbose]
    eval.py (-h | --help)
    eval.py --version

Options:
    -o --output=<OUTPUT_DIR>    Write output files to this directory [default: ./output/eval]
    -v --verbose                Print debug infos [default: False]
    --version                   Show version
    -h --help                   Show this screen and exit"""
import logging
import sys
from pathlib import Path

import duckdb
from docopt import docopt

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eval.py")

SQL_ALL = """
SELECT *
FROM configurations
NATURAL JOIN metrics
NATURAL JOIN data_distributions
"""

def get_runid_from_file(filename: str) -> str:
    return filename.split(".")[0]

def single_col_to_list(results):
    return [row[0] for row in results]

def main():
    arguments = docopt(__doc__, version='eval.py 1.0')
    if arguments['--verbose']:
        logger.setLevel(logging.DEBUG)

    dbfile = Path(arguments['<RUN_DB>'])
    if not dbfile.exists() or not dbfile.is_file():
        print(f'No such file or directory "{dbfile.absolute()}"', file=sys.stderr)
        return

    outdir = Path(arguments['--output'])
    outdir.mkdir(parents=True, exist_ok=True)

    run_id = get_runid_from_file(dbfile.name)
    logger.info(f"Evaluating run {run_id}")
    conn = duckdb.connect(str(dbfile))

    logs = conn.sql(SQL_ALL)

    algorithms = single_col_to_list(conn.sql("SELECT DISTINCT algorithm FROM logs").fetchall())
    datasets = single_col_to_list(conn.sql("SELECT DISTINCT dataset FROM logs").fetchall())
    variables = single_col_to_list(conn.sql("SELECT DISTINCT variable FROM logs").fetchall())
    imbalance_types = single_col_to_list(conn.sql("SELECT DISTINCT imbalance_type FROM logs").fetchall())
    imbalance_values = single_col_to_list(conn.sql("SELECT DISTINCT imbalance_value FROM logs").fetchall())
    # sub_ids = single_col_to_list(conn.sql("SELECT DISTINCT sub_id FROM logs").fetchall())

    for dataset in datasets:
        for imbalance_type in imbalance_types:
            for algorithm in algorithms:
                logs_algorithm = conn.sql("SELECT * FROM logs WHERE dataset = dataset AND imbalance_type = imbalance_type AND algorithm = algorithm")
                prefix_algorithm = f"{run_id}_{dataset}_{imbalance_type}_{algorithm}"
                for imbalance_value in imbalance_values:
                    logs_imbalance = conn.sql("SELECT * FROM logs_algorithm WHERE imbalance_value = imbalance_value").df()
                    prefix_imbalance = f"{prefix_algorithm}_{imbalance_value}"

                    fig = px.histogram(logs_imbalance, x="label_imbalance", title="Label Imbalance", nbins=10)
                    fig.write_image(outdir / (prefix_imbalance + "_label_imbalance_hist.png"))

                    fig = px.histogram(logs_imbalance, x="label_distribution_imbalance", title="Label Distribution Imbalance", nbins=10)
                    fig.write_image(outdir / (prefix_imbalance + "_label_dist_imbalance_hist.png"))

                    fig = px.histogram(logs_imbalance, x="quantity_imbalance", title="Quantity Imbalance", nbins=10)
                    fig.write_image(outdir / (prefix_imbalance + "_quantity_imbalance_hist.png"))

    conn.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted ^C")
        sys.exit(1)
