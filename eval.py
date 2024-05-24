import logging
import sys
from pathlib import Path

import duckdb
from docopt import docopt

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


def save_plot(plot, path: str | Path):
    plot.get_figure().savefig(path)


def read_data(dbfile):
    conn = duckdb.connect(str(dbfile))
    logs = conn.sql(SQL_ALL)
    algorithms = single_col_to_list(conn.sql("SELECT DISTINCT algorithm FROM logs").fetchall())
    datasets = single_col_to_list(conn.sql("SELECT DISTINCT dataset FROM logs").fetchall())
    variables = single_col_to_list(conn.sql("SELECT DISTINCT variable FROM logs").fetchall())
    imbalance_types = single_col_to_list(conn.sql("SELECT DISTINCT imbalance_type FROM logs").fetchall())
    imbalance_values = single_col_to_list(conn.sql("SELECT DISTINCT imbalance_value FROM logs").fetchall())
    penalties = single_col_to_list(conn.sql("SELECT DISTINCT penalty FROM logs").fetchall())
    # sub_ids = single_col_to_list(conn.sql("SELECT DISTINCT sub_id FROM logs").fetchall())
    return conn, logs, algorithms, datasets, imbalance_types, imbalance_values, penalties
