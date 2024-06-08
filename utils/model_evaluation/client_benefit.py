"""Calculation of client benefit & related plots"""
import json

import duckdb
import pandas as pd

import seaborn as sns


def extract_cluster_assignments(info: str | None) -> list[int] | None:
    """Extract cluster assignments from JSON column"""
    if info is None or pd.isna(info) or pd.isnull(info) or info == "null":
        return info

    d = json.loads(info)
    clusters = d["cluster_identities"]

    return list(map(int, clusters.split('|')))


def compute_client_benefit(conn: duckdb.DuckDBPyConnection, data: duckdb.DuckDBPyRelation) -> pd.DataFrame:
    last_round_f1scores = conn.sql("""SELECT algorithm, client, imbalance_value, client_size, value, quantity_imbalance, 
                        label_imbalance, label_distribution_imbalance, feature_imbalance, feature_distribution_imbalance,
                        info
                     FROM data WHERE round = rounds - 1 AND stage = 'test' AND variable = 'f1'""")

    f1scores_local = conn.sql("""SELECT * FROM last_round_f1scores WHERE algorithm = 'local'""")
    f1scores_nonlocal = conn.sql("""SELECT * FROM last_round_f1scores WHERE algorithm <> 'local'""")

    benefits = conn.sql("""
    SELECT client,
           imbalance_value,
           a.algorithm,
           a.client_size,
           a.quantity_imbalance,
           a.label_imbalance,
           a.label_distribution_imbalance,
           a.feature_imbalance,
           a.feature_distribution_imbalance,
           a.value - l.value AS client_benefit,
           a.info
    FROM f1scores_nonlocal a
        JOIN f1scores_local l USING(client, imbalance_value)
    """).df()

    benefits["cluster_identities"] = benefits["info"].apply(extract_cluster_assignments)
    return benefits


def benefit_imbalance_plots(benefits, measure: str = 'quantity_imbalance'):
    grid = sns.FacetGrid(data=benefits, col='algorithm')
    return grid.map_dataframe(sns.regplot, x=measure, y='client_benefit', scatter_kws={'s': 5},
                              line_kws={'color': 'orange'}, ci=95).set_titles("{col_name}")


def benefit_imbalance_cluster_plots(benefits, measure: str = 'quantity_imbalance'):
    alg_benefits = benefits.query("algorithm in ['IFCA', 'FLSC']").explode('cluster_identities')
    grid = sns.FacetGrid(data=alg_benefits, col='cluster_identities', row='algorithm')
    return grid.map_dataframe(sns.regplot, x=measure, y='client_benefit', scatter_kws={'s': 5},
                              line_kws={'color': 'orange'}, ci=95).set_titles("cluster {col_name}")
