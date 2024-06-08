import duckdb
import polars as pl

import matplotlib.pyplot as plt
import seaborn as sns

def compute_client_contribution(conn: duckdb.DuckDBPyConnection, data: duckdb.DuckDBPyRelation) -> pl.DataFrame:
    last_round_f1scores = conn.sql("""SELECT algorithm, client, imbalance_value, client_size, value, quantity_imbalance, 
                        label_imbalance, label_distribution_imbalance, feature_imbalance, feature_distribution_imbalance,
                        left_out_clients
                     FROM data WHERE round = rounds - 1 AND stage = 'test' AND variable = 'f1'""")
    f1scores_wide = conn.sql("""PIVOT last_round_f1scores ON algorithm USING first(value)""")
    # df = df.with_columns(df['info'].str.json_decode()).unnest('info')

    benefits = conn.sql("""SELECT client,
                                  imbalance_value,
                                  client_size,
                                  quantity_imbalance,
                                  label_imbalance,
                                  label_distribution_imbalance,
                                  feature_imbalance,
                                  feature_distribution_imbalance,
                                  FedAvg - local AS "FedAvg",
                                  FedProx - local AS "FedProx",
                                  IFCA - local AS "IFCA",
                                  FLSC - local AS "FLSC"
                           FROM f1scores_wide""")

    unpivot_algorithm = conn.sql("UNPIVOT benefits ON FedAvg, FedProx, IFCA, FLSC "
                                 "INTO NAME algorithm VALUE client_benefit")
    return unpivot_algorithm.pl()