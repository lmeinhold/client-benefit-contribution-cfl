import duckdb
import polars as pl
import seaborn as sns
import statsmodels.formula.api as smf
import numpy as np

_ = np.nan # prevent optimizing numpy import away

from utils.model_evaluation.common import MEASURE_LABELS, ALGORITHMS


def compute_client_contribution(conn: duckdb.DuckDBPyConnection, data: duckdb.DuckDBPyRelation) -> pl.DataFrame:
    last_round_f1scores = conn.sql("""
                                   SELECT algorithm, client, imbalance_type, imbalance_value, client_size, value,
                                        info, left_out_clients
                                    FROM data
                                    WHERE round = rounds - 1
                                        AND stage = 'test'
                                        AND variable = 'f1'
                                        AND algorithm <> 'local'
                                   """)

    f1scores_no_clients_left_out = conn.sql("""SELECT * FROM last_round_f1scores WHERE len(left_out_clients) = 0""")
    f1scores_lxo = conn.sql("""SELECT * FROM last_round_f1scores WHERE len(left_out_clients) > 0""")

    contributions = conn.sql("""
    SELECT imbalance_type,
           imbalance_value,
           algorithm,
           MEAN(fll.value) AS f1_full,
           MEAN(lxo.value) AS f1_lxo,
           len(lxo.left_out_clients) AS clients_in_cluster,
           unnest(lxo.left_out_clients) AS left_out_client
    FROM f1scores_no_clients_left_out fll
        JOIN f1scores_lxo lxo USING(client, imbalance_type, imbalance_value, algorithm)
    GROUP BY imbalance_type, imbalance_value, algorithm, lxo.left_out_clients
    """)

    contributions_with_imbalances = conn.sql("""
        SELECT c.imbalance_value,
            c.algorithm,
            (c.f1_full - c.f1_lxo) / c.clients_in_cluster AS client_contribution,
            c.left_out_client,
            d.client_size,
            d.quantity_imbalance,
            d.label_imbalance,
            d.label_distribution_imbalance,
            d.feature_imbalance,
            d.feature_distribution_imbalance
        FROM contributions c
            JOIN data_distributions d ON
                 c.imbalance_type = d.imbalance_type
             AND c.imbalance_value = d.imbalance_value
             AND c.left_out_client = d.client
    """)

    return contributions_with_imbalances.pl()


def contribution_imbalance_plots(contribution, measure: str = 'quantity_imbalance'):
    grid = sns.FacetGrid(data=contribution, col='algorithm', col_order=ALGORITHMS)
    return grid.map_dataframe(sns.regplot, x=measure, y='client_contribution', scatter_kws={'s': 5},
                              line_kws={'color': 'orange'}, ci=95).set_titles("{col_name}") \
        .set_xlabels(label=MEASURE_LABELS[measure]) \
        .set_ylabels(label="client contribution")


def contribution_imbalance_reg_quantity(contributions):
    algorithms = list(contributions["algorithm"].unique())
    intercepts, qis, p_intercepts, p_qis, adj_rsqs = [], [], [], [], []
    for a in algorithms:
        df = contributions.filter(pl.col("algorithm") == a)
        mod = smf.ols(formula="client_contribution ~ quantity_imbalance", data=df)
        res = mod.fit()

        intercepts.append(res.params.iloc[0])
        qis.append(res.params.iloc[1])
        p_intercepts.append(res.pvalues.iloc[0])
        p_qis.append(res.pvalues.iloc[1])
        adj_rsqs.append(res.rsquared_adj)

    return pl.DataFrame({
        "algorithm": algorithms,
        "intercept": intercepts,
        "p_intercept": p_intercepts,
        "beta_QI": qis,
        "p_QI": p_qis,
        "adj_Rsq": adj_rsqs,
    }).to_pandas()


def contribution_imbalance_reg_label(contributions):
    algorithms = list(contributions["algorithm"].unique())
    intercepts, p_intercepts, lis, p_lis, ldis, p_ldis, adj_rsqs = [], [], [], [], [], [], []
    for a in algorithms:
        df = contributions.filter(pl.col("algorithm") == a)
        mod = smf.ols(
            formula="client_contribution ~ + label_imbalance + label_distribution_imbalance",
            data=df)
        res = mod.fit()

        intercepts.append(res.params.iloc[0])
        lis.append(res.params.iloc[1])
        ldis.append(res.params.iloc[2])
        p_lis.append(res.pvalues.iloc[1])
        p_ldis.append(res.pvalues.iloc[2])
        adj_rsqs.append(res.rsquared_adj)

    return pl.DataFrame({
        "algorithm": algorithms,
        "intercept": intercepts,
        "beta_LI": lis,
        "p_LI": p_lis,
        "beta_LDI": ldis,
        "p_LDI": p_ldis,
        "adj_Rsq": adj_rsqs,
    }).to_pandas()


def contribution_imbalance_reg_feature(contributions):
    algorithms = list(contributions["algorithm"].unique())
    intercepts, p_intercepts, fis, p_fis, fdis, p_fdis, adj_rsqs = [], [], [], [], [], [], []
    for a in algorithms:
        df = contributions.filter(pl.col("algorithm") == a)
        mod = smf.ols(
            formula="client_contribution ~ + feature_imbalance + feature_distribution_imbalance",
            data=df)
        res = mod.fit()

        intercepts.append(res.params.iloc[0])
        fis.append(res.params.iloc[1])
        fdis.append(res.params.iloc[2])
        p_fis.append(res.pvalues.iloc[1])
        p_fdis.append(res.pvalues.iloc[2])
        adj_rsqs.append(res.rsquared_adj)

    return pl.DataFrame({
        "algorithm": algorithms,
        "intercept": intercepts,
        "beta_LI": fis,
        "p_LI": p_fis,
        "beta_LDI": fdis,
        "p_LDI": p_fdis,
        "adj_Rsq": adj_rsqs,
    }).to_pandas()
