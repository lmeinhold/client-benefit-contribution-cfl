"""Calculation of client benefit & related plots"""
import json

import duckdb
import pandas as pd

import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf

from utils.model_evaluation.common import MEASURE_LABELS


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
                              line_kws={'color': 'orange'}, ci=95).set_titles("{col_name}")\
                              .set_xlabels(label=MEASURE_LABELS[measure])\
                              .set_ylabels(label="client benefit")


def benefit_imbalance_reg_quantity(benefits):
    algorithms = list(benefits["algorithm"].unique())
    intercepts, qis, p_intercepts, p_qis, adj_rsqs = [], [], [], [], []
    for a in algorithms:
        df = benefits.query("algorithm == @a")
        mod = smf.ols(formula="client_benefit ~ quantity_imbalance", data=df)
        res = mod.fit()

        intercepts.append(res.params.iloc[0])
        qis.append(res.params.iloc[1])
        p_intercepts.append(res.pvalues.iloc[0])
        p_qis.append(res.pvalues.iloc[1])
        adj_rsqs.append(res.rsquared_adj)

    return pd.DataFrame({
        "algorithm": algorithms,
        "intercept": intercepts,
        "p_intercept": p_intercepts,
        "beta_QI": qis,
        "p_QI": p_qis,
        "adj_Rsq": adj_rsqs,
    })


def benefit_imbalance_reg_label(benefits):
    algorithms = list(benefits["algorithm"].unique())
    intercepts, p_intercepts, qis, p_qis, lis, p_lis, ldis, p_ldis, adj_rsqs = [], [], [], [], [], [], [], [], []
    for a in algorithms:
        df = benefits.query("algorithm == @a")
        mod = smf.ols(formula="client_benefit ~ quantity_imbalance + label_imbalance + label_distribution_imbalance", data=df)
        res = mod.fit()

        intercepts.append(res.params.iloc[0])
        qis.append(res.params.iloc[1])
        lis.append(res.params.iloc[2])
        ldis.append(res.params.iloc[3])
        p_qis.append(res.pvalues.iloc[1])
        p_lis.append(res.pvalues.iloc[2])
        p_ldis.append(res.pvalues.iloc[3])
        adj_rsqs.append(res.rsquared_adj)

    return pd.DataFrame({
        "algorithm": algorithms,
        "intercept": intercepts,
        "beta_QI": qis,
        "p_QI": p_qis,
        "beta_LI": lis,
        "p_LI": p_lis,
        "beta_LDI": ldis,
        "p_LDI": p_ldis,
        "adj_Rsq": adj_rsqs,
    })


def benefit_imbalance_reg_feature(benefits):
    algorithms = list(benefits["algorithm"].unique())
    intercepts, p_intercepts, qis, p_qis, fis, p_fis, fdis, p_fdis, adj_rsqs = [], [], [], [], [], [], [], [], []
    for a in algorithms:
        df = benefits.query("algorithm == @a")
        mod = smf.ols(formula="client_benefit ~ quantity_imbalance + feature_imbalance + feature_distribution_imbalance", data=df)
        res = mod.fit()

        intercepts.append(res.params.iloc[0])
        qis.append(res.params.iloc[1])
        fis.append(res.params.iloc[2])
        fdis.append(res.params.iloc[3])
        p_qis.append(res.pvalues.iloc[1])
        p_fis.append(res.pvalues.iloc[2])
        p_fdis.append(res.pvalues.iloc[3])
        adj_rsqs.append(res.rsquared_adj)

    return pd.DataFrame({
        "algorithm": algorithms,
        "intercept": intercepts,
        "beta_QI": qis,
        "p_QI": p_qis,
        "beta_LI": fis,
        "p_LI": p_fis,
        "beta_LDI": fdis,
        "p_LDI": p_fdis,
        "adj_Rsq": adj_rsqs,
    })


def benefit_imbalance_cluster_plots(benefits, measure: str = 'quantity_imbalance'):
    alg_benefits = benefits.query("algorithm in ['IFCA', 'FLSC']").explode('cluster_identities')
    grid = sns.FacetGrid(data=alg_benefits, col='cluster_identities', row='algorithm')
    return grid.map_dataframe(sns.regplot, x=measure, y='client_benefit', scatter_kws={'s': 5},
                              line_kws={'color': 'orange'}, ci=95).set_titles("cluster {col_name}")\
                              .set_xlabels(label=MEASURE_LABELS[measure])\
                              .set_ylabels(label="client benefit")

