"""Calculation of client benefit & related plots"""
import json

import duckdb
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import numpy as np

_ = np.nan # prevent optimizing numpy import away

from utils.model_evaluation.common import MEASURE_LABELS, ALGORITHMS, CLUSTER_ALGORITHMS, fix_client_labels, \
    extract_majority_label


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
                        info, client_size, client_labels, client_features
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
           a.info,
           a.client_size,
           a.client_labels,
           a.client_features
    FROM f1scores_nonlocal a
        JOIN f1scores_local l USING(client, imbalance_value)
    """).df()

    benefits["cluster_identities"] = benefits["info"].apply(extract_cluster_assignments)
    benefits.loc[:, "client_labels"] = benefits["client_labels"].astype(str).apply(fix_client_labels)
    benefits["majority_label"] = benefits["client_labels"].apply(extract_majority_label)
    benefits["majority_feature"] = benefits["client_features"].apply(extract_majority_label)

    return benefits


def benefit_imbalance_plots(benefits, measure: str = 'quantity_imbalance'):
    grid = sns.FacetGrid(data=benefits, col='algorithm', col_order=ALGORITHMS)
    return grid.map_dataframe(sns.regplot, x=measure, y='client_benefit', scatter_kws={'s': 5},
                              line_kws={'color': 'orange'}, ci=95, logistic=measure == 'quantity_imbalance').set_titles(
        "{col_name}") \
        .set_xlabels(label=MEASURE_LABELS[measure]) \
        .set_ylabels(label="client benefit")


def benefit_imbalance_reg_quantity(benefits):
    algorithms = list(benefits["algorithm"].unique())
    intercepts, qis, p_intercepts, p_qis, adj_rsqs = [], [], [], [], []
    for a in algorithms:
        df = benefits.query("algorithm == @a")
        mod = smf.ols(formula="client_benefit ~ np.log(quantity_imbalance)", data=df)
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
        "beta_log(QI)": qis,
        "p_QI": p_qis,
        "adj_Rsq": adj_rsqs,
    })


def benefit_imbalance_reg_label(benefits):
    algorithms = list(benefits["algorithm"].unique())
    intercepts, p_intercepts, lis, p_lis, ldis, p_ldis, adj_rsqs = [], [], [], [], [], [], []
    for a in algorithms:
        df = benefits.query("algorithm == @a")
        mod = smf.ols(formula="client_benefit ~ label_imbalance + label_distribution_imbalance",
                      data=df)
        res = mod.fit()

        intercepts.append(res.params.iloc[0])
        lis.append(res.params.iloc[1])
        ldis.append(res.params.iloc[2])
        p_lis.append(res.pvalues.iloc[1])
        p_ldis.append(res.pvalues.iloc[2])
        adj_rsqs.append(res.rsquared_adj)

    return pd.DataFrame({
        "algorithm": algorithms,
        "intercept": intercepts,
        "beta_LI": lis,
        "p_LI": p_lis,
        "beta_LDI": ldis,
        "p_LDI": p_ldis,
        "adj_Rsq": adj_rsqs,
    })


def benefit_imbalance_reg_feature(benefits):
    algorithms = list(benefits["algorithm"].unique())
    intercepts, p_intercepts, fis, p_fis, fdis, p_fdis, adj_rsqs = [], [], [], [], [], [], []
    for a in algorithms:
        df = benefits.query("algorithm == @a")
        mod = smf.ols(
            formula="client_benefit ~ feature_imbalance + feature_distribution_imbalance", data=df)
        res = mod.fit()

        intercepts.append(res.params.iloc[0])
        fis.append(res.params.iloc[1])
        fdis.append(res.params.iloc[2])
        p_fis.append(res.pvalues.iloc[1])
        p_fdis.append(res.pvalues.iloc[2])
        adj_rsqs.append(res.rsquared_adj)

    return pd.DataFrame({
        "algorithm": algorithms,
        "intercept": intercepts,
        "beta_LI": fis,
        "p_LI": p_fis,
        "beta_LDI": fdis,
        "p_LDI": p_fdis,
        "adj_Rsq": adj_rsqs,
    })


def benefit_imbalance_cluster_plots(benefits, measure: str = 'quantity_imbalance', imbalance_value: float = 0.1):
    alg_benefits = benefits.query("algorithm in ['IFCA', 'FLSC'] and imbalance_value == @imbalance_value") \
        .explode('cluster_identities')
    grid = sns.FacetGrid(data=alg_benefits, col='cluster_identities', row='algorithm', row_order=CLUSTER_ALGORITHMS)

    scatter_kws = {'s': 5}

    return grid.map_dataframe(sns.regplot, x=measure, y='client_benefit', scatter_kws=scatter_kws,
                              line_kws={'color': 'orange'}, ci=95, logistic=measure == 'quantity_imbalance') \
        .set_titles("{row_name}, cl. {col_name}") \
        .set_xlabels(label=MEASURE_LABELS[measure]) \
        .set_ylabels(label="client benefit")


def benefit_imbalance_cluster_plots_colors(benefits, measure: str = 'quantity_imbalance', color_by: str = None,
                                           imbalance_value: float = 0.1):
    alg_benefits = benefits.query("algorithm in ['IFCA', 'FLSC'] and imbalance_value == @imbalance_value") \
        .explode('cluster_identities')
    grid = sns.FacetGrid(data=alg_benefits, col='cluster_identities', row='algorithm', row_order=CLUSTER_ALGORITHMS)

    scatter_kws = {'s': 10}
    if color_by == 'label':
        scatter_kws['hue'] = 'majority_label'
    elif color_by == 'feature':
        scatter_kws['hue'] = 'majority_feature'

    return grid.map_dataframe(sns.scatterplot, x=measure, y='client_benefit', **scatter_kws) \
        .set_titles("{row_name}, cl. {col_name}") \
        .set_xlabels(label=MEASURE_LABELS[measure]) \
        .set_ylabels(label="client benefit")


def benefit_cluster_histogram(benefits, imbalance_value: float = 0.1, algorithm="IFCA", by='label', title: str = None):
    value_col = 'client_labels' if by == 'label' else 'client_features'
    alg_benefits = benefits.query("algorithm == @algorithm and imbalance_value == @imbalance_value") \
        .explode('cluster_identities') \
        .explode(value_col)
    alg_benefits['cluster_identities'] = alg_benefits['cluster_identities'].astype('category')
    alg_benefits[value_col] = alg_benefits[value_col].astype('category')

    ax = sns.histplot(
        data=alg_benefits,
        x='cluster_identities',
        hue=value_col,
        multiple='stack',
        alpha=1.0,
    )

    if title is not None:
        ax.set_title(title)

    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), title=by)
    ax.set(xlabel='cluster', ylabel='# of samples')

    return ax.get_figure()
