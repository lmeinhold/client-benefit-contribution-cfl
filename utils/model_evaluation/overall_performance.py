"""Evaluation of general model performance, such as loss and f1 scores during training"""
import duckdb

import seaborn as sns

ALGORITHM_ORDER = ["local", "FedAvg", "FedProx", "IFCA", "FLSC"]


def loss_plots(conn: duckdb.DuckDBPyConnection, run_data: duckdb.DuckDBPyRelation, weighted: bool = True):
    """Plot loss vs. round for all algorithms/alphas"""
    average_loss_data = conn.sql(f"""SELECT algorithm,
                                            imbalance_type,
                                            imbalance_value as alpha,
                                            round,
                                            stage,
                                            MEAN(value) AS loss,
                                            SUM(value * client_size) / SUM(client_size) AS weighted_loss,
                                            FROM run_data
                                            WHERE variable = 'loss'
                                            GROUP BY algorithm, imbalance_type, imbalance_value, round, stage""").pl()

    title_format = "{col_name} (alpha={row_name})"
    imbalance_order = sorted(average_loss_data["alpha"].unique())

    y = "weighted_loss" if weighted else "loss"

    return sns.relplot(average_loss_data, x="round", y=y, col="algorithm", row="alpha", hue="stage", kind="line",
                       row_order=imbalance_order, col_order=ALGORITHM_ORDER)\
        .set_titles(title_format)\
        .set_axis_labels("round", "weighted client loss")


def f1_plots(conn: duckdb.DuckDBPyConnection, run_data: duckdb.DuckDBPyRelation, weighted: bool = True):
    """Plot F1 score vs round for all algorithms/alphas"""
    average_f1_data = conn.sql(f"""SELECT algorithm,
                                            imbalance_type,
                                            imbalance_value as alpha,
                                            round,
                                            MEAN(value) AS f1score,
                                            SUM(value * client_size) / SUM(client_size) AS weighted_f1score,
                                            FROM run_data
                                            WHERE variable = 'f1' AND stage = 'test'
                                            GROUP BY algorithm, imbalance_type, imbalance_value, round, n_clients""").pl()

    title_format = "{col_name}"
    imbalance_order = sorted(average_f1_data["alpha"].unique())

    y = "weighted_f1score" if weighted else "f1score"

    return sns.relplot(average_f1_data, x="round", y=y, col="algorithm", hue="alpha", kind="line",
                       row_order=imbalance_order, col_order=ALGORITHM_ORDER)\
        .set_titles(title_format)\
        .set_axis_labels("round", "weighted F1 score")


def overall_f1_vs_imbalance_plots(conn: duckdb.DuckDBPyConnection, run_data: duckdb.DuckDBPyRelation, weighted: bool = True):
    """Plot the final F1 scores for all algorithms/alphas"""
    final_f1_data = conn.sql(f"""SELECT algorithm,
                                                imbalance_type,
                                                imbalance_value as alpha,
                                                round,
                                                MEAN(value) AS f1score,
                                                SUM(value * client_size) / SUM(client_size) AS weighted_f1score,
                                                FROM run_data
                                                WHERE variable = 'f1'
                                                    AND stage = 'test'
                                                    AND round = rounds - 1 -- last round
                                                GROUP BY algorithm, imbalance_type, imbalance_value, round, n_clients""").pl()

    y = "weighted_f1score" if weighted else "f1score"

    ax = sns.pointplot(final_f1_data, x="alpha", y=y, hue="algorithm")
    ax.set(xlabel="alpha", ylabel="F1 score")
    return ax.get_figure()
