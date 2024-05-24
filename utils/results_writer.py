import json
from typing import Any

import numpy as np
import pandas as pd


class ResultsWriter:
    """Utility class for aggregating model training logs"""

    def __init__(self):
        self.metrics = []
        self.infos = []

    def write(self, round: int, client: str, stage: str, info: dict[str, Any] = None, **kwargs) -> "ResultsWriter":
        """
        Write metrics for a client/round

        Parameters:
            round: current federated learning round
            client: current client
            stage: train or test
            info: additional infos as a dictionary
            kwargs: metrics to log

        Returns:
            self for easy chaining
        """
        for k, v in kwargs.items():
            self._write_single_metric(round, client, stage, k, v)
        self._write_info(round, client, stage, info)
        return self

    def _write_single_metric(self, round: int, client: str, stage: str, name: str, value):
        self.metrics.append([round, client, stage, name, value])

    def _write_info(self, round: int, client: str, stage: str, info: dict[str, Any]):
        self.infos.append([round, client, stage, json.dumps(info)])

    def as_dataframes(self):
        """Get collected metrics and infos as dataframes"""
        metric_df = pd.DataFrame(self.metrics, columns=["round", "client", "stage", "variable", "value"])
        info_df = pd.DataFrame(self.infos, columns=["round", "client", "stage", "info"])
        return metric_df, info_df


def join_cluster_identities(identities: np.ndarray) -> str:
    """
    Serialize a list of cluster identities for logging

        Parameters:
            identities: list of cluster identities as numpy array

        Returns:
            a string representation of the cluster identities
    """
    return "|".join(map(str, identities))
