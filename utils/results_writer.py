import json
from typing import Any

import numpy as np
import pandas as pd


class ResultsWriter:
    def __init__(self):
        self.metrics = []
        self.infos = []

    def write(self, round: int, client: str, stage: str, info: dict[str, Any] = None, **kwargs) -> "ResultsWriter":
        for k, v in kwargs.items():
            self._write_single_metric(round, client, stage, k, v)
        self._write_info(round, client, stage, info)
        return self

    def _write_single_metric(self, round: int, client: str, stage: str, name: str, value):
        self.metrics.append([round, client, stage, name, value])

    def _write_info(self, round: int, client: str, stage: str, info: dict[str, Any]):
        self.infos.append([round, client, stage, json.dumps(info)])

    def as_dataframes(self):
        metric_df = pd.DataFrame(self.metrics, columns=["round", "client", "stage", "variable", "value"])
        info_df = pd.DataFrame(self.infos, columns=["round", "client", "stage", "info"])
        return metric_df, info_df


def join_cluster_identities(identities: np.ndarray) -> str:
    return "|".join(map(str, identities))
