import numpy as np
import pandas as pd


class ResultsWriter:
    def __init__(self):
        self.metrics = []

    def write(self, round: int, client: str, stage: str, **kwargs) -> "ResultsWriter":
        for k, v in kwargs.items():
            self._write_single_metric(round, client, stage, k, v)
        return self

    def _write_single_metric(self, round: int, client: str, stage: str, name: str, value):
        self.metrics.append([round, client, stage, name, value])

    def as_dataframe(self):
        return pd.DataFrame(self.metrics, columns=["round", "client", "stage", "variable", "value"])

    def save(self, path):
        df = self.as_dataframe()
        df.to_csv(path, index=False)


def join_cluster_identities(identities: np.ndarray) -> str:
    return "|".join(map(str, identities))
