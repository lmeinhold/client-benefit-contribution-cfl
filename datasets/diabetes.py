from pathlib import Path

import pandas as pd
import torch
from torch.utils import data
from ucimlrepo import fetch_ucirepo

from datasets.base import Dataset


class Diabetes(Dataset):
    """
    A wrapper for the diabetes dataset from the UCI ML repository.
    https://www.archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
    """

    def __init__(self, path: str):
        self.path = Path(path)
        features, labels = self._get_data()
        self.train = data.TensorDataset(features, labels)

    def _get_data(self):
        self.path.mkdir(parents=True, exist_ok=True)

        feature_file = self.path / 'diabetes_features.csv'
        label_file = self.path / 'diabetes_labels.csv'

        if not feature_file.exists() or not label_file.exists():
            cdc_diabetes_health_indicators = fetch_ucirepo(id=891)

            X = cdc_diabetes_health_indicators.data.features
            X.to_csv(feature_file, index=False)
            y = cdc_diabetes_health_indicators.data.targets
            y.to_csv(label_file, index=False)

        X = pd.read_csv(feature_file).to_numpy()
        y = pd.read_csv(label_file).to_numpy()

        return torch.Tensor(X), torch.Tensor(y)

    def train_data(self) -> data.Dataset:
        return self.train

    def test_data(self) -> data.Dataset:
        raise NotImplementedError()

    def get_name(self) -> str:
        return "Diabetes"

    def num_classes(self):
        return 2
