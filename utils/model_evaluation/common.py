import re

import numpy as np

MEASURE_LABELS = {
    "quantity_imbalance": "QI",
    "feature_imbalance": "FI",
    "feature_distribution_imbalance": "FDI",
    "label_imbalance": "LI",
    "label_distribution_imbalance": "LDI",
}

ALGORITHMS = ["FedAvg", "FedProx", "IFCA", "FLSC"]
CLUSTER_ALGORITHMS = ["IFCA", "FLSC"]


def fix_client_labels(s: str) -> np.ndarray:
    """Extract client labels saved as string"""
    return np.asarray(list(map(int, re.findall(r"\d+", s))))


def extract_majority_label(labels: np.ndarray) -> int:
    values, counts = np.unique(labels, return_counts=True)
    max_index = np.argmax(counts)
    return values[max_index]


def extract_minority_label(labels: np.ndarray) -> int:
    values, counts = np.unique(labels, return_counts=True)
    min_index = np.argmin(counts)
    return values[min_index]
