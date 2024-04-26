"""Different metrics for measuring data imbalance for each client"""

import numpy as np
from scipy.spatial import distance
from torch.utils.data import Dataset

from utils.torchutils import TransformingSubset


def _phi_j(dataset: TransformingSubset) -> np.ndarray:
    return dataset.features.sum(axis=0)


def secure_aggregation_features(datasets: list[TransformingSubset]) -> tuple[np.ndarray, np.ndarray]:
    """Computer secure aggregation feature vectors"""
    phi_js = np.asarray(list(map(_phi_j, datasets)))
    Phi = phi_js.sum(axis=0)
    return phi_js, Phi


def _feature_imbalance(phi_j: np.ndarray) -> float:
    """Compute the label imbalance for a single client"""
    max_p = phi_j.max()
    min_p = phi_j.min()

    return max_p / min_p if min_p > 0 else max_p


def feature_imbalances(phi_js: np.ndarray) -> np.ndarray:
    """Compute the label imbalance per client"""
    return np.asarray(list(map(_feature_imbalance, phi_js)))


def _feature_distribution_imbalance(phi_j: np.ndarray, Phi: np.ndarray) -> float:
    """Compute the label distribution imbalance for a single client"""
    return distance.cosine(phi_j, Phi)


def feature_distribution_imbalances(phi_js: np.ndarray, Phi: np.ndarray) -> np.ndarray:
    """Compute the label distribution imbalance per client"""
    return np.asarray(list(map(lambda v_j: _feature_distribution_imbalance(v_j, Phi), phi_js)))


def _v_j(dataset: Dataset) -> np.ndarray:
    """Compute v_j for a single client"""
    labels = np.asarray([dataset[i][1] for i in range(len(dataset))])
    return labels.sum(axis=0)


def secure_aggregation(datasets: list[Dataset]) -> tuple[np.ndarray, np.ndarray]:
    """Compute the secure aggregation vectors v_j and V"""
    vjs = np.asarray(list(map(_v_j, datasets)))
    V = vjs.sum(axis=0)
    return vjs, V


def _label_imbalance(v_j: np.ndarray) -> float:
    """Compute the label imbalance for a single client"""
    max_p = v_j.max()
    min_p = v_j.min()

    return max_p / min_p if min_p > 0 else max_p


def label_imbalances(vjs: np.ndarray) -> np.ndarray:
    """Compute the label imbalance per client"""
    return np.asarray(list(map(_label_imbalance, vjs)))


def _label_distribution_imbalance(v_j: np.ndarray, V: np.ndarray) -> float:
    """Compute the label distribution imbalance for a single client"""
    return distance.cosine(v_j, V)


def label_distribution_imbalances(vjs: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Compute the label distribution imbalance per client"""
    return np.asarray(list(map(lambda v_j: _label_distribution_imbalance(v_j, V), vjs)))


def _quantity_imbalance(v_j: np.ndarray, N: int, J: int) -> float:
    """Compute the quantity imbalance for a single client"""
    N_j = v_j.sum()
    return (N_j * J) / N


def quantity_imbalances(vjs: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Compute the quantity imbalance per client"""
    N = V.sum()
    J = vjs.shape[0]
    return np.asarray(list(map(lambda v_j: _quantity_imbalance(v_j, N, J), vjs)))


def li_ldi_qi(datasets: list[Dataset]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute label imbalance, label distribution imbalance and quantity imbalance for the given client datasets"""
    vjs, V = secure_aggregation(datasets)
    li = label_imbalances(vjs)
    ldi = label_distribution_imbalances(vjs, V)
    qi = quantity_imbalances(vjs, V)

    return li, ldi, qi


def fi_fdi(datasets: list[TransformingSubset]) -> tuple[np.ndarray, np.ndarray]:
    phi_js, Phi = secure_aggregation_features(datasets)
    fi = feature_imbalances(phi_js)
    fdi = feature_distribution_imbalances(phi_js, Phi)

    return fi, fdi
