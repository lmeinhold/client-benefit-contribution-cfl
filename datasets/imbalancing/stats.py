"""Different metrics for measuring data imbalance for each client"""
from typing import Any, Sequence

import numpy as np
from scipy.spatial import distance
from sklearn.cluster import KMeans
from torch.utils.data import Dataset

from utils.torchutils import TransformingSubset


class UniqueElementCounter:
    """A counter that collects elements and counts how many unique items were added"""

    def __init__(self):
        self.elements = set()

    def add(self, elem: Any) -> "UniqueElementCounter":
        """Add a single element"""
        self.elements.add(elem)
        return self

    def add_all(self, elems: Sequence[Any]) -> "UniqueElementCounter":
        """Add a sequence of elements"""
        for elem in elems:
            self.elements.add(elem)
        return self

    @property
    def count(self) -> int:
        """Get the count of unique elements that were added to the counter"""
        return len(self.elements)


def _phi_j(dataset: TransformingSubset, n_features: int) -> np.ndarray:
    counts = dict(zip(*np.unique(dataset.features, return_counts=True)))
    return np.asarray([counts[f] if f in counts else 0 for f in range(n_features)])


def secure_aggregation_features(datasets: list[TransformingSubset]) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute secure aggregation feature vectors

        Parameters:
            datasets: list of client datasets

        Returns:
            client feature vectors, global feature vector (sum of phi_js)
    """
    feature_ids = UniqueElementCounter()
    for ds in datasets:
        feature_ids.add_all(ds.features)
    n_features = feature_ids.count

    phi_js = np.asarray(list(map(lambda d: _phi_j(d, n_features), datasets)))
    Phi = phi_js.sum(axis=0)

    return phi_js, Phi


def _feature_imbalance(phi_j: np.ndarray) -> float:
    """Compute the label imbalance for a single client"""
    max_p = phi_j.max()
    min_p = phi_j.min()

    return max_p / min_p if min_p > 0 else max_p


def feature_imbalances(phi_js: np.ndarray) -> np.ndarray:
    """
    Compute the label imbalance per client

        Parameters:
            phi_js: client feature vectors

        Returns:
            client feature imbalances
    """
    return np.asarray(list(map(_feature_imbalance, phi_js)))


def _feature_distribution_imbalance(phi_j: np.ndarray, Phi: np.ndarray) -> float:
    """Compute the label distribution imbalance for a single client"""
    return distance.cosine(phi_j, Phi)


def feature_distribution_imbalances(phi_js: np.ndarray, Phi: np.ndarray) -> np.ndarray:
    """
    Compute the label distribution imbalance per client

        Parameters:
            phi_js: client feature vectors
            Phi: global feature vector

        Returns:
            client feature distribution imbalances
    """
    return np.asarray(list(map(lambda v_j: _feature_distribution_imbalance(v_j, Phi), phi_js)))


def _v_j(dataset: Dataset) -> np.ndarray:
    """Compute v_j for a single client"""
    labels = np.asarray([dataset[i][1] for i in range(len(dataset))])
    return labels.sum(axis=0)


def secure_aggregation(datasets: list[Dataset]) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the secure aggregation vectors v_j and V

        Parameters:
            datasets: list of client datasets

        Returns:
            client label vectors, global label vector
    """
    vjs = np.asarray(list(map(_v_j, datasets)))
    V = vjs.sum(axis=0)
    n = len(vjs[0])
    for vj in vjs:
        assert len(vj) == n
    return vjs, V


def _label_imbalance(v_j: np.ndarray) -> float:
    """Compute the label imbalance for a single client"""
    max_p = v_j.max()
    min_p = v_j.min()

    return max_p / min_p if min_p > 0 else max_p


def label_imbalances(vjs: np.ndarray) -> np.ndarray:
    """
    Compute the label imbalance per client

        Parameters:
            vjs: client label vectors

        Returns:
            client label imbalances
    """
    return np.asarray(list(map(_label_imbalance, vjs)))


def _label_distribution_imbalance(v_j: np.ndarray, V: np.ndarray) -> float:
    """Compute the label distribution imbalance for a single client"""
    return distance.cosine(v_j, V)


def label_distribution_imbalances(vjs: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute the label distribution imbalance per client

        Parameters:
            vjs: client label vectors
            V: global label vector

        Returns:
            client label distribution imbalances
    """
    return np.asarray(list(map(lambda v_j: _label_distribution_imbalance(v_j, V), vjs)))


def _quantity_imbalance(v_j: np.ndarray, N: int, J: int) -> float:
    """Compute the quantity imbalance for a single client"""
    N_j = v_j.sum()
    return (N_j * J) / N


def quantity_imbalances(vjs: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute the quantity imbalance per client

        Parameters:
            vjs: client label vectors
            V: global label vector

        Returns:
            client quantity imbalances
    """
    N = V.sum()
    J = vjs.shape[0]
    return np.asarray(list(map(lambda v_j: _quantity_imbalance(v_j, N, J), vjs)))


def li_ldi_qi(datasets: list[Dataset]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute label imbalance, label distribution imbalance and quantity imbalance for the given client datasets

        Parameters:
            datasets: list of client datasets

        Returns:
            label imbalances, label distribution imbalances, quantity imbalances
    """
    vjs, V = secure_aggregation(datasets)
    li = label_imbalances(vjs)
    ldi = label_distribution_imbalances(vjs, V)
    qi = quantity_imbalances(vjs, V)

    return li, ldi, qi


def fi_fdi(datasets: list[TransformingSubset]) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute feature imbalance and feature distribution imbalance for the given client datasets

        Parameters:
            datasets: list of client datasets

        Returns:
            feature imbalance, feature distribution imbalance
    """
    phi_js, Phi = secure_aggregation_features(datasets)
    fi = feature_imbalances(phi_js)
    fdi = feature_distribution_imbalances(phi_js, Phi)

    return fi, fdi


def get_clusters_for_lxo(n_clusters, *args):
    """
    Compute cluster identities for client contribution LXO via KMeans clustering

        Parameters:
            n_clusters: number of clusters
            *args: clustering features

        Returns:
            predicted labels for each client
    """
    client_vectors = np.asarray(args).transpose()  # create feature vectors with clients row-wise
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(client_vectors)
    return kmeans.labels_
