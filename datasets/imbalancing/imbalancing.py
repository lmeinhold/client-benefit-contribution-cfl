import numpy as np
import torch
from torch.utils.data import Dataset, random_split, Subset


def generator_with_seed(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def extract_raw_data(dataset: Dataset) -> tuple[np.ndarray, np.ndarray]:
    labels = np.array(dataset.targets)
    features = np.array(dataset.data)
    return features, labels


def split_dataset_equally(dataset: Dataset, n: int, seed: int, *args, **kwargs):
    """Split a dataset into n parts of equal length"""
    return random_split(dataset=dataset, lengths=np.repeat(int(len(dataset) / n), n),
                        generator=generator_with_seed(seed))


def split_with_quantity_skew(dataset: Dataset, n_clients: int, alpha: float = 1, seed: int = None, *args, **kwargs) -> list[
    Dataset]:
    """Split a dataset into n datasets with varying size following a dirichlet distribution"""
    n = len(dataset)

    indices = np.random.permutation(n)

    if seed is not None:
        np.random.seed(seed)
    proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
    proportions /= proportions.sum()
    proportions = (np.cumsum(proportions) * n).astype(int)[:-1]

    batch_indices = np.split(indices, proportions)
    batch_indices = list(map(np.ndarray.tolist, batch_indices))

    apply_minimum_num_of_samples(batch_indices, n_clients)

    return [Subset(dataset, batch_indices[i]) for i, idx in enumerate(batch_indices)]


def split_with_label_distribution_skew(dataset: Dataset, n_clients: int, alpha: float = 1, seed: int = None, *args, **kwargs):
    features, labels = extract_raw_data(dataset)

    n = len(dataset)
    n_classes = np.unique(labels).shape[0]

    batch_indices = [[] for _ in range(n_clients)]

    if seed is not None:
        np.random.seed(seed)
    # iterate all classes
    for idx_class in range(n_classes):
        idx_k = np.where(labels == idx_class)[0]

        proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
        proportions = np.array([p * (len(idx_j) < (n / n_clients)) for p, idx_j in zip(proportions, batch_indices)])
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

        batch_indices = [idx_j + idx.tolist() for idx_j, idx in zip(batch_indices, np.split(idx_k, proportions))]

    apply_minimum_num_of_samples(batch_indices, n_clients)

    return [Subset(dataset, indices) for indices in batch_indices]


def apply_minimum_num_of_samples(batch_indices, n_clients, min_size: int = 7):
    largest_client_index = np.argmax([len(x) for x in batch_indices])
    for j in range(n_clients):
        if len(batch_indices[j]) < min_size:
            transfer = min_size - len(batch_indices[j])
            batch_indices[j].extend(batch_indices[largest_client_index][-transfer:])
            batch_indices[largest_client_index] = batch_indices[largest_client_index][:-transfer]


def train_test_split(datasets, p_test=0.2, seed: int = 42):
    """Split a list of datasets into a list of train datasets and a list of test datasets"""

    train_sets = []
    test_sets = []
    for ds in datasets:
        split = random_split(ds, [(1 - p_test), p_test], generator=generator_with_seed(seed))
        train_sets.append(split[0])
        test_sets.append(split[1])
    return train_sets, test_sets
