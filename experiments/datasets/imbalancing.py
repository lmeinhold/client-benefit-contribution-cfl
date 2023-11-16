import numpy as np
from torch.utils.data import Dataset, random_split, Subset

SEED = 42


def extract_raw_data(dataset: Dataset) -> tuple[np.ndarray, np.ndarray]:
    labels = np.array(dataset.targets)
    features = np.array(dataset.data)
    return features, labels


def split_dataset_equally(dataset: Dataset, n: int, seed=SEED):
    """Split a dataset into n parts of equal length"""
    np.random.seed(seed)
    return random_split(dataset=dataset, lengths=np.repeat(int(len(dataset) / n), n))


def split_with_quantity_skew(dataset: Dataset, alpha: float, n_clients: int, seed=SEED) -> list[Dataset]:
    """Split a dataset into n datasets with varying size following a dirichlet distribution"""
    n = len(dataset)

    np.random.seed(seed)
    indices = np.random.permutation(n)

    proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
    proportions /= proportions.sum()
    proportions = (np.cumsum(proportions) * n).astype(int)[:-1]

    batch_indices = np.split(indices, proportions)
    batch_indices = list(map(np.ndarray.tolist, batch_indices))

    apply_minimum_num_of_samples(batch_indices, n_clients)

    return [Subset(dataset, batch_indices[i]) for i, idx in enumerate(batch_indices)]


def split_with_label_distribution_skew(dataset: Dataset, alpha: float, n_clients: int, seed=SEED):
    np.random.seed(seed)

    features, labels = extract_raw_data(dataset)

    n = len(dataset)
    n_classes = np.unique(labels).shape[0]

    batch_indices = [[] for _ in range(n_clients)]

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


def apply_minimum_num_of_samples(batch_indices, n_clients, min: int = 5):
    largest_client_index = np.argmax([len(x) for x in batch_indices])
    for j in range(n_clients):
        if len(batch_indices[j]) < min:
            transfer = min - len(batch_indices[j])
            batch_indices[j].extend(batch_indices[largest_client_index][-transfer:])
            batch_indices[largest_client_index] = batch_indices[largest_client_index][:-transfer]