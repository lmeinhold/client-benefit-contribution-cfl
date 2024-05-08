import numpy as np
import torch
from torch.utils.data import Dataset, random_split, Subset, TensorDataset

from utils.torchutils import FixedRotationTransform, SingleTransformingSubset, PerSampleTransformingSubset, CustomSubset


def generator_with_seed(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def extract_raw_data(dataset: Dataset) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(dataset, TensorDataset):
        labels = np.asarray(dataset.tensors[1])
        features = np.asarray(dataset.tensors[0])
        return features, labels

    labels = np.array(dataset.targets)
    features = np.array(dataset.data)
    return features, labels


def split_dataset_equally(dataset: Dataset, n: int, seed: int, *args, **kwargs):
    """Split a dataset into n parts of equal length"""
    return random_split(dataset=dataset, lengths=np.repeat(int(len(dataset) / n), n),
                        generator=generator_with_seed(seed))


def split_with_quantity_skew(dataset: Dataset, n_clients: int, alpha: float = 1, seed: int = None, *args, **kwargs) -> \
        list[
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

    return [CustomSubset(dataset, batch_indices[i]) for i, idx in enumerate(batch_indices)]


def split_with_fixed_num_labels(dataset: Dataset, n_clients: int, c: int = 2, seed: int = None, *args, **kwargs) -> \
        list[Dataset]:
    """Split a dataset into n subsets with exactly c different labels per client"""
    _, labels = extract_raw_data(dataset)
    n = len(labels)
    classes = np.unique(labels).reshape(-1, 1)
    n_classes = classes.shape[0]
    if seed is not None:
        np.random.seed(seed)

    chosen_classes = [np.random.choice(np.arange(n_classes), c, replace=False) for _ in range(n_clients)]
    batch_indices = [[] for _ in range(n_clients)]
    for class_idx in np.arange(n_classes):
        clients_for_class = np.asarray([i for i in range(n_clients) if classes[class_idx] in chosen_classes[i]])
        class_idxs_in_dataset = np.asarray([i for i, l in enumerate(labels) if l == classes[class_idx]])
        np.random.shuffle(class_idxs_in_dataset)
        max_len = len(class_idxs_in_dataset) - (len(class_idxs_in_dataset) % len(clients_for_class))
        split_idxs = np.split(class_idxs_in_dataset[:max_len], len(clients_for_class))
        for client_idx, split_idx in zip(clients_for_class, split_idxs):
            batch_indices[client_idx] += split_idx.tolist()

    return [CustomSubset(dataset, indices) for indices in batch_indices]


def split_with_label_distribution_skew(dataset: Dataset, n_clients: int, alpha: float = 1, seed: int = None, *args,
                                       **kwargs):
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

    return [CustomSubset(dataset, indices) for indices in batch_indices]


def split_with_feature_distribution_skew(dataset: Dataset, n_clients: int, alpha: float = 1, seed: int = None, *args,
                                         **kwargs):
    features, labels = extract_raw_data(dataset)

    n = len(dataset)
    angles = [0, 90, 180, 270]
    transforms = list(map(FixedRotationTransform, angles))
    n_features = len(angles)

    indices = np.random.permutation(n)

    if seed is not None:
        np.random.seed(seed)
    proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
    proportions /= proportions.sum()
    proportions = (np.cumsum(proportions) * n).astype(int)[:-1]

    batch_indices = np.split(indices, proportions)
    batch_indices = list(map(np.ndarray.tolist, batch_indices))

    apply_minimum_num_of_samples(batch_indices, n_clients)

    group_proportions = np.random.dirichlet(np.repeat(alpha, n_features))
    batch_group_assignments = [np.random.choice(a=np.arange(n_features), size=len(batch), p=group_proportions) for batch
                               in batch_indices]

    return [PerSampleTransformingSubset(dataset, idxs, [transforms[g] for g in groups], groups) for idxs, groups in
            zip(batch_indices, batch_group_assignments)]


def split_with_transform_imbalance(dataset: Dataset, n_clients: int, transform_fn, n_transforms: int, alpha: float = 1,
                                   seed: int = None, *args, **kwargs) -> list[Dataset]:
    """Split a dataset into n parts of equal length, applying a transformation function to each client,
    createing a feature imbalance distribution dependend on alpha"""
    if seed is not None:
        np.random.seed(seed)

    group_proportions = np.random.dirichlet(np.repeat(alpha, n_transforms))
    client_group_assignments = np.random.choice(a=np.arange(n_transforms), size=n_clients, p=group_proportions)

    split_datasets = split_dataset_equally(dataset=dataset, n=n_clients, seed=seed)

    return [SingleTransformingSubset(ds.dataset, ds.indices, transform_fn(group_idx), group_idx) for group_idx, ds in
            zip(client_group_assignments, split_datasets)]


def split_with_rotation(dataset: Dataset, n_clients: int, alpha: float = 1, seed: int = None, *args, **kwargs) -> list[
    Dataset]:
    """Split an image dataset applying rotation according to a dirichlet distribution"""
    rotations = [0, 90, 180, 270]

    def transform_rotate(idx):
        degrees = rotations[idx]
        return FixedRotationTransform(degrees)

    return split_with_transform_imbalance(
        dataset=dataset,
        n_clients=n_clients,
        alpha=alpha,
        seed=seed,
        n_transforms=len(rotations),
        transform_fn=transform_rotate,
        *args,
        **kwargs
    )


def apply_minimum_num_of_samples(batch_indices, n_clients, min_size: int = 5):
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
