import abc
from functools import cache
from typing import List, Sequence, Callable

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Subset
from torch.utils.data.dataset import T_co, Dataset


def get_device() -> str:
    """Return a string representing the device to use for training"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


StateDict = dict[str, torch.Tensor]


def average_parameters(state_dicts: list[StateDict], weights=None):
    """Average model parameters.
    If `weights` is provided, perform a weighted average"""
    weights = np.ones_like(state_dicts) if weights is None else weights
    aggregated_state_dicts = collect_state_dicts(state_dicts)
    return dict([(key, tensor_weighted_mean(tensors, weights)) for key, tensors in aggregated_state_dicts.items()])


def tensor_weighted_mean(tensors, weights):
    """Calculate the weighted mean of a list of tensors"""
    assert len(tensors) == len(weights), \
        f"Length of state_dicts ({len(tensors)}) does not match length of weights ({len(weights)})"
    weight_sum = weights.sum()
    return torch.stack([tensors[i] * weights[i] for i in range(len(tensors))]).sum(dim=0).div(weight_sum)


def collect_state_dicts(state_dicts: list[StateDict]) -> dict[str, list[torch.Tensor]]:
    """Turn a list of state dicts into a dict of lists of weights"""
    aggregated_state_dict = {}
    for state_dict in state_dicts:
        for k in state_dict.keys():
            if k in aggregated_state_dict:
                aggregated_state_dict[k].append(state_dict[k])
            else:
                aggregated_state_dict[k] = [state_dict[k]]
    return aggregated_state_dict


class FixedRotationTransform:
    """Apply a fixed angle rotation to all images in the dataset"""

    def __init__(self, angle: int | float):
        self.angle = angle

    def __call__(self, image):
        return TF.rotate(image, self.angle)


class CustomSubset(Subset):
    """Subset subclass with additional functionality"""

    @property
    def targets(self):
        return [self.dataset.targets[i] for i in self.indices]


class TransformingSubset(CustomSubset, metaclass=abc.ABCMeta):
    """Abstract base class for subsets that apply a transformation."""

    @property
    @abc.abstractmethod
    def features(self) -> np.ndarray:
        """Return a numpy array that represents the transformations for each subset index"""
        raise NotImplementedError()


class SingleTransformingSubset(TransformingSubset):
    """A subset that applies a transform to each item in the subset"""

    @property
    def features(self) -> np.ndarray:
        return np.full_like(self.indices, self.transform_id)

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int], transform: Callable, transform_id):
        super().__init__(dataset, indices)
        self.transform = transform
        self.transform_id = transform_id

    def __getitem__(self, idx) -> T_co:
        item = super().__getitem__(idx)
        return self.transform(item[0]), item[1]

    def __getitems__(self, indices: List[int]) -> List[T_co]:
        items = super().__getitems__(indices)
        return [(self.transform(it[0]), it[1]) for it in items]


class PerSampleTransformingSubset(TransformingSubset):
    """A subset that applies a different transform to each item in the subset"""

    @property
    def features(self) -> np.ndarray:
        return self.transform_ids

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int], transforms: list[Callable],
                 transform_ids: np.ndarray):
        super().__init__(dataset, indices)
        assert len(transforms) == len(indices), "Must provide a transform for every item in the subset"
        self.transforms = transforms
        self.transform_ids = transform_ids

    @cache
    def transform_sample(self, index, item):
        return self.transforms[index](item)

    def __getitem__(self, idx) -> T_co:
        item = super().__getitem__(idx)
        return self.transform_sample(idx, item[0]), item[1]

    def __getitems__(self, indices: List[int]) -> List[T_co]:
        items = super().__getitems__(indices)
        return [(self.transform_sample(idx, it[0]), it[1]) for idx, it in zip(indices, items)]
