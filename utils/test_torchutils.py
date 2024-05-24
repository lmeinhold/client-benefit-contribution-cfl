import numpy as np
import torch
from torch import Tensor

from utils.torchutils import tensor_weighted_mean


def tensors_equal(a: Tensor, b: Tensor, eps: float = 1e-12) -> bool:
    """
    Check if two tensors are equal, element-wise.

        Parameters:
            a: First tensor
            b: Second tensor
            eps: Tolerance for float comparison

        Returns:
            true if the two tensors elements are approximately equal, otherwise false.
    """
    return torch.all(torch.lt(torch.abs(torch.add(a, -b)), eps)).item()


def test_tensor_weighted_mean_equal():
    a = Tensor([1, 2, 3, 4])
    b = Tensor([2, 4, 6, 8])
    weights = np.ones(2)
    expected = Tensor([1.5, 3, 4.5, 6])
    res = tensor_weighted_mean([a, b], weights)
    assert tensors_equal(expected, res)


def test_tensor_weighted_mean_weighted():
    a = Tensor([1, 2, 3, 4])
    b = Tensor([2, 4, 6, 8])
    weights = np.array([1, 2])
    expected = (a * 1 + b * 2) / 3
    res = tensor_weighted_mean([a, b], weights)
    assert tensors_equal(expected, res)
