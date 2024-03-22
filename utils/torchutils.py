import numpy as np
import torch


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
