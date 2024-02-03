import numpy as np
import torch


def get_device() -> str:
    """Return a string representing the device to use for training"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


StateDict = dict[str, torch.Tensor]


def average_parameters(state_dicts: list[StateDict], include_biases=True, weights=None):
    """Average model parameters.
    If `weights` is provided, perform a weighted average"""
    weights = np.ones_like(state_dicts) if weights is None else weights
    updated_dict = {}
    for state_gen in state_dicts:
        state_dict = state_gen if isinstance(state_gen, dict) else dict(state_gen)
        for k in state_dict.keys():
            if k in updated_dict:
                updated_dict[k].append(state_dict[k])
            else:
                updated_dict[k] = [state_dict[k]]
    for k, v in updated_dict.items():
        #assert len(v) == len(
        #    weights), f"Length of state_dicts ({len(v)}) does not match length of weights ({len(weights)}) for key {k}, {v}"
        updated_dict[k] = torch.stack([v[i] * weights[i] for i in range(len(v))]).sum(dim=0) / weights.sum()
    return updated_dict
