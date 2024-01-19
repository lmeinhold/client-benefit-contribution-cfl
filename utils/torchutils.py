import numpy as np
import torch


def get_device() -> str:
    """Return a string representing the device to use for training"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device '{device}'")
    return device


StateDict = dict[str, torch.Tensor]


def average_state_dicts(state_dicts: list[StateDict], include_biases=True, weights=None):
    """Average weights (and biases) in a state dict
    If weights is provided, perform a weighted average"""
    weights = np.ones_like(state_dicts) if weights is None else weights
    updated_dict = {}
    for state_gen in state_dicts:
        state_dict = state_gen if isinstance(state_gen, dict) else dict(state_gen)
        for k in state_dict.keys():
            if k.endswith('.weight') or (include_biases and k.endswith('.bias')):
                if k in updated_dict:
                    updated_dict[k].append(state_dict[k])
                else:
                    updated_dict[k] = [state_dict[k]]
    for k, v in updated_dict.items():
        assert len(v) == len(weights), f"Length of state_dicts does not match length of weights for key {k}"
        updated_dict[k] = torch.stack([v[i] * weights[i] for i in range(len(v))]).sum(dim=0) / weights.sum()
    return updated_dict


def get_weights(model, biases=True) -> StateDict:
    return {k: v for k, v in model.state_dict().items() if k.endswith('.weight') or (biases and k.endswith('.bias'))}
