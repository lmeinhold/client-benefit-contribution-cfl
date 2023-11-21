import logging

import torch


def get_device() -> str:
    """Return a string representing the device to use for training"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device '{device}'")
    return device


StateDict = dict[str, torch.Tensor]


def average_state_dicts(state_dicts: list[StateDict], include_biases=True):
    updated_dict = {}
    for state_dict in state_dicts:
        for k in state_dict.keys():
            if k.endswith('.weight') or (include_biases and k.endswith('.bias')):
                if k in updated_dict:
                    updated_dict[k].append(state_dict[k])
                else:
                    updated_dict[k] = [state_dict[k]]
    for k, v in updated_dict.items():
        updated_dict[k] = torch.stack(v).mean(dim=0)
    return updated_dict


def get_weights(model, biases=True) -> StateDict:
    return {k: v for k, v in model.state_dict().items() if k.endswith('.weight') or (biases and k.endswith('.bias'))}
