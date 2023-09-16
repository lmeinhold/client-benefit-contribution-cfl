import logging

import torch


def get_device() -> str:
    """Return a string representing the device to use for training"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.debug(f"Using device '{device}'")
    return device


StateDict = dict[str, torch.Tensor]
