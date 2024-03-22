import abc
from typing import Type

import numpy as np
import torch
from torch import nn

from utils.results_writer import ResultsWriter


class FederatedLearningAlgorithm(metaclass=abc.ABCMeta):
    """Common base class for all federated learning algorithms
    Arguments:
        model_class: the class of the model that is used by all clients or a function that evaluates to a model
        loss_fn: the loss function to be used (not including special terms/penalties used by the algorithms)
        optimizer_fn: a function that returns an optimizer given the `model.parameters()`
        rounds: number of federated learning rounds
        epochs: number of local model epochs per round
        clients_per_round: a fraction of clients to use per round [default: all clients]
        device: device to train the model on"""
    def __init__(self,
                 model_class: Type[nn.Module],
                 loss_fn: Type[nn.Module],
                 optimizer_fn,
                 rounds: int,
                 epochs: int,
                 clients_per_round: float = 1.0,
                 device="cpu"):
        self.model_class = model_class
        self.loss_fn = loss_fn
        self.optimizer_fn = optimizer_fn
        self.rounds = rounds
        self.epochs = epochs
        self.clients_per_round = clients_per_round
        self.device = device
        self.results = ResultsWriter()

    @abc.abstractmethod
    def fit(self, train_data, test_data) -> ResultsWriter:
        """Fit the federated model, returning metrics"""
        raise NotImplementedError()

    def effective_clients_per_round(self, n_clients: int) -> int:
        """Calculate the effective (integer) number of clients that participate in a round,
        based on the provided fraction"""
        return int(np.floor(self.clients_per_round * n_clients))

    @staticmethod
    def choose_clients_for_round(n_clients: int, clients_per_round: int) -> np.ndarray:
        """Choose a random subset of clients that will participate in the round"""
        return np.random.choice(np.arange(n_clients), size=clients_per_round, replace=False)
