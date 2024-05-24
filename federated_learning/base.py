import abc

import numpy as np
from torch.utils.data import DataLoader

from utils.results_writer import ResultsWriter


class FederatedLearningAlgorithm(metaclass=abc.ABCMeta):
    """Common base class for all federated learning algorithms
    Parameters:
        model_class: the class of the model that is used by all clients or a function that evaluates to a model
        loss_fn: the loss function to be used (not including special terms/penalties used by the algorithms)
        optimizer_fn: a function that returns an optimizer given the `model.parameters()`
        rounds: number of federated learning rounds
        epochs: number of local model epochs per round
        clients_per_round: a fraction of clients to use per round [default: all clients]
        binary: whether to perform binary instead of multiclass classification [default: False]
        device: device to train the model on"""

    def __init__(self,
                 model_class,
                 loss_fn,
                 optimizer_fn,
                 rounds: int,
                 epochs: int,
                 clients_per_round: float = 1.0,
                 binary: bool = False,
                 device="cpu"):
        self.model_class = model_class
        self.loss_fn = loss_fn
        self.optimizer_fn = optimizer_fn
        self.rounds = rounds
        self.epochs = epochs
        self.clients_per_round = clients_per_round
        self.device = device
        self.binary = binary
        self.results = ResultsWriter()

    @abc.abstractmethod
    def fit(self, train_data: list[DataLoader], test_data: list[DataLoader]) -> ResultsWriter:
        """
        Fit the federated model

            Parameters:
                train_data: the training data as a list of torch DataLoaders
                test_data: the test data as a list of torch DataLoaders

            Returns:
                training and model performance metrics
        """
        raise NotImplementedError()

    def effective_clients_per_round(self, n_clients: int) -> int:
        """
        Calculate the effective (integer) number of clients that participate in a round,
        based on the provided fraction

            Parameters:
                n_clients: the number of client datasets

            Returns:
                the absolute number of clients that participate in a round
        """
        return int(np.floor(self.clients_per_round * n_clients))

    @staticmethod
    def choose_clients_for_round(n_clients: int, clients_per_round: int) -> np.ndarray:
        """
        Choose a random subset of clients that will participate in the round

            Parameters:
                n_clients: the number of clients
                clients_per_round: the absolute number of client datasets

            Returns:
                an array of client indices that were chosen to participate in the round
        """
        return np.random.choice(np.arange(n_clients), size=clients_per_round, replace=False)
