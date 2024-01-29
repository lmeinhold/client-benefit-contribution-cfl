import abc

from utils.torchutils import StateDict


class FederatedLearningAlgorithm(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(self):
        """Fit a federated model"""
        pass

    @abc.abstractmethod
    def client_count(self):
        """Number of clients in the federated model"""
        pass


class FederatedLearningClient(metaclass=abc.ABCMeta):
    pass
