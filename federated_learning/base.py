import abc

from utils.results_writer import ResultsWriter
from utils.torchutils import StateDict


class FederatedLearningAlgorithm(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(self, train_data, test_data) -> ResultsWriter:
        """Fit a federated model"""
        pass
