"""
Iterative Federated Clustering Algorithm
from "An Efficient Framework for Clustered Federated Learning" (Gosh et al., 2021)
"""
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from federated_learning.base import FederatedLearningAlgorithm
from federated_learning.fedavg import FedAvgClient


class IFCA(FederatedLearningAlgorithm):
    """Iterative Federated Clustering Algorithm"""

    def __init__(self, client_data: list[DataLoader], model_fn, optimizer_fn, loss_fn, rounds: int, epochs: int, k: int,
                 alpha: float = 0.3, device: str = "cpu", test_data: DataLoader = None):
        """Create a new IFCA instance
            Parameters:
                client_data: list of DataLoaders holding the data for each client
                model_fn: a function that returns the model to use on each client
                optimizer_fn: a function that returns the optimizer to use on each client
                loss_fn: a loss function to use
                rounds: number of federated learning rounds
                epochs: number of epochs on each client per federated learning round
                k: number of clusters
                alpha: fraction of clients that are selected for each round
                device: the torch device to use for training
                test_data: a DataLoader with test data to evaluate the global models OR None if no test evaluation
                    should be performed
        """
        self.client_data = client_data
        self.model_fn = model_fn
        self.optimizer_fn = optimizer_fn
        self.loss_fn = loss_fn
        self.rounds = rounds
        self.epochs = epochs
        self.k = k
        self.alpha = alpha
        self.device = device
        self.test_data = test_data
        self.clients = [self.create_client(i, d) for i, d in enumerate(client_data)]
        self.cluster_models = dict([(i, model_fn().to(self.device)) for i in range(k)])

    def create_client(self, client_id, data_loader):
        return FedAvgClient(client_id, data_loader, self.model_fn, self.optimizer_fn, self.loss_fn)

    def fit(self):
        for r in range(self.rounds):
            print(f"IFCA round {r + 1} --------------")
            self.train_round()
            if self.test_data is not None:
                self.test_round()

    def client_count(self):
        return len(self.clients)

    def train_round(self):
        cluster_weights = dict([(i, self.cluster_models[i].state_dict()) for i in range(self.k)])
        weights = {}
        num_clients = int(np.ceil(self.alpha * len(self.clients)))
        for c in tqdm(np.random.choice(self.clients, num_clients)):
            w_i = c.train_round(global_weights, self.epochs)
            weights.append(w_i)

        updated_weights = self.aggregate_weights(global_weights, weights)

        self.global_model.load_state_dict(updated_weights, strict=False)

    def test_round(self):
        pass
