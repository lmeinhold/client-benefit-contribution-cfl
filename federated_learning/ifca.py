"""
Iterative Federated Clustering Algorithm
from "An Efficient Framework for Clustered Federated Learning" (Gosh et al., 2021)
"""

import numpy as np
import torch
from numpy import signedinteger
from torch.utils.data import DataLoader
from tqdm import tqdm

from federated_learning.base import FederatedLearningAlgorithm
from federated_learning.fedavg import FedAvgClient
from utils.torchutils import StateDict, average_state_dicts


class IFCA(FederatedLearningAlgorithm):
    """Iterative Federated Clustering Algorithm"""

    def __init__(self, client_data: list[DataLoader], model_fn, optimizer_fn, loss_fn, rounds: int, epochs: int, k: int,
                 alpha: float = 0.3, device: str = "cpu", test_data: DataLoader = None):
        """Create a new IFCA instance
            Parameters:
                client_data: list of DataLoaders holding the datasets for each client
                model_fn: a function that returns the model to use on each client
                optimizer_fn: a function that returns the optimizer to use on each client
                loss_fn: a loss function to use
                rounds: number of federated learning rounds
                epochs: number of epochs on each client per federated learning round
                k: number of clusters
                alpha: fraction of clients that are selected for each round
                device: the torch device to use for training
                test_data: a DataLoader with test datasets to evaluate the global models OR k DataLoaders to evaluate each
                    client on OR None if no test evaluation should be performed TODO: implement multiple test loaders
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
        self.cluster_models = [model_fn().to(self.device) for i in range(k)]

    def create_client(self, client_id, data_loader) -> "IfcaClient":
        return IfcaClient(client_id, data_loader, self.model_fn, self.optimizer_fn, self.loss_fn)

    def fit(self):
        for r in range(self.rounds):
            print(f"IFCA round {r + 1} --------------")
            self.train_round()
            if self.test_data is not None:
                self.test_round()

    def client_count(self):
        return len(self.clients)

    def train_round(self):
        cluster_weights = [self.cluster_models[i].state_dict() for i in range(self.k)]
        new_weights = []
        cluster_estimates = []
        num_clients = int(np.ceil(self.alpha * len(self.clients)))
        for c in tqdm(np.random.choice(self.clients, num_clients)):
            w_i, s_i = c.train_round(cluster_weights, self.epochs)
            new_weights.append(w_i)
            cluster_estimates.append(s_i)

        updated_weights = self.aggregate_weights(cluster_weights, new_weights, cluster_estimates)
        for i, w_i in enumerate(updated_weights):
            self.cluster_models[i].load_state_dict(w_i, strict=False)

    @torch.no_grad()
    def test_round(self):
        for j in range(self.k):
            model = self.cluster_models[j]
            test_loss, correct = 0, 0
            for X, y in self.test_data:
                X, y = X.to(self.device), y.to(self.device)
                pred = model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
            test_loss /= len(self.test_data)
            correct /= len(self.test_data.dataset)
            print(f"Cluster [{j}]: Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    def aggregate_weights(self, cluster_weights, weights, cluster_estimates) -> list[StateDict]:
        """Aggregate weights using model averaging (option II from the IFCA paper)"""
        updated_cluster_weights = [[] for _ in range(self.k)]
        for j in range(self.k):
            for i, estimate in enumerate(cluster_estimates):
                if cluster_estimates[i].argmax() == j:
                    theta_ij = weights[i]
                    updated_cluster_weights[j].append(theta_ij)

        aggregated_cluster_weights = [average_state_dicts(updated_cluster_weights[j]) for j in range(self.k)]
        return aggregated_cluster_weights


class IfcaClient(FedAvgClient):
    """Client for the IFCA algorithm. Similar to FedAvg, but performs cluster estimation while training."""

    def train_round(self, cluster_states: [StateDict], epochs: int) -> tuple[StateDict, np.array]:
        j_hat = self.estimate_cluster(cluster_states)

        s_i = np.zeros_like(cluster_states)
        s_i[j_hat] = 1

        model = self.build_model(cluster_states[j_hat])
        optimizer = self.build_optimizer(model)

        for t in range(epochs):
            size = len(self.data_loader)

            model.train()

            for batch, (X, y) in enumerate(self.data_loader):
                X = X.to(self.device)
                y = y.to(self.device)

                pred = model(X)
                loss = self.loss_fn(pred, y)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        return model.state_dict(), s_i

    @torch.no_grad()
    def estimate_cluster(self, cluster_states: list[StateDict]) -> signedinteger:
        losses = []
        for w_ij in cluster_states:
            model = self.build_model(w_ij)
            model.eval()
            loss = 0
            for X, y in self.data_loader:
                X = X.to(self.device)
                y = y.to(self.device)

                pred = model(X)
                loss += self.loss_fn(pred, y)
            losses.append(loss)

        losses = np.array(losses)
        return losses.argmin()
