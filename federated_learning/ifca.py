"""
Iterative Federated Clustering Algorithm
from "An Efficient Framework for Clustered Federated Learning" (Gosh et al., 2021)
"""

import numpy as np
import torch
from numpy import signedinteger
from torch import tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from federated_learning.base import FederatedLearningAlgorithm
from federated_learning.fedavg import FedAvgClient
from utils.metrics_logging import Logger
from utils.torchutils import StateDict, average_state_dicts


class IFCA(FederatedLearningAlgorithm):
    """Iterative Federated Clustering Algorithm"""

    def __init__(self, client_data: list[DataLoader], model_fn, optimizer_fn, loss_fn, rounds: int, epochs: int, k: int,
                 logger: Logger, alpha: float = 0.3, device: str = "cpu", test_data: DataLoader = None):
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
        self.logger = logger
        self.device = device
        self.test_data = test_data
        if test_data is not None and isinstance(test_data, list):
            self.clients = [self.create_client(i, d, test_data[i]) for i, d in enumerate(client_data)]
        else:
            self.clients = [self.create_client(i, d) for i, d in enumerate(client_data)]
        self.cluster_models = [model_fn().to(self.device) for i in range(k)]

    def create_client(self, client_id, data_loader, test_dataloader: DataLoader = None) -> "IfcaClient":
        return IfcaClient(client_id, data_loader, self.model_fn, self.optimizer_fn, self.loss_fn, self.logger,
                          test_dataloader, self.device)

    def fit(self):
        for r in range(self.rounds):
            self.train_round(r+1)
            if self.test_data is not None and not isinstance(self.test_data, list):
                self.test_round()

    def client_count(self):
        return len(self.clients)

    def train_round(self, round):
        cluster_weights = [self.cluster_models[i].state_dict() for i in range(self.k)]
        new_weights = []
        cluster_estimates = []
        num_clients = int(np.ceil(self.alpha * len(self.clients)))

        losses = 0
        corrects = 0
        samples = 0

        for c in tqdm(np.random.choice(self.clients, num_clients)):
            w_i, s_i, test_loss, test_correct = c.train_round(cluster_weights, self.epochs, round)
            new_weights.append(w_i)
            cluster_estimates.append(s_i)
            if test_loss is not None:
                n = len(c.test_dataloader.dataset)
                losses += test_loss * n
                corrects += test_correct
                samples += n

        if samples > 0:
            weighted_client_loss = losses / samples / self.k
            weighted_client_accuracy = corrects / samples
            self.logger.log_server_metrics(round, stage="test", accuracy=weighted_client_accuracy,
                                           loss=weighted_client_loss)

        updated_weights = self.aggregate_weights(new_weights, cluster_estimates)
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

    def aggregate_weights(self, weights, cluster_estimates) -> list[StateDict]:
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

    def train_round(self, cluster_states: [StateDict], epochs: int, round: int):
        j_hat = self.estimate_cluster(cluster_states)

        s_i = np.zeros_like(cluster_states)
        s_i[j_hat] = 1

        model = self.build_model(cluster_states[j_hat])
        optimizer = self.build_optimizer(model)

        self.train_epochs(epochs, model, optimizer, round)

        test_loss = None
        test_correct = None

        if self.test_dataloader is not None:
            test_loss, test_correct = self.test_round(model, round)

        return model.state_dict(), s_i, test_loss, test_correct

    @torch.no_grad()
    def test_round(self, model, round):
        test_loss = 0
        test_correct = 0

        model.eval()
        for batch, (X, y) in enumerate(self.test_dataloader):
            X = X.to(self.device)
            y = y.to(self.device)

            pred = model(X)
            loss = self.loss_fn(pred, y)

            test_loss += loss.item()
            test_correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

        self.logger.log_client_metrics(str(self.client_id), stage="test", epoch=None, round=round, loss=test_loss,
                                       accuracy=test_correct/len(self.test_dataloader.dataset))

        return test_loss, test_correct

    def train_epochs(self, epochs, model, optimizer, round):
        train_loss = 0
        train_correct = 0

        for t in range(epochs):
            model.train()

            for batch, (X, y) in enumerate(self.data_loader):
                X = X.to(self.device)
                y = y.to(self.device)

                pred = model(X)
                loss = self.loss_fn(pred, y)

                train_loss += loss.item()
                train_correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            self.logger.log_client_metrics(str(self.client_id), round, t, stage="train",
                                           accuracy=train_correct / len(self.data_loader.dataset), loss=train_loss)

    @torch.no_grad()
    def estimate_cluster(self, cluster_states: list[StateDict]) -> signedinteger:
        losses = self.evaluate_losses(cluster_states)
        return losses.argmin()

    def evaluate_losses(self, cluster_states):
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
        losses = tensor(losses).detach().numpy()
        return losses
