import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from federated_learning.base import FederatedLearningAlgorithm, FederatedLearningClient
from utils.metrics_logging import Logger
from utils.torchutils import StateDict


class FedAvg(FederatedLearningAlgorithm):
    def __init__(self, client_data: list[DataLoader], model_fn, optimizer_fn, loss_fn, rounds: int, epochs: int,
                 alpha: float, logger: Logger, device: str = "cpu", test_data: DataLoader = None):
        self.logger = logger
        self.alpha = alpha
        self.device = device
        self.test_data = test_data
        self.model_fn = model_fn
        self.loss_fn = loss_fn
        self.optimizer_fn = optimizer_fn
        self.epochs = epochs
        self.rounds = rounds
        if test_data is not None and isinstance(test_data, list):
            self.clients = [self.create_client(i, d, test_data[i]) for i, d in enumerate(client_data)]
        else:
            self.clients = [self.create_client(i, d) for i, d in enumerate(client_data)]
        self.global_model = model_fn().to(self.device)

    def create_client(self, client_id, data_loader, test_data_loader=None):
        return FedAvgClient(client_id, data_loader, self.model_fn, self.optimizer_fn, self.loss_fn, self.logger,
                            test_data_loader, self.device)

    def train_round(self, round):
        global_weights = self.global_model.state_dict()
        weights = []
        n_c = int(np.ceil(self.alpha * len(self.clients)))
        losses = 0
        corrects = 0
        samples = 0
        for c in tqdm(np.random.choice(self.clients, n_c), position=1, desc="Clients", leave=False):
            w_i, loss, correct = c.train_round(global_weights, round, self.epochs)
            weights.append(w_i)
            if loss is not None:
                n = len(c.test_dataloader.dataset)
                losses += loss * n
                corrects += correct
                samples += n

        if samples > 0:
            weighted_client_loss = losses / samples
            weighted_client_accuracy = corrects / samples
            self.logger.log_server_metrics(round, stage="test", accuracy=weighted_client_accuracy,
                                           loss=weighted_client_loss)

        updated_weights = self.aggregate_weights(global_weights, weights)

        self.global_model.load_state_dict(updated_weights, strict=False)

    def test_round(self, round):
        model = self.global_model
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in self.test_data:
                X, y = X.to(self.device), y.to(self.device)
                pred = model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
            test_loss /= len(self.test_data)
            correct /= len(self.test_data.dataset)
            self.logger.log_server_metrics(round, stage="test", accuracy=correct, loss=test_loss)

    @staticmethod
    def aggregate_weights(global_weights, weights):
        updated_weights = {}
        for k in global_weights.keys():
            if k.endswith('.weight') or k.endswith('.bias'):
                client_weights = torch.stack([w_i[k] for w_i in weights])
                updated_weights[k] = client_weights.mean(dim=0)
        return updated_weights

    def fit(self):
        for r in tqdm(range(self.rounds), position=0, desc="Round"):
            self.train_round(r + 1)
            if self.test_data is not None and not isinstance(self.test_data, list):
                self.test_round(r + 1)

    def client_count(self):
        return len(self.clients)


class FedAvgClient(FederatedLearningClient):
    def __init__(self, client_id: int, data_loader: DataLoader, model_fn, optimizer_fn, loss_fn, logger: Logger,
                 test_dataloader: DataLoader = None, device="cpu"):
        self.client_id = client_id
        self.data_loader = data_loader
        self.model_fn = model_fn
        self.optimizer_fn = optimizer_fn
        self.loss_fn = loss_fn
        self.device = device
        self.logger = logger
        self.test_dataloader = test_dataloader

    def build_model(self, state_dict: dict[str, torch.Tensor]) -> torch.nn.Module:
        """build a model from given weights"""
        m = self.model_fn()
        m.load_state_dict(state_dict, strict=False)
        return m.to(self.device)

    def build_optimizer(self, model) -> torch.optim.Optimizer:
        return self.optimizer_fn(model.parameters())

    def train_round(self, shared_state: StateDict, round: int, epochs: int) -> StateDict:
        model = self.build_model(shared_state)
        optimizer = self.build_optimizer(model)

        for t in range(epochs):
            size = len(self.data_loader)

            model.train()

            overall_loss = 0
            correct = 0

            for batch, (X, y) in enumerate(self.data_loader):
                X = X.to(self.device)
                y = y.to(self.device)

                pred = model(X)
                loss = self.loss_fn(pred, y)
                overall_loss += loss.item()
                correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            overall_loss /= len(self.data_loader)
            accuracy = correct / len(self.data_loader.dataset)
            self.logger.log_client_metrics(client_id=str(self.client_id), epoch=t, round=round, stage="train",
                                           loss=overall_loss, accuracy=accuracy)

        test_loss = None
        correct = None

        if self.test_dataloader is not None:
            model.eval()

            overall_loss = 0
            correct = 0

            with torch.no_grad():
                for batch, (X, y) in enumerate(self.test_dataloader):
                    X = X.to(self.device)
                    y = y.to(self.device)

                    pred = model(X)
                    loss = self.loss_fn(pred, y)
                    overall_loss += loss.item()
                    correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

                test_loss = overall_loss / len(self.data_loader)
                test_accuracy = correct / len(self.test_dataloader.dataset)
                self.logger.log_client_metrics(client_id=str(self.client_id), epoch=None, round=round, stage="test",
                                               loss=test_loss, accuracy=test_accuracy)

        return model.state_dict(), test_loss, correct
