import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from federated_learning.base import FederatedLearningAlgorithm, FederatedLearningClient
from utils.metrics_logging import Logger


class LocalModels(FederatedLearningAlgorithm):
    """Train a local model on each client, without any federation. Used to evaluate client benefit.
    Arguments:
        client_data: List of client datasets as DataLoaders
        model_fn: a function that instantiates a model
        optimizer_fn: a function that instantiates an optimizer from model parameters
        loss_fn: loss function to use for training
        epochs: number of epochs to run on each client
        logger: a data logger to log accuracy, loss, etc.
        device: the device to train model on
        test_data: test data as a data loader OR None if no test data should be evaluated"""

    def __init__(self, client_data: list[DataLoader], model_fn, optimizer_fn, loss_fn, epochs: int,
                 logger: Logger, device: str = "cpu", test_data: DataLoader = None):
        self.logger = logger
        self.device = device
        self.test_data = test_data
        self.model_fn = model_fn
        self.loss_fn = loss_fn
        self.optimizer_fn = optimizer_fn
        self.epochs = epochs
        self.client_data = client_data

    def create_client(self, client_id, data_loader, model):
        return LocalClient(client_id, data_loader, model, self.optimizer_fn, self.loss_fn, self.logger,
                           self.device)

    def fit(self):
        model = self.model_fn().to(self.device)
        model = torch.compile(model=model, mode="reduce-overhead")
        init_state = model.state_dict()

        for i in tqdm(range(len(self.client_data)), desc="Clients"):
            client = self.create_client(i, self.client_data[i], model)
            client.train(self.epochs)
            if self.test_data is not None:
                if isinstance(self.test_data, list):
                    data = self.test_data[i]
                else:
                    data = self.test_data
                client.test(data)

            # reset model parameters for next iter
            model.load_state_dict(init_state, strict=True)

    def client_count(self):
        return len(self.client_data)


class LocalClient(FederatedLearningClient):
    def __init__(self, client_id: int, data_loader, model, optimizer_fn, loss_fn, logger: Logger, device="cpu"):
        self.client_id = client_id
        self.data_loader = data_loader
        self.model = model
        self.optimizer = optimizer_fn(self.model.parameters())
        self.loss_fn = loss_fn.to(device)
        self.device = device
        self.logger = logger

    @torch.compile(mode="reduce-overhead")
    def train(self, epochs: int):
        for t in range(epochs):
            size = len(self.data_loader)

            self.model.train()

            overall_loss = 0
            correct = 0

            for batch, (X, y) in enumerate(self.data_loader):
                X = X.to(self.device)
                y = y.to(self.device)

                pred = self.model(X)
                loss = self.loss_fn(pred, y)
                overall_loss += loss.item()
                correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            overall_loss /= len(self.data_loader)
            accuracy = correct / len(self.data_loader.dataset)
            self.logger.log_client_metrics(client_id=str(self.client_id), epoch=t, round=None, stage="train",
                                           loss=overall_loss, accuracy=accuracy)

    def test(self, test_data: DataLoader):
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in test_data:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
            test_loss /= len(test_data)
            correct /= len(test_data.dataset)
            self.logger.log_client_metrics(client_id=str(self.client_id), stage="test", accuracy=correct,
                                           loss=test_loss)
