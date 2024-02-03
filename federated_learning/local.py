import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from federated_learning.base import FederatedLearningAlgorithm
from utils.results_writer import ResultsWriter


class LocalModels(FederatedLearningAlgorithm):
    def __init__(self, model_class, loss, optimizer, rounds: int, epochs: int, device="cpu"):
        self.model_class = model_class
        self.loss = loss
        self.optimizer = optimizer
        self.rounds = rounds
        self.epochs = epochs
        self.device = device
        self.results = ResultsWriter()

    def fit(self, train_data: list[DataLoader], test_data: list[DataLoader] = None) -> ResultsWriter:
        n_clients = len(train_data)

        shared_model = self.model_class().to(self.device)
        # shared_model = torch.compile(model=shared_model, mode="reduce-overhead")

        init_state = dict(shared_model.named_parameters())

        for k in tqdm(range(n_clients), desc="Clients", position=1):
            shared_model.load_state_dict(init_state, strict=False)  # reset state

            optimizer = self.optimizer(shared_model.parameters())

            for r in range(self.rounds):
                train_loss = 0
                for e in range(self.epochs):
                    train_loss += self._train_client_epoch(shared_model, train_data[k], optimizer)

                self.results.write(
                    round=r,
                    client=str(k),
                    stage="train",
                    loss=train_loss / self.epochs,
                    n_samples=len(train_data[k].dataset)
                )

                if test_data is not None:
                    test_loss, f1 = self._test_client_epoch(shared_model, test_data[k])
                    self.results.write(
                        round=r,
                        client=str(k),
                        stage="test",
                        loss=test_loss,
                        f1=f1,
                        n_samples=len(test_data[k].dataset)
                    )

        return self.results

    def _train_client_epoch(self, model, train_dataloader, optimizer) -> float:
        model.train()

        epoch_loss = 0

        for X, y in train_dataloader:
            X, y = X.to(self.device), y.to(self.device)

            pred = model(X)

            loss = self.loss(pred, y)
            epoch_loss += loss.cpu().item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        return epoch_loss / len(train_dataloader)

    def _test_client_epoch(self, model, test_dataloader) -> tuple[float, float]:
        epoch_loss = 0
        epoch_y_true = []
        epoch_y_pred = []

        model.eval()

        for X, y in test_dataloader:
            X, y = X.to(self.device), y.to(self.device)

            with torch.no_grad():
                pred = model(X)
                loss = self.loss(pred, y)
                epoch_loss += loss.cpu().item()

            epoch_y_pred.append(pred.argmax(1).detach().cpu())
            epoch_y_true.append(y.argmax(1).detach().cpu())

        epoch_loss /= len(test_dataloader.dataset)
        epoch_y_pred = np.concatenate(epoch_y_pred)
        epoch_y_true = np.concatenate(epoch_y_true)
        assert len(epoch_y_pred) == len(epoch_y_true)
        assert len(epoch_y_pred) == len(test_dataloader.dataset)

        return epoch_loss, f1_score(epoch_y_true, epoch_y_pred, average="macro")
