import numpy as np
import torch
from tqdm.auto import tqdm

from utils.results_writer import ResultsWriter
from utils.torchutils import average_state_dicts, StateDict


class FedProx:
    def __init__(self,
                 model_class,
                 loss,
                 optimizer,
                 rounds: int,
                 epochs: int,
                 gamma: float = 1.0,
                 mu: float = 0.0,
                 device="cpu"):
        self.model_class = model_class
        self.loss = loss
        self.optimizer = optimizer
        self.rounds = rounds
        self.epochs = epochs
        self.gamma = gamma
        self.mu = mu
        self.device = device
        self.results = ResultsWriter()

    def fit(self, train_data, test_data):
        n_clients = len(train_data)

        if isinstance(test_data, list) and len(test_data) != n_clients:
            raise Exception(f"Test data must be either of length 1 or the same length as the training data")

        clients_per_round = int(np.floor(self.gamma * n_clients))

        model = self.model_class().to(self.device)
        model = torch.compile(model=model, mode="reduce-overhead")
        global_weights = model.named_parameters()

        for t in tqdm(np.arange(self.rounds) + 1, desc="Round", position=0):
            updated_weights = []

            chosen_client_indices = np.random.choice(np.arange(n_clients), size=clients_per_round)

            for k in tqdm(np.arange(n_clients), desc="Client", position=1, leave=False):
                if k in chosen_client_indices:
                    client_train_data = train_data[k]
                    client_test_data = test_data[k] if isinstance(test_data, list) else test_data

                    client_weights, train_loss = self._train_client_round(global_weights, model, client_train_data)

                    test_loss, test_accuracy = self._test_client_round(model, client_test_data)

                    self.results.write(
                        round=t,
                        client=str(k),
                        stage="train",
                        loss=train_loss.mean()
                    ).write(
                        round=t,
                        client=str(k),
                        stage="test",
                        loss=test_loss,
                        accuracy=test_accuracy
                    )

            global_weights = average_state_dicts(updated_weights)

        return self.results

    def _train_client_round(self, global_weights, model, client_train_data) -> tuple[StateDict, np.ndarray]:
        model.load_state_dict(dict(global_weights), strict=False)
        optimizer = self.optimizer(model.parameters())

        model.train()

        round_train_losses = []

        for e in range(self.epochs):
            round_train_loss = self._train_epoch(model, optimizer, client_train_data, global_weights)
            round_train_losses.append(round_train_loss)

        return model.named_parameters(), np.array(round_train_losses)

    def _test_client_round(self, model, client_test_data):
        n_samples = len(client_test_data.dataset)

        round_loss = 0
        round_correct = 0

        model.eval()

        for X, y in client_test_data:
            X, y = X.to(self.device), y.to(self.device)

            with torch.no_grad():
                pred = model(X)

                batch_loss = self.loss(pred, y)
                round_loss += batch_loss.cpu().item()

            batch_correct = (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().cpu().item()
            round_correct += batch_correct

        round_loss /= len(client_test_data)
        round_accuracy = round_correct / n_samples

        return round_loss, round_accuracy

    def _train_epoch(self, model, optimizer, client_train_data, global_weights):
        epoch_loss = 0

        for X, y in client_train_data:
            X, y = X.to(self.device), y.to(self.device)

            pred = model(X)
            # calculate proximal term only if mu != 0 to speed up FedAvg
            loss = self._proximal_loss(pred, y, global_weights,
                                       model.named_parameters()) if self.mu != 0 else self.loss(pred, y)
            epoch_loss += loss.cpu().item()

            loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()

        return epoch_loss / len(client_train_data)

    def _proximal_loss(self, pred, y, old_state, new_state):
        """Compute the loss including the FedProx proximal term"""

        old_state = dict(old_state)
        new_state = dict(new_state)

        return self.loss(pred, y) + self._proximal_term(old_state, new_state)

    @torch.compile(mode="reduce-overhead")
    def _proximal_term(self, old_state, new_state):
        proximal_term = 0
        for k in old_state.keys():
            proximal_term += (new_state[k] - old_state[k]).norm(2)

        proximal_term = (self.mu / 2.0) * proximal_term
        # self.results.write(proximal_term=proximal_term)

        return proximal_term
