import warnings

import numpy as np
import torch
from sklearn.metrics import f1_score
from tqdm.auto import tqdm

from utils.results_writer import ResultsWriter
from utils.torchutils import average_parameters, StateDict


class FedProx:
    def __init__(self,
                 model_class,
                 loss,
                 optimizer_fn,
                 rounds: int,
                 epochs: int,
                 clients_per_round: float = 1.0,
                 mu: float = 0.0,
                 device="cpu"):
        self.model_class = model_class
        self.loss = loss
        self.optimizer_fn = optimizer_fn
        self.rounds = rounds
        self.epochs = epochs
        self.clients_per_round = clients_per_round
        self.mu = mu
        self.device = device
        self.results = ResultsWriter()

    def fit(self, train_data, test_data):
        n_clients = len(train_data)
        client_data_lengths = np.asarray([len(dl.dataset) for dl in train_data], dtype=np.double)

        if isinstance(test_data, list) and len(test_data) != n_clients:
            raise Exception(f"Test data must be either of length 1 or the same length as the training data")

        eff_clients_per_round = int(np.floor(self.clients_per_round * n_clients))
        print(f"Clients per round: {eff_clients_per_round}")

        model = self.model_class()
        # model = torch.compile(model=model, mode="reduce-overhead", dynamic=True)
        global_weights = dict(model.named_parameters())
        del model
        client_models = [self.model_class().to(self.device) for _ in range(n_clients)]
        optimizers = [self.optimizer_fn(m.parameters()) for m in client_models]

        for t in tqdm(np.arange(self.rounds), desc="Round", position=0):
            updated_weights = []

            chosen_client_indices = np.random.choice(np.arange(n_clients), size=eff_clients_per_round, replace=False)

            for k in tqdm(np.arange(n_clients), desc="Client", position=1, leave=False):
                if k in chosen_client_indices:
                    client_train_data = train_data[k]
                    client_test_data = test_data[k] if isinstance(test_data, list) else test_data

                    client_weights, train_loss = self._train_client_round(global_weights, client_models[k],
                                                                          optimizers[k], client_train_data)

                    test_loss, f1 = self._test_client_round(client_models[k], client_test_data)

                    if test_loss is None or np.isnan(test_loss):
                        warnings.warn(f"Test loss is undefined for client {k} in round {t}")
                    if f1 is None or np.isnan(f1):
                        warnings.warn(f"F1 is undefined for client {k} in round {t}")

                    self.results.write(
                        round=t,
                        client=str(k),
                        stage="train",
                        loss=train_loss.mean(),
                        n_samples=len(client_train_data.dataset),
                    ).write(
                        round=t,
                        client=str(k),
                        stage="test",
                        loss=test_loss,
                        f1=f1,
                        n_samples=len(client_test_data.dataset),
                    )

                    updated_weights.append(client_weights)

            global_weights = average_parameters(updated_weights, client_data_lengths[chosen_client_indices])

        return self.results

    def _train_client_round(self, global_weights, model, optimizer, client_train_data) -> tuple[StateDict, np.ndarray]:
        model.load_state_dict(dict(global_weights), strict=False)
        global_params = model.parameters()

        model.train()

        round_train_losses = []
        for e in range(self.epochs):
            round_train_loss = self._train_epoch(model, optimizer, client_train_data, global_params)
            round_train_losses.append(round_train_loss)

        return dict(model.named_parameters()), np.array(round_train_losses)

    def _test_client_round(self, model, client_test_data):
        round_loss = 0
        round_y_pred, round_y_true = [], []

        model.eval()

        for X, y in client_test_data:
            X, y = X.to(self.device), y.to(self.device)

            with torch.no_grad():
                pred = model(X)

                batch_loss = self.loss(pred, y)
                round_loss += batch_loss.cpu().item()

            round_y_pred.append(pred.argmax(1).detach().cpu())
            round_y_true.append(y.argmax(1).detach().cpu())

        round_loss /= len(client_test_data)
        round_y_pred = np.concatenate(round_y_pred)
        round_y_true = np.concatenate(round_y_true)
        assert len(round_y_pred) == len(round_y_true)
        assert len(round_y_pred) == len(client_test_data.dataset)

        return round_loss, f1_score(round_y_true, round_y_pred, average='macro')

    def _train_epoch(self, model, optimizer, client_train_data, global_parameters):
        epoch_loss = 0

        proximal_loss_term = 0 if self.mu == 0 else self._proximal_term(global_parameters, model.parameters())

        for X, y in client_train_data:
            X, y = X.to(self.device), y.to(self.device)

            pred = model(X)
            # calculate proximal term only if mu != 0 to speed up FedAvg
            loss = self.loss(pred, y) + proximal_loss_term
            epoch_loss += loss.cpu().item()

            loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()

        return epoch_loss / len(client_train_data)

    @torch.compile(mode="reduce-overhead")
    def _proximal_term(self, old_state, new_state):
        proximal_loss = 0
        for w, w_t in zip(old_state, new_state):
            proximal_loss += (w - w_t).norm(2)

        return (self.mu / 2.0) * proximal_loss
