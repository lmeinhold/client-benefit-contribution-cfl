import warnings

import numpy as np
import torch
from sklearn.metrics import f1_score
from tqdm.auto import tqdm

from federated_learning.base import FederatedLearningAlgorithm
from utils.results_writer import join_cluster_identities
from utils.torchutils import average_parameters, StateDict


def _fuse_model_weights(global_weights: list[StateDict], cluster_identities):
    return average_parameters([global_weights[c] for c in cluster_identities])


class FLSC(FederatedLearningAlgorithm):
    """
    Federate Learning with Soft-Clustering (FLSC). From Li et al., 2022: Federated Learning with Soft-Clustering.

        Parameters:
            model_class: the class of the model that is used by all clients or a function that evaluates to a model
            optimizer_fn: a function that returns an optimizer given the `model.parameters()`
            rounds: number of federated learning rounds
            epochs: number of local model epochs per round
            n_clusters: number of clusters to use
            loss_fn: the loss function to be used (not including special terms/penalties used by the algorithms)
            clusters_per_client: number of clusters per client
            clients_per_round: a fraction of clients to use per round [default: all clients]
            device: device to train the model on
    """

    def __init__(self, model_class, optimizer_fn, rounds: int, epochs: int, n_clusters: int,
                 loss_fn, clusters_per_client: int = 1, clients_per_round: float = 1.0,
                 device="cpu"):
        super().__init__(model_class, loss_fn, optimizer_fn, rounds, epochs, clients_per_round, False, device)
        self.n_clusters = n_clusters
        self.clusters_per_client = clusters_per_client

    def fit(self, train_data, test_data):
        n_clients = len(train_data)
        train_dataset_sizes = np.asarray([len(dl.dataset) for dl in train_data], dtype=np.double)

        if isinstance(test_data, list) and len(test_data) != n_clients:
            raise Exception(f"Test data must be either of length 1 or the same length as the training data")

        eff_clients_per_round = self.effective_clients_per_round(n_clients)

        # initial model weights
        global_weights = [dict(self.model_class().named_parameters()) for _ in
                          range(self.n_clusters)]  # copy global weights before models are (re-)used
        client_models = [self.model_class().to(self.device) for _ in range(n_clients)]
        optimizers = [self.optimizer_fn(m.parameters()) for m in client_models]

        # inital cluster identities
        cluster_identities = np.random.choice(np.arange(self.n_clusters), size=[n_clients, self.clusters_per_client])

        for t in tqdm(np.arange(self.rounds), desc="Round"):
            updated_weights = []

            chosen_client_indices = self.choose_clients_for_round(n_clients, eff_clients_per_round)

            for k in np.arange(n_clients):
                if k in chosen_client_indices:
                    client_train_data = train_data[k]
                    client_test_data = test_data[k] if isinstance(test_data, list) else test_data

                    if t > 0:
                        cluster_identities[k] = self._update_cluster_identity_estimates(global_weights,
                                                                                        client_models[k],
                                                                                        client_train_data)

                    client_weights, train_loss = self._train_client_round(global_weights, client_models[k],
                                                                          optimizers[k], cluster_identities[k],
                                                                          client_train_data)

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
                        info=dict(cluster_identities=join_cluster_identities(cluster_identities[k].tolist())),
                        n_samples=len(client_train_data.dataset)
                    ).write(
                        round=t,
                        client=str(k),
                        stage="test",
                        loss=test_loss,
                        f1=f1,
                        info=dict(cluster_identities=join_cluster_identities(cluster_identities[k].tolist())),
                        n_samples=len(client_test_data.dataset)
                    )

                    updated_weights.append(client_weights)

            global_weights = self._aggregate_cluster_weights(global_weights, updated_weights, chosen_client_indices,
                                                             train_dataset_sizes, cluster_identities)

        return self.results

    def _train_client_round(self, global_weights: list[StateDict], model, optimizer, cluster_identities,
                            client_train_data) -> \
            tuple[
                StateDict, np.ndarray]:

        fused_weights = _fuse_model_weights(global_weights, cluster_identities)
        model.load_state_dict(fused_weights, strict=False)

        model.train()

        round_train_losses = []

        for e in range(self.epochs):
            round_train_loss = self._train_epoch(model, optimizer, client_train_data)
            round_train_losses.append(round_train_loss)

        return dict(model.named_parameters()), np.array(round_train_losses)

    def _update_cluster_identity_estimates(self, global_weights, model, client_train_data) -> np.ndarray:
        cluster_losses = []
        for c in range(self.n_clusters):
            model.load_state_dict(dict(global_weights[c]), strict=False)
            model.eval()

            loss = 0
            for X, y in client_train_data:
                X, y = X.to(self.device), y.to(self.device)

                with torch.no_grad():
                    pred = model(X)
                    loss += self.loss_fn(pred, y).cpu().item()

            cluster_losses.append(loss / len(client_train_data))

        return self._get_new_cluster_identities_from_losses(cluster_losses, self.clusters_per_client)

    @staticmethod
    def _get_new_cluster_identities_from_losses(cluster_losses: np.ndarray | list[float], n: int) -> np.ndarray:
        """Get clusters with the lowest loss"""
        return np.argsort(cluster_losses)[:n]

    def _test_client_round(self, model, client_test_data):
        round_loss = 0
        round_y_pred, round_y_true = [], []

        model.eval()

        for X, y in client_test_data:
            X, y = X.to(self.device), y.to(self.device)

            with torch.no_grad():
                pred = model(X)

                batch_loss = self.loss_fn(pred, y)
                round_loss += batch_loss.cpu().item()

            round_y_pred.append(pred.argmax(1).detach().cpu())
            round_y_true.append(y.argmax(1).detach().cpu())

        round_loss /= len(client_test_data)
        round_y_pred = np.concatenate(round_y_pred)
        round_y_true = np.concatenate(round_y_true)
        assert len(round_y_pred) == len(round_y_true)
        assert len(round_y_pred) == len(client_test_data.dataset)

        return round_loss, f1_score(round_y_true, round_y_pred, average='macro')

    def _train_epoch(self, model, optimizer, client_train_data):
        epoch_loss = 0

        for X, y in client_train_data:
            X, y = X.to(self.device), y.to(self.device)

            pred = model(X)
            loss = self.loss_fn(pred, y)
            epoch_loss += loss.cpu().item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        return epoch_loss / len(client_train_data)

    def _aggregate_cluster_weights(self, old_weights, updated_weights, chosen_client_identities, dataset_sizes,
                                   cluster_identities) -> list[StateDict]:
        updated_cluster_weights = []
        for c in range(self.n_clusters):
            relevant_clients = [k for k in chosen_client_identities if c in cluster_identities[k, :]]
            if len(relevant_clients) == 0:
                # keep old weights if no clients were trained for c in this round
                warnings.warn(f"No clients for cluster {c}")
                updated_cluster_weights.append(old_weights[c])
            else:
                relevant_weights = [updated_weights[i] for i, k in enumerate(chosen_client_identities) if
                                    c in cluster_identities[k, :]]
                dataset_weights = dataset_sizes[relevant_clients]

                updated_cluster_weights.append(average_parameters(relevant_weights, weights=dataset_weights))

        return updated_cluster_weights
