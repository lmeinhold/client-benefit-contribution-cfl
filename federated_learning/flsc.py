import numpy as np
import torch
from numpy import signedinteger
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from federated_learning.ifca import IFCA, IfcaClient
from utils.metrics_logging import Logger
from utils.torchutils import StateDict, average_state_dicts


class FLSC(IFCA):
    """Federated Learning with Soft Clustering (FLSC)"""

    def __init__(self, client_data: list[DataLoader], model_fn, optimizer_fn, loss_fn, rounds: int, epochs: int, k: int,
                 n: int, logger: Logger, alpha: float = 0.3, device: str = "cpu", test_data: DataLoader = None):
        """Create a new IFCA instance
            Parameters:
                client_data: list of DataLoaders holding the datasets for each client
                model_fn: a function that returns the model to use on each client
                optimizer_fn: a function that returns the optimizer to use on each client
                loss_fn: a loss function to use
                rounds: number of federated learning rounds
                epochs: number of epochs on each client per federated learning roundzo
                k: number of cluster models
                n: number of clusters each client is assigned to
                alpha: fraction of clients that are selected for each round
                device: the torch device to use for training
                test_data: a DataLoader with test datasets to evaluate the global models OR k DataLoaders to evaluate each
                    client on OR None if no test evaluation should be performed
        """
        self.n = n
        super().__init__(client_data, model_fn, optimizer_fn, loss_fn, rounds, epochs, k, logger, alpha, device,
                         test_data)
        self.cluster_estimates = np.array([np.random.randint(low=0, high=k, size=n) for _ in
                                           range(len(client_data))])  # initial random cluster assignments

    def create_client(self, client_id, data_loader, test_dataloader) -> "FlscClient":
        return FlscClient(client_id, data_loader, self.model_fn, self.optimizer_fn, self.loss_fn, self.n, self.logger,
                          test_dataloader, self.device)

    def train_round(self):
        cluster_weights = [self.cluster_models[i].state_dict() for i in range(self.k)]
        new_weights = []
        num_clients = int(np.ceil(self.alpha * len(self.clients)))
        client_choices = np.random.choice(range(len(self.clients)), size=num_clients)
        old_cluster_estimates = self.cluster_estimates.copy()

        losses = 0
        corrects = 0
        samples = 0

        for i in tqdm(client_choices):
            c_i = self.clients[i]
            w_i, alpha_k, test_loss, test_correct = c_i.train_round(cluster_weights, old_cluster_estimates[i],
                                                                    self.epochs)
            new_weights.append(w_i)
            self.cluster_estimates[i] = alpha_k

            if test_loss is not None:
                n = len(c_i.test_dataloader.dataset)
                losses += test_loss * n
                corrects += test_correct
                samples += n

        if samples > 0:
            weighted_client_loss = losses / samples
            weighted_client_accuracy = corrects / samples
            self.logger.log_server_metrics(round, stage="test", accuracy=weighted_client_accuracy,
                                           loss=weighted_client_loss)

        updated_weights = self.aggregate_weights(new_weights, old_cluster_estimates[client_choices])
        self.cluster_estimates = np.array(self.cluster_estimates, dtype=int)
        for i, w_i in enumerate(updated_weights):
            self.cluster_models[i].load_state_dict(w_i, strict=False)

    def aggregate_weights(self, weights, cluster_estimates) -> list[StateDict]:
        updated_cluster_weights = [[] for _ in range(self.k)]
        for j in range(self.k):
            for i, estimate in enumerate(cluster_estimates):
                if j in estimate:
                    theta_ij = weights[i]
                    updated_cluster_weights[j].append(theta_ij)

        aggregated_cluster_weights = [average_state_dicts(updated_cluster_weights[j]) for j in range(self.k)]
        return aggregated_cluster_weights


def fuse_weights(state_dicts: list[StateDict]) -> StateDict:
    return average_state_dicts(state_dicts)


class FlscClient(IfcaClient):
    def __init__(self, client_id: int, data_loader, model_fn, optimizer_fn, loss_fn, n: int, logger: Logger,
                 test_dataloader: DataLoader = None, device="cpu"):
        super().__init__(client_id, data_loader, model_fn, optimizer_fn, loss_fn, logger, test_dataloader, device)
        self.n = n

    def train_round(self, cluster_states: [StateDict], cluster_assignments: np.ndarray, epochs: int) -> tuple[
        StateDict, np.array]:
        new_cluster_assignments = self.estimate_cluster(cluster_states)

        model = self.build_model([cluster_states[i] for i in cluster_assignments])
        optimizer = self.build_optimizer(model)

        self.train_epochs(epochs, model, optimizer)

        test_loss = None
        test_correct = None

        if self.test_dataloader is not None:
            test_loss, test_correct = self.test_round(model, round)

        return model.state_dict(), new_cluster_assignments, test_loss, test_correct

    def estimate_cluster(self, cluster_states: list[StateDict]) -> list[signedinteger]:
        losses = self.evaluate_losses(cluster_states)
        return losses.argpartition(self.n)[0:self.n]

    def build_model(self, state_dicts: list[StateDict] | StateDict) -> torch.nn.Module:
        if isinstance(state_dicts, list):
            state_dict = fuse_weights(state_dicts)
            return super().build_model(state_dict)
        else:
            return super().build_model(state_dicts)
