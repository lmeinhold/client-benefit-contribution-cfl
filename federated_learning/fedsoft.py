import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class FedSoft:
    """FedSoft: Soft Clustered Federated Learning with Proximal Local Updating"""
    def __init__(self, client_data: list[DataLoader], model_fn, optimizer_fn, loss_fn, rounds: int, epochs: int,
                 alpha: float = 0.3, device: str = "cpu", test_data: DataLoader = None):
        self.alpha = alpha
        self.device = device
        self.test_data = test_data
        self.model_fn = model_fn
        self.loss_fn = loss_fn
        self.optimizer_fn = optimizer_fn
        self.epochs = epochs
        self.rounds = rounds
        self.clients = [self.create_client(i, d) for i, d in enumerate(client_data)]
        self.global_model = model_fn().to(self.device)

    def create_client(self, client_id, data_loader):
        return FedSoftClient(client_id, data_loader, self.model_fn, self.optimizer_fn, self.loss_fn)


    def update_clustering(self):

    def train_round(self):
        global_weights = self.global_model.state_dict()
        weights = []
        n_c = int(np.ceil(self.alpha * len(self.clients)))
        for c in tqdm(np.random.choice(self.clients, n_c)):
            w_i = c.train_round(global_weights, self.epochs)
            weights.append(w_i)

        updated_weights = self.aggregate_weights(global_weights, weights)

        self.global_model.load_state_dict(updated_weights, strict=False)

    def test_round(self):
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
            print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    @staticmethod
    def aggregate_weights(global_weights, weights):
        updated_weights = {}
        for k in global_weights.keys():
            if k.endswith('.weight') or k.endswith('.bias'):
                client_weights = torch.stack([w_i[k] for w_i in weights])
                updated_weights[k] = client_weights.mean(dim=0)
        return updated_weights

    def fit(self):
        for r in range(self.rounds):
            print(f"FedAvg round {r + 1} --------------")
            self.train_round()
            if self.test_data is not None:
                self.test_round()


class FedSoftClient:
    def __init__(self, client_id: int, data_loader, model_fn, optimizer_fn, loss_fn, initial_clusters: list[int], device="cpu"):
        self.client_id = client_id
        self.data_loader = data_loader
        self.model_fn = model_fn
        self.optimizer_fn = optimizer_fn
        self.loss_fn = loss_fn
        self.device = device
        self.cluster_identities = initial_clusters

    def build_model(self, state_dict: dict[str, torch.Tensor]) -> torch.nn.Module:
        """build a model from given weights"""
        m = self.model_fn()
        m.load_state_dict(state_dict)
        return m.to(self.device)

    def build_optimizer(self, model) -> torch.optim.Optimizer:
        return self.optimizer_fn(model.parameters())

    @staticmethod
    def fuse_model_weights(self, weights: list[dict[str, torch.Tensor]]):
        """Combine cluster weights into a single state_dict for round training"""
        fused_weights = {}
        for k in weights[0].keys():
            if k.endswith('.weight') or k.endswith('.bias'):
                cluster_weights = torch.stack([w_i[k] for w_i in weights])
                fused_weights[k] = cluster_weights.mean(dim=0)
        return fused_weights

    def train_round(self, shared_states: list[dict[str, torch.Tensor]], epochs: int):
        model = self.build_model(self.fuse_model_weights(shared_states))
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

        return model.state_dict()
