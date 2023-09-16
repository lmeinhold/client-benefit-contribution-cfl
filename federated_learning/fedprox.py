import torch
from torch.utils.data import DataLoader

from federated_learning.fedavg import FedAvg, FedAvgClient


class FedProx(FedAvg):
    def __init__(self, client_data: list[DataLoader], model_fn, optimizer_fn, loss_fn, rounds: int, epochs: int,
                 mu: float = 0.01, alpha: float = 0.3, device: str = "cpu", test_data: DataLoader = None):
        self.mu = mu
        super().__init__(client_data, model_fn, optimizer_fn, loss_fn, rounds, epochs, alpha, device, test_data)

    def create_client(self, client_id, data_loader):
        return FedProxClient(client_id, data_loader, self.model_fn, self.optimizer_fn, self.loss_fn, self.mu)


class FedProxClient(FedAvgClient):
    def __init__(self, client_id: int, data_loader, model_fn, optimizer_fn, loss_fn, mu: float = 0.01):
        super().__init__(client_id, data_loader, model_fn, optimizer_fn, loss_fn)
        self.mu = mu

    def train_round(self, shared_state: dict[str, torch.Tensor], epochs: int):
        model = self.build_model(shared_state)
        optimizer = self.build_optimizer(model)

        for t in range(epochs):
            size = len(self.data_loader)

            model.train()

            for batch, (X, y) in enumerate(self.data_loader):
                X = X.to(self.device)
                y = y.to(self.device)

                pred = model(X)

                proximal_term = 0
                new_state = model.state_dict()
                for k in shared_state.keys():
                    if not k.endswith("weight") or k.endswith("bias"):
                        continue
                    local_weights = new_state[k]
                    global_weights = shared_state[k]
                    proximal_term += (local_weights - global_weights).norm(2)

                loss = self.loss_fn(pred, y) + (self.mu / 2.0) * proximal_term

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        return model.state_dict()
