import torch
from torch.utils.data import DataLoader


class FedAvg:
    def __init__(self, client_data: list[DataLoader], model_fn,
                 optimizer_fn, loss_fn, rounds: int, epochs: int):
        self.model_fn = model_fn
        self.loss_fn = loss_fn
        self.optimizer_fn = optimizer_fn
        self.epochs = epochs
        self.rounds = rounds
        self.clients = [self.create_client(i, d) for i, d in enumerate(client_data)]
        self.global_model = model_fn()

    def create_client(self, client_id, data_loader):
        return _Client(client_id, data_loader, self.model_fn, self.optimizer_fn, self.loss_fn)

    def train_round(self):
        global_weights = self.global_model.state_dict()
        weights = []
        for c in self.clients:
            w_i = c.train_round(global_weights, self.epochs)
            weights.append(w_i)

        updated_weights = self.aggregate_weights(global_weights, weights)

        self.global_model.load_state_dict(updated_weights)

    @staticmethod
    def aggregate_weights(global_weights, weights):
        updated_weights = {}
        for k in global_weights.keys():
            client_weights = torch.stack([w_i[k] for w_i in weights])
            print(f"{k}: {client_weights.dtype}")
            updated_weights[k] = client_weights.mean(dim=0)
        return updated_weights

    def fit(self):
        for r in range(self.rounds):
            print(f"FedAvg round {r + 1} --------------")
            self.train_round()


class _Client:
    def __init__(self, client_id: int, data_loader, model_fn, optimizer_fn, loss_fn, device="cpu"):
        self.client_id = client_id
        self.data_loader = data_loader
        self.model_fn = model_fn
        self.optimizer_fn = optimizer_fn
        self.loss_fn = loss_fn
        self.device = device

    def build_model(self, state_dict: dict[str, torch.Tensor]) -> torch.nn.Module:
        """build a model from given weights"""
        m = self.model_fn()
        m.load_state_dict(state_dict)
        return m.to(self.device)

    def build_optimizer(self, model) -> torch.optim.Optimizer:
        return self.optimizer_fn(model.parameters())

    def train_round(self, shared_state: dict[str, torch.Tensor], epochs: int):
        model = self.build_model(shared_state)
        optimizer = self.build_optimizer(model)

        for t in range(epochs):
            print(f"Client {self.client_id}: Epoch {t + 1}")
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

                if batch % 200 == 0:
                    loss = loss.item()
                    print(f"Client {self.client_id}: loss: {loss:>7f}")

        return model.state_dict()
