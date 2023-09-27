from torch.nn import CrossEntropyLoss

from experiments.fedavg_mnist import load_data, create_dataloader, split_dataset, create_model, create_optimizer
from federated_learning.fedprox import FedProx
from federated_learning.torchutils import get_device

BATCH_SIZE = 64
N_CLIENTS = 100
LR = 2e-3
EPOCHS = 5
ROUNDS = 30
ALPHA = 0.8
MU = 1


def main():
    train_data, test_data = load_data()
    client_data_loaders = [create_dataloader(d) for d in split_dataset(train_data, N_CLIENTS)]

    fp = FedProx(
        client_data_loaders,
        model_fn=create_model,
        optimizer_fn=lambda p: create_optimizer(p, LR),
        loss_fn=CrossEntropyLoss(),
        rounds=ROUNDS,
        epochs=EPOCHS,
        mu=MU,
        alpha=ALPHA,
        device=get_device(),
        test_data=create_dataloader(test_data),
    )

    fp.fit()


if __name__ == "__main__":
    main()
