from pathlib import Path

from torch.nn import CrossEntropyLoss

from experiments.datasets.base import split_dataset, create_dataloader
from experiments.datasets.mnist import MNIST
from experiments.fedavg_mnist import create_optimizer
from federated_learning.fedprox import FedProx
from models.mnist import CNN
from utils.metrics_logging import JsonAdapter, Logger
from utils.torchutils import get_device

BATCH_SIZE = 64
N_CLIENTS = 100
LR = 2e-3
EPOCHS = 5
ROUNDS = 30
ALPHA = 0.8
MU = 1
MODEL = CNN


def main():
    logger = Logger(algorithm="FedProx", dataset="MNIST", model="CNN", rounds=ROUNDS, epochs=EPOCHS)
    logger_adapter = JsonAdapter(Path("../output"), logger.run_id)
    logger.attach(logger_adapter)
    logger.log_run_data()

    ds = MNIST("../data")
    train_data, test_data = ds.train_data(), ds.test_data()
    client_data_loaders = [create_dataloader(d, BATCH_SIZE) for d in split_dataset(train_data, N_CLIENTS)]

    fp = FedProx(
        client_data_loaders,
        model_fn=lambda: MODEL(),
        optimizer_fn=lambda p: create_optimizer(p, LR),
        loss_fn=CrossEntropyLoss(),
        rounds=ROUNDS,
        epochs=EPOCHS,
        mu=MU,
        alpha=ALPHA,
        logger=logger,
        device=get_device(),
        test_data=create_dataloader(test_data, BATCH_SIZE)
    )

    fp.fit()


if __name__ == "__main__":
    main()
