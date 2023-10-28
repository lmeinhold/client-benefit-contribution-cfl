from pathlib import Path

from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from experiments.datasets.base import create_dataloader, split_dataset
from experiments.datasets.cifar import CIFAR10
from federated_learning.fedavg import FedAvg
from models.cifar import CNN
from utils.metrics_logging import Logger, JsonAdapter
from utils.torchutils import get_device

BATCH_SIZE = 64
N_CLIENTS = 100
LR = 2e-3
EPOCHS = 5
ROUNDS = 30
ALPHA = 0.8
MODEL = CNN


def create_optimizer(params, lr):
    return SGD(params, lr)


def main():
    logger = Logger(algorithm="FedAvg", dataset="CIFAR10", model="CNN", rounds=ROUNDS, epochs=EPOCHS)
    logger_adapter = JsonAdapter(Path("../output"), logger.run_id)
    logger.attach(logger_adapter)
    logger.log_run_data()

    ds = CIFAR10 ("../data")

    train_data, test_data = ds.train_data(), ds.test_data()
    client_data_loaders = [create_dataloader(d, batch_size=BATCH_SIZE) for d in split_dataset(train_data, N_CLIENTS)]

    fa = FedAvg(
        client_data_loaders,
        model_fn=lambda: MODEL(),
        optimizer_fn=lambda p: create_optimizer(p, LR),
        loss_fn=CrossEntropyLoss(),
        rounds=ROUNDS,
        epochs=EPOCHS,
        alpha=ALPHA,
        logger=logger,
        device=get_device(),
        test_data=create_dataloader(test_data, BATCH_SIZE)
    )

    fa.fit()


if __name__ == "__main__":
    main()
