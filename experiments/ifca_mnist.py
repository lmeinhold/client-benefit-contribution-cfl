from pathlib import Path

from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from experiments.datasets.base import create_dataloader, split_dataset
from experiments.datasets.mnist import MNIST
from federated_learning.ifca import IFCA
from models.mnist import CNN
from utils.metrics_logging import JsonAdapter, Logger, ConsoleAdapter
from utils.torchutils import get_device

BATCH_SIZE = 64
N_CLIENTS = 100
LR = 2e-3
EPOCHS = 5
ROUNDS = 30
ALPHA = 0.8
N_CLUSTERS = 3
DATASET = MNIST(save_dir="../data")
MODEL = CNN


def create_optimizer(params, lr):
    return SGD(params, lr)


def main():
    logger = Logger(algorithm="IFCA", dataset=DATASET.get_name(), model="CNN", epochs=EPOCHS)
    logger_adapter = ConsoleAdapter()  # JsonAdapter(Path("../output"), logger.run_id)
    logger.attach(logger_adapter)
    logger.log_run_data()

    train_data, test_data = DATASET.train_data(), DATASET.test_data()
    client_data_loaders = [create_dataloader(d, BATCH_SIZE) for d in split_dataset(train_data, N_CLIENTS)]

    ifca = IFCA(
        client_data_loaders,
        model_fn=MODEL,
        optimizer_fn=lambda p: create_optimizer(p, LR),
        loss_fn=CrossEntropyLoss(),
        rounds=ROUNDS,
        epochs=EPOCHS,
        logger=logger,
        alpha=ALPHA,
        device=get_device(),
        test_data=create_dataloader(test_data, BATCH_SIZE),
        k=N_CLUSTERS
    )

    ifca.fit()


if __name__ == "__main__":
    main()
