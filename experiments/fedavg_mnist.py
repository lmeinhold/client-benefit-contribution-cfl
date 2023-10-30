from pathlib import Path

from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from experiments.datasets.base import create_dataloader, split_dataset
from experiments.datasets.emnist import EMNIST
from experiments.datasets.mnist import MNIST
from federated_learning.fedavg import FedAvg
from models.mnist import CNN
from utils.metrics_logging import Logger, JsonAdapter
from utils.torchutils import get_device

BATCH_SIZE = 64
N_CLIENTS = 100
LR = 2e-3
EPOCHS = 5
ROUNDS = 30
ALPHA = 0.6
MODEL = CNN

DATASET = EMNIST("../data")


def create_optimizer(params, lr):
    return SGD(params, lr)


def main():
    logger = Logger(algorithm="FedAvg", dataset=DATASET.get_name(), model="CNN", rounds=ROUNDS, epochs=EPOCHS)
    logger_adapter = JsonAdapter(Path("../output"), logger.run_id)
    logger.attach(logger_adapter)
    logger.log_run_data()

    train_data, test_data = DATASET.train_data(), DATASET.test_data()
    client_data_loaders = [create_dataloader(d, batch_size=BATCH_SIZE) for d in split_dataset(train_data, N_CLIENTS)]

    fa = FedAvg(
        client_data_loaders,
        model_fn=lambda: MODEL(n_output_classes=DATASET.num_classes()),
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
