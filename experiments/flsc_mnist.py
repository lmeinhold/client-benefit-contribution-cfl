from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from experiments.datasets.base import create_dataloader, split_dataset
from experiments.datasets.mnist import MNIST
from federated_learning.flsc import FLSC
from models.mnist import CNN
from utils.metrics_logging import Logger, ConsoleAdapter
from utils.torchutils import get_device

BATCH_SIZE = 64
N_CLIENTS = 100
LR = 2e-3
EPOCHS = 5
ROUNDS = 30
ALPHA = 0.8
N_CLUSTERS = 3
N_CLUSTERS_PER_CLIENT = 2
DATASET = MNIST(save_dir="../data")
MODEL = CNN


def create_optimizer(params, lr):
    return SGD(params, lr)


def main():
    logger = Logger(algorithm="FLSC", dataset=DATASET.get_name(), model="CNN", epochs=EPOCHS)
    logger_adapter = ConsoleAdapter()  # JsonAdapter(Path("../output"), logger.run_id)
    logger.attach(logger_adapter)
    logger.log_run_data()

    train_data, test_data = DATASET.train_data(), DATASET.test_data()
    client_data_loaders = [create_dataloader(d, BATCH_SIZE) for d in split_dataset(train_data, N_CLIENTS)]

    flsc = FLSC(
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
        k=N_CLUSTERS,
        n=N_CLUSTERS_PER_CLIENT
    )

    flsc.fit()


if __name__ == "__main__":
    main()
