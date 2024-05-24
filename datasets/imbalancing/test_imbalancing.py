import numpy as np

from datasets.imbalancing.imbalancing import split_with_fixed_num_labels, extract_raw_data
from datasets.mnist import MNIST


def test_split_with_fixed_num_labels():
    #ds = MNIST("/var/tmp/") # TODO use synthetic dataset and re-enable test
    #train = ds.train_data()
    #_, labels = extract_raw_data(train)
    #split_ds = split_with_fixed_num_labels(train, n_clients=20, c=2)

    #assert len(split_ds) == 20, "Expected 20 datasets, got {}".format(len(split_ds))
    #for sds in split_ds:
    #    subset_labels = labels[sds.indices]
    #    n_classes = np.unique(subset_labels).shape[0]
    #    assert n_classes == 2, "Wrong number of classes in subset, expected 2, got {}".format(n_classes)
    pass
