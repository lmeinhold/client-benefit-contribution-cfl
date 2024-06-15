#!/usr/bin/env bash

FEDPROX_MU=1

ROUNDS_BENEFIT=150
CLIENTS_BENEFIT=120
IMBALANCES_BENEFIT=

ROUNDS_CONTRIBUTION=100
CLIENTS_CONTRIBUTION=30
LXO_CONTRIBUTION=3


# Label Distribution on CIFAR-10
./train.py --datasets cifar10 --imbalance-types label_distribution --imbalances 0.01,0.1,1 --rounds 150 --epochs 8 --penalty 1 --clusters 10 --clusters-per-client 6 --n-clients 120

# Label Distribution on MNIST
./train.py --datasets mnist --imbalance-types label_distribution --imbalances 0.01,0.1,1 --rounds 150 --epochs 8 --penalty 1 --clusters 10 --clusters-per-client 6 --n-clients 120
