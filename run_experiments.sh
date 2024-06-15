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
./train.py --type benefit --datasets cifar10 --imbalance-types feature_distribution --imbalances 0.03,0.1,0.3,1 --rounds 150 --epochs 6 --penalty 1 --clusters 4 --clusters-per-client 3 --n-clients 120
./train.py --type benefit --datasets mnist --imbalance-types feature_distribution --imbalances 0.03,0.1,0.3,1 --rounds 150 --epochs 6 --penalty 1 --clusters 6 --clusters-per-client 4 --n-clients 120
