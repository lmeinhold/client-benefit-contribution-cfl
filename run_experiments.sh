#!/usr/bin/env bash

# Label Distribution on CIFAR-10
./train.py --datasets cifar10 --imbalance-types label_distribution --imbalances 0.01,0.1,1,10 --rounds 150 --epochs 8 --penalty 0.1 --clusters 10 --clusters-per-client 6 --n-clients 120
