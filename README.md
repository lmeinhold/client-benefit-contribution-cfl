[![pipeline status](https://gitlab.com/bowerick/masterthesis-clustered-fl/badges/main/pipeline.svg)](https://gitlab.com/bowerick/masterthesis-clustered-fl/-/commits/main) ![coverage](https://gitlab.com/bowerick/masterthesis-clustered-fl/badges/main/coverage.svg?job=test)

# Client Benefit and Contribution in Clustered Federated Learning
## How to Run
### Docker
```sh
docker build -t mcfl:latest .

# Run all experiments
docker run -it --rm -v ./output:/usr/src/mcfl/output mcfl:latest

# Run specific experiments
docker run -it --rm -v ./output:/usr/src/mcfl/output -e MCFL_EXPERIMENTS=fedavg_mnist,ifca_cifar10 mcfl:latest

# List available experiments
docker run -it --rm mcfl:latest scripts/run_experiments.py --list
```
