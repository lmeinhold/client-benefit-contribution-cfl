[![Python Tests](https://github.com/lmeinhold/client-benefit-contribution-cfl/actions/workflows/python-test.yaml/badge.svg)](https://github.com/lmeinhold/client-benefit-contribution-cfl/actions/workflows/python-test.yaml)

# Client Benefit and Contribution in Clustered Federated Learning
## Setup
```shell
# Install dependencies needed for model training 
pip install -r requirements.txt

# Install additional dependencies for tests and utility scripts
pip install -r requirements-dev.txt
```

## How to Run
### Local
Model training example:
```shell
./train.py --datasets=MNIST\
      --type benefit
      --imbalance-types=label_distribution\
      --imbalances=0.1,1,10\
      --algorithms=FedAvg,IFCA\
      --rounds=3\
      --epochs=5\
      --penalty=1\
      --n-clients=80\
      --clients-per-round=0.8\
      --clusters=5\
      --clusters-per-client=2
```

View help:
```shell
./train.py --help
```

### Docker
Build container:
```shell
docker build -t cfl:latest .
```

Run everything:
```shell
docker run -it --rm -v ./output:/usr/src/cfl/output cfl:latest ./train.py
```

List algorithms and datasets:
```shell
docker run -it --rm mcfl:latest ./train.py --list-algorithms
docker run -it --rm mcfl:latest ./train.py --list-datasets
```

View help:
```shell
docker run -it --rm cfl:latest ./train.py --help
```

### Docker with NVIDIA GPU
Install and configure the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Then run the docker commands with gpu enabled:

Using the NVIDIA docker runtime:
```shell
docker run -it --rm --runtime=nvidia --gpus=all -v ./output:/usr/src/cfl/output cfl:latest ./train.py
```

Or using CDI:
```shell
podman run -it --rm --device nvidia.com/gpu=all --security-opt=label=disable -v ./output:/usr/src/cfl/output cfl:latest ./train.py
```