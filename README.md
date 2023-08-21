# Fed2Tier: A Two-Tier Federated Learning System Towards Green Computation
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

`Fed2Tier` is a novel two-tier federated learning framework aimed at efficient and green computation. It represents an innovative approach to distributed machine learning that emphasizes privacy, scalability, and environmental sustainability. The Fed2Tier framework seeks to enhance model generalizability by involving a greater number of edge devices in the training process.

## Supported devices
`Fed2Tier` has been extensively tested on and works with the following devices:

* Intel CPUs
* Nvidia GPUs
* Nvidia Jetson
* Raspberry Pi
* Intel NUC

## Installation

```
$ git clone https://github.com/anupamkliv/FedERA.git
$ cd `Fed2Tier`
$ pip install -r requirements.txt
```
## Starting server

```
python -m fed2tier.server.start_server \
 --algorithm fedavg \
 --nodes 1 \
 --n_rounds 10 \
 --s_rounds 10 \
 --batch_size 10 \
 --dataset MNIST \
```

## Starting Node

```
python -m fed2tier.node.start_node \
 --device cpu \
 --algorithm scaffold \
 --rounds 10 \
 --epochs 10 \
 --niid 4 \
 --clients 15 \
```

## Arguments to the clients and server

### Server

| Argument | Description | Default |
| --- | --- | --- |
| `--algorithm` | Algorithm to be used for aggregation by server | `fedavg` |
| `--nodes` | Number of nodes to be used | `1` |
| `--s_rounds` | Number of communication rounds to be executed by server | `10` |
| `--n_rounds` | Maximum number of communication rounds to be executed by nodes | `10` |
| `--batch_size` | Batch size to be used | `10` |
| `--dataset` | Dataset to be used | `MNIST` |
| `--net` | Network to be used | `LeNet` |
| `--accept_conn`| determines if connections accepted after FL begins          | `1`       |
| `--model_path` | specifies initial server model path                         | `initial_model.pt` |
| `--resize_size`| specifies dataset resize dimension                          | `32`      |
| `--threshold`  | specifies accuracy threshold for early stopping at each node| `0.8`     |

### Node

| Argument      | Description                                             | Default            |
|---------------|---------------------------------------------------------|--------------------|
| `--device`    | Device to run the client on                              | `cpu`              |
| `--wait_time` | Time to wait before sending the next request             | `5`                |
| `--clients`   | Number of clients to run                                 | `10`               |
| `--niid`      | niid or iid                                              | `1`                |
| `--algorithm` | Algorithm to run                                         | `fedavg`           |
| `--epochs`    | Number of epochs                                         | `5`                |
| `--mu`        | mu hyperparameter for fedprox                            | `0.1`              |
| `--rounds`    | Number of communication rounds                           | `20`               |
| `--carbon`    | If 1, track carbon emission of the node                   | `0`                |

