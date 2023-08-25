.. _running:

*******************************
Running the Server and Nodes
*******************************

Starting the Server
-------------------

The server is started by running the following command in the root directory of the framework:

.. code-block:: console

    python -m fed2tier.server.start_server

### Server Arguments

You can customize the server's behavior using various arguments:

.. list-table:: 
   :header-rows: 1
   :widths: 20 50 15

   * - Argument
     - Description
     - Default
   * - `--algorithm`
     - Algorithm to be used for aggregation by server
     - `fedavg`
   * - `--nodes`
     - Number of nodes to be used
     - `1`
   * - `--s_rounds`
     - Number of communication rounds to be executed by server
     - `10`
   * - `--n_rounds`
     - Maximum number of communication rounds to be executed by nodes
     - `10`
   * - `--batch_size`
     - Batch size to be used
     - `10`
   * - `--dataset`
     - Dataset to be used
     - `MNIST`
   * - `--net`
     - Network to be used
     - `LeNet`
   * - `--accept_conn`
     - Determines if connections accepted after FL begins
     - `1`
   * - `--model_path`
     - Specifies initial server model path
     - `initial_model.pt`
   * - `--resize_size`
     - Specifies dataset resize dimension
     - `32`
   * - `--threshold`
     - Specifies accuracy threshold for early stopping at each node
     - `0.8`

Starting the Nodes
-------------------

The nodes are started by running the following command in the root directory of the framework:

.. code-block:: console

    python fed2tier.node.start_node

### Node Arguments

You can customize the node's behavior using various arguments:

.. list-table:: 
   :header-rows: 1
   :widths: 20 50 15

   * - Argument
     - Description
     - Default
   * - `--device`
     - Device to run the client on
     - `cpu`
   * - `--wait_time`
     - Time to wait before sending the next request
     - `5`
   * - `--clients`
     - Number of clients to run
     - `10`
   * - `--niid`
     - niid or iid
     - `1`
   * - `--algorithm`
     - Algorithm to run
     - `fedavg`
   * - `--epochs`
     - Number of epochs
     - `5`
   * - `--mu`
     - mu hyperparameter for fedprox
     - `0.1`
   * - `--rounds`
     - Number of communication rounds
     - `20`
   * - `--carbon`
     - If 1, track carbon emission of the node
     - `0`

