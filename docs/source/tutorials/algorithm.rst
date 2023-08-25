.. _algorithm:

*******************************
Federated Learning Algorithms
*******************************

The implementation of federated learning algorithms in Fed2Tier consists of two components: the training part on the client side and the aggregation part on the server and node side. The training functions are coded in the `net_lib.py` file located in the `node/src` directory, while the aggregation functions are located in various files within the `algorithms` folder at the `server/src` and node/src directory.

The algorithms currently implemented in **Fed2Tier** are:

.. list-table:: 
   :header-rows: 1

   * - Algorithm
     - Paper
     - Server
     - Node
   * - FedAvg
     - `Communication-Efficient Learning of Deep Networks from Decentralized Data <http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf>`_
     - ✅
     - ✅
   * - FedDyn
     - `Federated Learning Based on Dynamic Regularization <https://openreview.net/forum?id=B7v4QMR6Z9w>`_
     - 
     - ✅
   * - Scaffold
     - `SCAFFOLD: Stochastic Controlled Averaging for Federated Learning <PLACEHOLDER_LINK_FOR_SCAFFOLD>`_
     - 
     - ✅
   * - FedAdagrad
     - `Adaptive Federated Optimization <https://arxiv.org/pdf/2003.00295.pdf>`_
     - ✅
     - ✅
   * - FedAdam
     - `Adaptive Federated Optimization <https://arxiv.org/pdf/2003.00295.pdf>`_
     - ✅
     - ✅
   * - FedYogi
     - `Adaptive Federated Optimization <https://arxiv.org/pdf/2003.00295.pdf>`_
     - ✅
     - ✅
   * - FedProx
     - `FedProx: Federated Learning with Proximity <https://proceedings.mlsys.org/paper_files/paper/2020/file/1f5fe83998a09396ebe6477d9475ba0c-Paper.pdf>`_
     - 
     - ✅

---
