.. _algorithm:

*****************************
Federated Learning Algorithms
*****************************

The implementation of federated learning algorithms in Feder consists of two components: the training part on the client side and the aggregation part on the server side. The training functions are coded in the net_lib.py file at client/src directory, while the aggregation functions are located in various files within the algorithms folder at server/src directory.

The algorithms currently implemented in **Fed2Tier** are:

* FedAvg
* FedDyn
* FedAdam
* FedAdagrad
* Scaffold
* FedYogi


