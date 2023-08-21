.. _models:

*******
Models 
*******

The models currently implemented in the framework are:

* LeNet-5
* ResNet-18
* ResNet-50
* VGG-16
* AlexNet

The `server_lib.py` file contains the implementation of Deep-Learning models for the server, while the `net.py` file contains the implementation of these models for the node. These models are either created by inheriting from torch.nn.module or are imported from torchvision.models.



