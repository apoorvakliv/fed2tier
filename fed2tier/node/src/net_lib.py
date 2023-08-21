
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Grayscale, Compose
from copy import deepcopy
from math import ceil
from tqdm import tqdm
import time
from collections import OrderedDict
import copy

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_data(trainset, testset):
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32)
    num_examples = {"trainset": len(trainset), "testset": len(testset)}
    return trainloader, testloader, num_examples

def flush_memory():
    torch.cuda.empty_cache()

def train_model(net, trainloader, epochs, device):

    """
    Trains a neural network model on a given dataset using SGD optimizer with Cross Entropy Loss criterion.
    Args:
        net: neural network model
        trainloader: PyTorch DataLoader object for training dataset
        epochs: number of epochs to train the model
        deadline: optional deadline time for training

    Returns:
        trained model with the difference between trained model and the received model
    """
    x = deepcopy(net)
    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # Set the model to training mode
    net.train()

    # Train the model for the specified number of epochs
    for _ in tqdm(range(epochs)):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()


    # Calculate the difference between the trained model and the received model
    for param_net, param_x in zip(net.parameters(), x.parameters()):
        param_net.data = param_net.data - param_x.data

    return net

def train_scaffold(net, server_c, client_c, trainloader, epochs, device):
    """
    Trains a given neural network using the Scaffold algorithm.

    Args:
    net: A PyTorch neural network model
    trainloader: A PyTorch DataLoader containing the training dataset
    epochs: An integer specifying the number of training epochs
    deadline: An optional deadline (in seconds) for the training process

    Returns:
    A trained PyTorch neural network model

    """
    x = deepcopy(net)
    if client_c is None:
        client_c = deepcopy(server_c)
    criterion = torch.nn.CrossEntropyLoss()
    lr = 0.001

    for _ in tqdm(range(epochs)):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            loss = criterion(net(images), labels)

            #Compute (full-batch) gradient of loss with respect to net's parameters
            grads = torch.autograd.grad(loss,net.parameters())

            #Update y's parameters using gradients, client_c and server_c [Algorithm line no:10]

            for param,grad,s_c,c_c in zip(net.parameters(),grads,server_c,client_c):
                s_c, c_c = s_c.to(device), c_c.to(device)
                param.data = param.data - lr * (grad.data + (s_c.data - c_c.data))

    delta_c = [torch.zeros_like(param) for param in net.parameters()]
    new_client_c = deepcopy(delta_c)

    for param_net, param_x in zip(net.parameters(), x.parameters()):
        param_net.data = param_net.data - param_x.data

    a = (ceil(len(trainloader.dataset) / trainloader.batch_size) * epochs * lr)
    for n_c, c_l, c_g, diff in zip(new_client_c, client_c, server_c, net.parameters()):
        c_l = c_l.to(device)
        c_g = c_g.to(device)
        n_c.data += c_l.data - c_g.data - diff.data / a

    #Calculate delta_c which equals to new_client_c-client_c
    for d_c, n_c_l, c_l in zip(delta_c, new_client_c, client_c):
        d_c = d_c.to(device)
        c_l = c_l.to(device)
        d_c.data.add_(n_c_l.data - c_l.data)


    return net, delta_c, new_client_c

def train_fedavg(net, trainloader, epochs, device):
    """
    Trains a given neural network using the Federated Averaging (FedAvg) algorithm.

    Args:
    net: A PyTorch neural network model
    trainloader: A PyTorch DataLoader containing the training dataset
    epochs: An integer specifying the number of training epochs
    deadline: An optional deadline (in seconds) for the training process

    Returns:
    A trained PyTorch neural network model
    """
    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Set model to train mode
    net.train()

    # Train the model for the specified number of epochs
    for _ in tqdm(range(epochs)):
        for images, labels in trainloader:
            # Move data to device (GPU or CPU)
            images, labels = images.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = net(images)

            # Compute the loss
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Update model parameters
            optimizer.step()

    # Return the trained model
    return net

def train_fedprox(net, trainloader, epochs, device, mu):
    """
    Trains a given neural network using the Federated Proximal (Fedprox) algorithm.

    Args:
    net: A PyTorch neural network model
    trainloader: A PyTorch DataLoader containing the training dataset
    epochs: An integer specifying the number of training epochs
    deadline: An optional deadline (in seconds) for the training process

    Returns:
    A trained PyTorch neural network model
    """
    # create a copy of the initial model
    init_model = copy.deepcopy(net)
    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Set model to train mode
    net.train()

    # Train the model for the specified number of epochs
    for _ in tqdm(range(epochs)):
        for images, labels in trainloader:
            Prox_loss = 0.0
            # Move data to device (GPU or CPU)
            images, labels = images.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = net(images)

            # Compute the loss
            loss = criterion(outputs, labels)
            for paramA, paramB in zip(init_model.parameters(), net.parameters()):
                Prox_loss += torch.square(torch.norm((paramA.detach() - paramB)))
            Prox_loss = (mu/2)*Prox_loss

            # Backward pass
            (loss + Prox_loss).backward()

            # Update model parameters
            optimizer.step()

    # Return the trained model
    return net

def test_model(net, testloader, device):
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in tqdm(testloader) :
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy



# trainloader, testloader, num_examples = load_data()
# model = Net().to(DEVICE)
# torch.save(model.state_dict(), "test_state_dict.pth")
# model.load_state_dict(torch.load('trained_model.pt'))
# train_model(model, trainloader, 3)
# print(test_model(model, testloader))
# # torch.save(model.state_dict(), 'trained_model.pt')

def fedadam(initial_state_dict, trained_state_dict):

    #### Calculate the difference between the initial state dict and the trained state dict
    delta_y = OrderedDict()
    for key in initial_state_dict.keys():
        delta_y[key] = trained_state_dict[key] - initial_state_dict[key].to(trained_state_dict[key].device)

    return delta_y

def train_feddyn(net, trainloader, epochs, device, prev_grads):
    """
    Trains a given neural network using the FedDyn algorithm.
    Args:
    net: A PyTorch neural network model
    trainloader: A PyTorch DataLoader containing the training dataset
    epochs: An integer specifying the number of training epochs
    deadline: An optional deadline (in seconds) for the training process

    Returns:
    A trained PyTorch neural network model
    """
    x = deepcopy(net)
    # prev_grads = None

    if prev_grads is not None:
        prev_grads = prev_grads.to(device)
    else:
        for param in net.parameters():
            if not isinstance(prev_grads, torch.Tensor):
                prev_grads = torch.zeros_like(param.view(-1))
                prev_grads.to(device)
            else:
                prev_grads = torch.cat((prev_grads, torch.zeros_like(param.view(-1))), dim=0)
                prev_grads.to(device)

    criterion = torch.nn.CrossEntropyLoss()
   
    lr = 0.1
    alpha = 0.01

    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    for _ in tqdm(range(epochs)):
        inputs,labels = next(iter(trainloader))
        inputs, labels = inputs.float().to(device), labels.long().to(device)
        output = net(inputs)
        loss = criterion(output, labels) #Calculate the loss with respect to y's output and labels

        #Dynamic Regularisation
        lin_penalty = 0.0
        curr_params = None
        for param in net.parameters():
            if not isinstance(curr_params, torch.Tensor):
                curr_params = param.view(-1)
            else:
                curr_params = torch.cat((curr_params, param.view(-1)), dim=0)

        lin_penalty = torch.sum(curr_params * prev_grads)
        loss -= lin_penalty

        quad_penalty = 0.0
        for y, z in zip(net.parameters(), x.parameters()):
            quad_penalty += torch.nn.functional.mse_loss(y.data, z.data, reduction='sum')

        loss += (alpha/2) * quad_penalty
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=1) # Clip gradients
        optimizer.step()

        # gradients = torch.autograd.grad(loss,net.parameters())

        # for param, grad in zip(net.parameters(),gradients):
        #     param.data -= lr * grad.data

    #Calculate the difference between updated model (y) and the received model (x)
    delta = None
    for y, z in zip(net.parameters(), x.parameters()):
        if not isinstance(delta, torch.Tensor):
            delta = torch.sub(y.data.view(-1), z.data.view(-1))
        else:
            delta = torch.cat((delta, torch.sub(y.data.view(-1), z.data.view(-1))),dim=0)

    #Update prev_grads using delta which is scaled by alpha
    prev_grads = torch.sub(prev_grads, delta, alpha = alpha)
    return net, prev_grads
