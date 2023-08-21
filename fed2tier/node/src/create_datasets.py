

from random import randint, shuffle
import subprocess
import os
import shutil
from tqdm import tqdm

import torch
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import random_split
from .distribution import data_distribution

from torch.utils.data import DataLoader

# trainset = CIFAR10("./", train=True, download=True, transform=transforms.ToTensor())

# def make_client_datasets(num_of_clients = 1):
#     client_sizes = []
#     remaining = len(mnist_dataset)
#     for _ in range(num_of_clients):
#         try:
#             client_sizes.append( randint(1, remaining + 1) )
#             remaining -= client_sizes[-1]
#         except ValueError:
#             client_sizes.append(0)
#     client_sizes[-1] += remaining

#     datasets = random_split(mnist_dataset, client_sizes)
#     torch.save(datasets, "client_datasets.pt")
def get_data(config):
    # If the dataset is not custom, create a dataset folder
    if config['dataset'] != 'CUSTOM':
        dataset_path = "client_dataset"
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)

    # Get the train and test datasets for each supported dataset
    if config['dataset'] == 'MNIST':
        # Apply transformations to the images
        apply_transform = transforms.Compose([transforms.Resize(config["resize_size"]), transforms.ToTensor()])
        # Download and load the trainset
        trainset = datasets.MNIST(root='client_dataset/MNIST', train=True, download=True, transform=apply_transform)
        # Download and load the testset
        testset = datasets.MNIST(root='client_dataset/MNIST', train=False, download=True, transform=apply_transform)
    elif config['dataset'] == 'FashionMNIST':
        apply_transform = transforms.Compose([transforms.Resize(config['resize_size']), transforms.ToTensor()])
        trainset = datasets.FashionMNIST(root='client_dataset/FashionMNIST',
                                        train=True, download=True, transform=apply_transform)
        testset = datasets.FashionMNIST(root='client_dataset/FashionMNIST',
                                        train=False, download=True, transform=apply_transform)
    elif config['dataset'] == 'CIFAR10':
        apply_transform = transforms.Compose([transforms.Resize(config['resize_size']), transforms.ToTensor()])
        trainset = datasets.CIFAR10(root='client_dataset/CIFAR10',
                                    train=True, download=True, transform=apply_transform)
        testset = datasets.CIFAR10(root='client_dataset/CIFAR10',
                                   train=False, download=True, transform=apply_transform)
    elif config['dataset'] == 'CIFAR100':
        apply_transform = transforms.Compose([transforms.Resize(config['resize_size']), transforms.ToTensor()])
        trainset = datasets.CIFAR100(root='client_dataset/CIFAR100',
                                     train=True, download=True, transform=apply_transform)
        testset = datasets.CIFAR100(root='client_dataset/CIFAR100',
                                    train=False, download=True, transform=apply_transform)
    else:
        # Raise an error if an unsupported dataset is specified
        raise ValueError(f"Unsupported dataset type: {config['dataset']}")


    # Return the train and test datasets
    return trainset, testset



def make_client_datasets(config):

    trainset, testset = get_data(config)
    data_distribution(config, trainset)
    data_path = os.path.join(os.getcwd(), 'Distribution/', config['dataset'],
                                 'data_split_niid_'+ str(config['niid'])+'.pt')
    return data_path, trainset, testset
