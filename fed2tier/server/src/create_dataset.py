
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

mnist_dataset = MNIST("./", train=False, download=True, transform=transforms.ToTensor())

torch.save(mnist_dataset, "testing_dataset.pt")