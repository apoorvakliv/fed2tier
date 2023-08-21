import torch

#from PIL import Image
from torch.utils import data
#from torchvision import transforms
#import pickle

class distributionDataloader(data.Dataset):

    def __init__(
        self,
        trainset,
        data_path,
        clientID = 0
    ):

        self.trainset = trainset
        self.path1=data_path
        self.clientID = clientID
        self.mean = 33.3184
        self.stdv = 78.5675


        self.data_idxs = torch.load(data_path)['datapoints'][clientID]


    def __len__(self):
        return len(self.data_idxs)

    def __getitem__(self, index):
        image = self.trainset[self.data_idxs[index]][0]
        label = self.trainset[self.data_idxs[index]][1]

        return image, label
