import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset
import numpy as np
from PIL import Image
import pandas as pd

class CustomSubset(Subset):
    def __init__(self, dataset, indices):
        super(CustomSubset, self).__init__(dataset, indices)
        self.classes = np.array(dataset.classes)
        self.targets = np.array([dataset.targets[i] for i in indices])
        self.data = np.array([dataset.data[i] for i in indices])
        
               
MEAN = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
    'imagenet': (0.485, 0.456, 0.406)
}

STD = {
    'cifar10': (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
    'imagenet': (0.229, 0.224, 0.225)
}

SIZE = {
    'cifar10': 32,
    'cifar100': 32,
    'imagenet': 224
}

class CustomWeightedDataset(Dataset):
    """Class for datasets with weight for each sample"""
    
    def __init__(self, dataset, weights):
        self.dataset = dataset
        self.weights = weights

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, label = self.dataset[idx]
        weight = self.weights[idx]
        return data, label, weight

    

def get_classes_count(dataset_name):
    if dataset_name == 'cifar10':
        return 10
    elif dataset_name == 'cifar100':
        return 100
    else:
        raise ValueError('Unknown dataset name: {}'.format(dataset_name))
    

DATASETS_DICT = {
    'cifar10': CIFAR10,
    'cifar100': CIFAR100,
}

def get_transforms(dataset_name):

    mean, std = MEAN[dataset_name], STD[dataset_name]
    size = SIZE[dataset_name]
    
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
            transforms.Resize(size=size),
            transforms.Lambda(lambda img: img.convert('RGB')),
            transforms.RandomCrop(size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
    ])
    
    test_transform = transforms.Compose([
            transforms.Resize(size=size),
            transforms.Lambda(lambda img: img.convert('RGB')),
            transforms.ToTensor(),
            normalize,
    ])
    
    return train_transform, test_transform


def get_dataset(dataset_name, root, train=True, transform=None, download=True):
    if dataset_name == 'cifar10':
        return CIFAR10(root, train=train, transform=transform, download=download)
    elif dataset_name == 'cifar100':
        return CIFAR100(root, train=train, transform=transform, download=download)
    else:
        raise ValueError('Unknown dataset name: {}'.format(dataset_name))