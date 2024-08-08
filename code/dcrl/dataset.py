import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

import numpy as np
from collections import defaultdict
from common import get_default_device

# Data loader definition
class UnsupervisedDataset(Dataset):
    def __init__(self, dataset, k):
        self.dataset = dataset
        self.k = k
        self.label_to_indices = defaultdict(list)
        
        for idx, (_, label) in enumerate(self.dataset):
            self.label_to_indices[label].append(idx)
        
        self.indices_by_label = {label: np.array(indices) for label, indices in self.label_to_indices.items()}
        self.all_labels = list(self.label_to_indices.keys())

    def __getitem__(self, index):
        x, label = self.dataset[index]
        x = x.view(-1)
        
        # Select x_positive with the same label
        positive_idx = np.random.choice(self.indices_by_label[label])
        while positive_idx == index:
            positive_idx = np.random.choice(self.indices_by_label[label])
        x_positive, _ = self.dataset[positive_idx]
        x_positive = x_positive.view(-1)

        # Select k negative samples with different labels
        negative_samples = []
        negative_labels = np.random.choice([l for l in self.all_labels if l != label], self.k, replace=False)
        for neg_label in negative_labels:
            negative_idx = np.random.choice(self.indices_by_label[neg_label])
            x_negative, _ = self.dataset[negative_idx]
            x_negative = x_negative.view(-1)
            negative_samples.append(x_negative)
        
        return (x, x_positive, negative_samples)

    def __len__(self):
        return len(self.dataset)

# Data loader
default_transform = transforms.Compose([transforms.ToTensor()])
def get_dataset(name='cifar100', k=3):
    # Get raw dataset
    train_data, test_data = None, None
    if name == 'cifar100':
        train_data = torchvision.datasets.CIFAR100('./data', train=True, download=True, transform=default_transform)
        test_data  = torchvision.datasets.CIFAR100('./data', train=False, download=True, transform=default_transform)
    elif name == 'mnist':
        train_data = torchvision.datasets.MNIST('./data', train=True, download=True, transform=default_transform)
        test_data  = torchvision.datasets.MNIST('./data', train=False, download=True, transform=default_transform)

    # Wrap them in custom dataset definition
    train_data = UnsupervisedDataset(train_data, k=k)
    test_data  = UnsupervisedDataset(test_data, k=k)
    return train_data, test_data

def get_dataloader(name='cifar100', batch_size=64, sample_ratio=1.0, k=3):
    train_data, test_data = get_dataset(name=name, k=k)

    # Sample fewer data samples
    train_sampler = SubsetRandomSampler(
        indices=torch.arange(int(len(train_data) * sample_ratio))
    )
    test_sampler = SubsetRandomSampler(
        indices=torch.arange(int(len(test_data) * sample_ratio))
    )

    # Create custom dataloaders
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader
