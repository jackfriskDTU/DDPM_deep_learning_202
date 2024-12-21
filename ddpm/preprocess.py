import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets,transforms

import matplotlib.pyplot as plt

class Preprocess:
    @staticmethod
    def load_dataset_with_transform(batch_size, dataset='cifar10', train_size=50000, test_size=10000):
        if dataset == 'cifar10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            train_dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
        elif dataset == 'mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5))
            ])
            train_dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST(root='../data', train=False, download=True, transform=transform)
        else:
            raise ValueError(f"Dataset {dataset} not supported. Choose 'mnist' or 'cifar10'")

        # Create subsets
        train_dataset = Subset(train_dataset, np.arange(0, train_size))
        test_dataset = Subset(test_dataset, np.arange(0, test_size))

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        return train_loader, test_loader

    @staticmethod
    def preprocess_dataset(batch_size, dataset='cifar10', train_size=50000, test_size=10000):
        # Just call our new loader function
        return Preprocess.load_dataset_with_transform(batch_size, dataset, train_size, test_size)

if __name__ == '__main__':
    ### Load data ###
    train_loader, test_loader = Preprocess.load_dataset_with_transform(200, dataset='cifar10', train_size=200, test_size=200)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### Visualize label distribution ###
    # Fetch all the labels and plot the distribution
    labels_train = [label for _, label in train_loader.dataset]
    labels_test = [label for _, label in test_loader.dataset]
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(labels_train, bins=10, color='blue', alpha=0.7)
    plt.title('Train labels distribution')

    plt.subplot(1, 2, 2)
    plt.hist(labels_test, bins=10, color='red', alpha=0.7)
    plt.title('Test labels distribution')
    plt.savefig('poster/labels_distribution.png')