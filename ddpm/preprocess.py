import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets,transforms

import matplotlib.pyplot as plt

# from .forward_process import add_noise
# from postprocess import save_image

class Preprocess:
    @staticmethod
    def load_dataset_with_transform(batch_size, dataset='cifar10', train_size=50000, test_size=10000):
        if dataset == 'cifar10':
            # CIFAR-10 normalization typically centers around 0.5 mean and std of 0.5
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            train_dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
        elif dataset == 'mnist':
            # MNIST images are already [0,1], so just convert to tensor
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5))
            ])
            train_dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST(root='../data', train=False, download=True, transform=transform)
        else:
            raise ValueError(f"Dataset {dataset} not supported. Choose 'mnist' or 'cifar10'")

        # Create subsets for training and testing if desired
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

# class Preprocess:
#     def load_dataset(batch_size, dataset, train_size=200, test_size=20):
#         if dataset == 'mnist':
#             train_dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transforms.ToTensor())
#             test_dataset = datasets.MNIST(root='../data', train=False, download=True, transform=transforms.ToTensor())
#         elif dataset == 'cifar10':
#             train_dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transforms.ToTensor())
#             test_dataset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transforms.ToTensor())
#         else:
#             raise ValueError(f"Dataset {dataset} not supported. Choose 'mnist' or 'cifar10'")

#         # create subsets
#         train_dataset = Subset(train_dataset, np.arange(0, train_size))
#         test_dataset = Subset(test_dataset, np.arange(0, test_size))

#         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#         test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        
#         return train_loader, test_loader

#     class TransformedDataset(torch.utils.data.Dataset):
#         def __init__(self, dataset, transform):
#             self.dataset = dataset
#             self.transform = transform

#         def __getitem__(self, idx):
#             image, label = self.dataset[idx]
#             return self.transform(image), label

#         def __len__(self):
#             return len(self.dataset)

#     def preprocess_dataset(batch_size, dataset='mnist', train_size=200, test_size=20):
#         train_loader, test_loader = Preprocess.load_dataset(batch_size, dataset, train_size, test_size)

#         def normalize(tensor, target_min=-1, target_max=1):
#             """
#             Normalize tensor to [target_min, target_max] range
#             Assumes input tensor is already in [0,1] range from ToTensor()
#             """
#             return (tensor * (target_max - target_min)) + target_min

#         # Create new loaders with transformed data
#         transformed_train_loader = DataLoader(
#             Preprocess.TransformedDataset(train_loader.dataset, lambda x: normalize(x)), 
#             batch_size=batch_size, 
#             shuffle=True
#         )
#         transformed_test_loader = DataLoader(
#             Preprocess.TransformedDataset(test_loader.dataset, lambda x: normalize(x)), 
#             batch_size=batch_size, 
#             shuffle=False
#         )

#         return transformed_train_loader, transformed_test_loader
        
#     # print the shape of the images
#     def print_shape(loader):
#         for batch in loader:
#             images, labels = batch
#             print(images.shape)
#             break

# def transform_range(tensor, source_min=None, source_max=None, target_min=0, target_max=255):
#     """
#     Transform tensor from [source_min, source_max] to [target_min, target_max] range.
#     If source_min/max are None, they are taken from the tensor.
    
#     Args:
#         tensor (torch.Tensor): Input tensor
#         source_min (float): Current minimum value. If None, uses tensor.min()
#         source_max (float): Current maximum value. If None, uses tensor.max()
#         target_min (float): Desired minimum value (default: 0)
#         target_max (float): Desired maximum value (default: 255)
    
#     Returns:
#         torch.Tensor: Transformed tensor
#     """
#     # Get source range if not provided
#     if source_min is None:
#         source_min = tensor.min()
#     if source_max is None:
#         source_max = tensor.max()
        
#     # First normalize to [0,1]
#     normalized = (tensor - source_min) / (source_max - source_min)
    
#     # Then scale to target range
#     transformed = normalized * (target_max - target_min) + target_min
    
#     return transformed

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