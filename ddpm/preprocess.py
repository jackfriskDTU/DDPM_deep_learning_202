import numpy as np
import os
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets,transforms
from PIL import Image

class Preprocess:

    def load_mnist(batch_size):

        # Load MNIST dataset
        train_dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transforms.ToTensor())
        test_dataset = datasets.MNIST(root='../data', train=False, download=True, transform=transforms.ToTensor())

        # create subsets
        train_dataset = Subset(train_dataset, np.arange(0, 100))
        test_dataset = Subset(test_dataset, np.arange(0, 100))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader


    def preprocess_mnist(batch_size):
        train_loader, test_loader = Preprocess.load_mnist(batch_size)

        def scale_to_neg_one_to_one(tensor):
            return tensor * 2 - 1

        # Scale images to [-1, 1]
        for loader in [train_loader, test_loader]:
            for batch in loader:
                images, labels = batch
                images = scale_to_neg_one_to_one(images)

        return train_loader, test_loader
    
    # print the shape of the images
    def print_shape(loader):
        for batch in loader:
            images, labels = batch
            print(images.shape)
            break

if __name__ == '__main__':
    train_loader, test_loader = Preprocess.preprocess_mnist(64)
    print('Train:', len(train_loader.dataset))
    print('Test:', len(test_loader.dataset))
    Preprocess.print_shape(train_loader)