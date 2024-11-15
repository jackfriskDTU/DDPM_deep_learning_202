import numpy as np
import os
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets,transforms
from PIL import Image
import matplotlib.pyplot as plt

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

    class TransformedDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, transform):
            self.dataset = dataset
            self.transform = transform

        def __getitem__(self, idx):
            image, label = self.dataset[idx]
            return self.transform(image), label

        def __len__(self):
            return len(self.dataset)

    def preprocess_mnist(batch_size):
        train_loader, test_loader = Preprocess.load_mnist(batch_size)

        def scale(min_val, max_val, tensor):
            # Scale to [0,1] first
            tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
            # Then scale to [min_val, max_val]
            tensor = tensor * (max_val - min_val) + min_val
            return tensor

        # Create new loaders with transformed data
        transformed_train_loader = DataLoader(
            Preprocess.TransformedDataset(train_loader.dataset, lambda x: scale(-1, 1, x)), 
            batch_size=batch_size, 
            shuffle=True
        )
        transformed_test_loader = DataLoader(
            Preprocess.TransformedDataset(test_loader.dataset, lambda x: scale(-1, 1, x)), 
            batch_size=batch_size, 
            shuffle=False
        )

        return transformed_train_loader, transformed_test_loader
        
    # print the shape of the images
    def print_shape(loader):
        for batch in loader:
            images, labels = batch
            print(images.shape)
            break
    
    def save_mnist_image(loader, save_dir, index=0):
        """
        Save a single MNIST image from the provided loader to a specified directory.
        
        Args:
            loader: DataLoader containing MNIST images
            save_dir: Directory where the image will be saved
            index: Index of the image to save (default: 0)
            
        Returns:
            tuple: (save_path, label) of the saved image
        """
        try:
            # Create save directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)
            
            # Get the dataset from the loader
            dataset = loader.dataset
            
            # Get the image and label
            if isinstance(dataset, Subset):
                image, label = dataset.dataset[dataset.indices[index]]
            else:
                image, label = dataset[index]
            
            # Convert image for saving
            # if isinstance(image, torch.Tensor):
            #     # If image is scaled to [-1, 1], rescale to [0, 1]
            #     if image.min() < 0:
            #         image = (image + 1) / 2
                
            #     # Convert to PIL Image
            #     image = transforms.ToPILImage()(image)
            
            # Create filename with index and label
            filename = f'mnist_image_{index}_label_{label}.png'
            save_path = os.path.join(save_dir, filename)
            
            # Save the image
            image.save(save_path)
            print(f"Image saved successfully to {save_path}")
            
            return save_path, label
            
        except Exception as e:
            print(f"Error saving image: {str(e)}")
            return None, None

if __name__ == '__main__':
    train_loader, test_loader = Preprocess.preprocess_mnist(64)
    # Preprocess.save_mnist_image(train_loader, save_dir='mnist_images', index=0)
    # Preprocess.print_shape(train_loader)
    print(train_loader.dataset[0])
    # print('Train:', len(train_loader.dataset))
    # print('Test:', len(test_loader.dataset))
    # Preprocess.print_shape(train_loader)