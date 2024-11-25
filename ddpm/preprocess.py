import numpy as np
import os
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets,transforms
from PIL import Image

import sys
from forward_process import add_noise

class Preprocess:
    def load_dataset(batch_size, dataset):
        if dataset == 'mnist':
            train_dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transforms.ToTensor())
            test_dataset = datasets.MNIST(root='../data', train=False, download=True, transform=transforms.ToTensor())
        elif dataset == 'cifar10':
            train_dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transforms.ToTensor())
            test_dataset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transforms.ToTensor())
        else:
            raise ValueError(f"Dataset {dataset} not supported. Choose 'mnist' or 'cifar10'")

        test_data = train_dataset[0]
        test_img, test_label = test_data
        # print('test_img.shape:')
        # print(test_img.shape)

        # print('test_img')
        # print(test_img)

        # create subsets
        train_dataset = Subset(train_dataset, np.arange(0, 6400))
        test_dataset = Subset(test_dataset, np.arange(0, 6400))

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

    def preprocess_dataset(batch_size, dataset='mnist'):
        train_loader, test_loader = Preprocess.load_dataset(batch_size, dataset)

        def normalize(tensor, target_min=-1, target_max=1):
            """
            Normalize tensor to [target_min, target_max] range
            Assumes input tensor is already in [0,1] range from ToTensor()
            """
            return (tensor * (target_max - target_min)) + target_min

        # Create new loaders with transformed data
        transformed_train_loader = DataLoader(
            Preprocess.TransformedDataset(train_loader.dataset, lambda x: normalize(x)), 
            batch_size=batch_size, 
            shuffle=True
        )
        transformed_test_loader = DataLoader(
            Preprocess.TransformedDataset(test_loader.dataset, lambda x: normalize(x)), 
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

def transform_range(tensor, source_min=None, source_max=None, target_min=0, target_max=255):
    """
    Transform tensor from [source_min, source_max] to [target_min, target_max] range.
    If source_min/max are None, they are taken from the tensor.
    
    Args:
        tensor (torch.Tensor): Input tensor
        source_min (float): Current minimum value. If None, uses tensor.min()
        source_max (float): Current maximum value. If None, uses tensor.max()
        target_min (float): Desired minimum value (default: 0)
        target_max (float): Desired maximum value (default: 255)
    
    Returns:
        torch.Tensor: Transformed tensor
    """
    # Get source range if not provided
    if source_min is None:
        source_min = tensor.min()
    if source_max is None:
        source_max = tensor.max()
        
    # First normalize to [0,1]
    normalized = (tensor - source_min) / (source_max - source_min)
    
    # Then scale to target range
    transformed = normalized * (target_max - target_min) + target_min
    
    return transformed

def save_image(image_tensor, save_dir, filename=None, index=0):
    try:
        os.makedirs(save_dir, exist_ok=True)
        
        image_pil = transforms.ToPILImage()(image_tensor)
        
        if filename is None:
            filename = f'image_{index}.png'
            
        save_path = os.path.join(save_dir, filename)
        
        image_pil.save(save_path)
        print(f"Image saved successfully to {save_path}")
        
        return save_path
        
    except Exception as e:
        print(f"Error saving image: {str(e)}")
        return None

if __name__ == '__main__':
    train_loader, test_loader = Preprocess.preprocess_dataset(64, 'mnist')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get a sample image and label
    fst_img, _ = train_loader.dataset[0]
    fst_img.to(device)

    # Save the original image before adding noise
    fst_img = transform_range(fst_img, fst_img.min(), fst_img.max(), 0, 1)
    save_image(fst_img, save_dir='saved_images_mnist', filename='original_image.png')
    
    # Add batch dimension to fst_img to make it work with add_noise()
    fst_img = fst_img.unsqueeze(0)

    # Add noise to the image
    T = 100
    betas = torch.linspace(1e-4, 0.02, T, device=device)

    for T_t in [1, 2, 5, 10, 50, 99]:
        t = torch.tensor([T_t], device=device)
        # Summary stats for fst_img
        print(f"Image min: {fst_img.min()}, max: {fst_img.max()}, mean: {fst_img.mean()}, std: {fst_img.std()}")

        fst_img_noisy, noise = add_noise(fst_img, betas, t, device=device)

        # Summary stats for fst_img_noisy
        print(f"Before squeeze: min: {fst_img_noisy.min()}, max: {fst_img_noisy.max()}, mean: {fst_img_noisy.mean()}, std: {fst_img_noisy.std()}")

        # Remove batch dimension and transform to [0,1] range to save the image
        fst_img_noisy = fst_img_noisy.squeeze(0)
        print(f"After squeeze: min: {fst_img_noisy.min()}, max: {fst_img_noisy.max()}, mean: {fst_img_noisy.mean()}, std: {fst_img_noisy.std()}")
        
        fst_img_noisy_normal = transform_range(fst_img_noisy, fst_img_noisy.min(), fst_img_noisy.max(), 0, 1)
        print(f"After transform: min: {fst_img_noisy_normal.min()}, max: {fst_img_noisy_normal.max()}, mean: {fst_img_noisy_normal.mean()}, std: {fst_img_noisy_normal.std()}")


        # Save the noisy image
        save_image(fst_img_noisy_normal, save_dir='saved_images_mnist', filename=f'noisy_{T_t}_image.png')
        #sys.exit(1)