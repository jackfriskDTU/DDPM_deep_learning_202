import hydra
from omegaconf import DictConfig

import argparse
import os
import sys
import torch

from model import UNet
from utils import set_project_root
from validate import validate
import preprocess
from preprocess import *
from preprocess import Preprocess

@hydra.main(config_path = "../config_files", config_name = "config", version_base = None)
def main(cfg: DictConfig):

    # Set the project root directory
    set_project_root()

    # Define the network
    input_size = cfg.model.input_size
    hidden_size = cfg.model.hidden_size
    output_size = cfg.model.output_size
    in_channels = cfg.model.in_channels
    out_channels = cfg.model.out_channels
    time_dim = cfg.model.time_dim
    learning_rate = cfg.training.learning_rate
    batch_size = cfg.training.batch_size
    epochs = cfg.training.epochs


    # initialize the U-net model
    model = UNet(in_channels, out_channels, batch_size)

    # test the model
    # x = torch.randn(batch_size, in_channels, input_size, input_size)
    
    train_loader, test_loader = Preprocess.preprocess_dataset(64, 'cifar10')

    # Get a sample image and label
    fst_img, fst_label = train_loader.dataset[0]

    # Save the original image before adding noise
    fst_img = transform_range(fst_img, -1, 1, 0, 1)
    # save_path = save_image(fst_img, save_dir='saved_images_cifar10')
    
    print('fst_img:', fst_img)
    print('fst_img shape:', fst_img.shape)
    print('fst_img label:', fst_label)

    # Add batch dimension to fst_img to make it work with add_noise()
    fst_img = fst_img.unsqueeze(0)
    betas = torch.linspace(1e-4, 0.02, time_dim)

    t = torch.randint(0, time_dim, (batch_size,))
    fst_img_noisy = add_noise(fst_img, betas, t)

    # Remove batch dimension and transform to [0,1] range to save the image
    # fst_img_noisy = fst_img_noisy.squeeze(0)
    fst_img_noisy_normal = transform_range(fst_img_noisy, -1, 1, 0, 1)

    # Save the noisy image
    # save_path = save_image(fst_img_noisy_normal, save_dir='saved_images_cifar10_noise') 
    y = model(fst_img_noisy, t, verbose=False)

    # # Create dummy validation data
    # validation_data = [(torch.randn(1, input_size), torch.tensor([1])) for _ in range(5)]

    # # Validate the model
    # validate(model, validation_data)

if __name__ == "__main__":
    main()

