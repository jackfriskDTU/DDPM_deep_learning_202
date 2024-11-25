import hydra
from omegaconf import DictConfig

import sys
import torch

from model import UNet, train_model
from utils import set_project_root, init_weights, get_optimizer, loss_function
from preprocess import Preprocess, save_image, transform_range
from forward_process import add_noise
from reverse_process import sample

@hydra.main(config_path = "../config_files", config_name = "config", version_base = None)
def main(cfg: DictConfig):

    # Set the project root directory
    set_project_root()

    # Define the mode
    mode_train = cfg.mode.train
    mode_sample = cfg.mode.sample
    dataset = cfg.mode.dataset

    # Define the model parameters
    in_channels = cfg.model.in_channels
    out_channels = cfg.model.out_channels
    time_dim = cfg.model.time_dim
    seed = cfg.model.seed

    # Define the training parameters
    learning_rate = cfg.training.learning_rate
    batch_size = cfg.training.batch_size
    epochs = cfg.training.epochs
    beta_lower = cfg.training.beta_lower
    beta_upper = cfg.training.beta_upper

    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(seed)

    # initialize the U-net model
    model = UNet(in_channels, out_channels)
    model.to(device)

    if mode_train:
        model.apply(init_weights)

        # Load the training data
        train, _ = Preprocess.preprocess_dataset(batch_size, dataset)

        # Train the model
        train_model(train, model, device, time_dim, beta_lower, beta_upper, learning_rate, epochs, batch_size)       
        
        torch.save(model.state_dict(), f'model_weights/main_{time_dim}_{seed}_{learning_rate}_{batch_size}_{epochs}_{dataset}.pt')

    if mode_sample:
        # Load the model weights
        model.load_state_dict(torch.load(f'model_weights/main_{time_dim}_{seed}_{learning_rate}_{batch_size}_{epochs}_{dataset}.pt',
                                         map_location=torch.device('cuda:0'),
                                         weights_only=True))
        model.eval()

        betas = torch.linspace(beta_lower, beta_upper, time_dim, device=device)

        if dataset == 'mnist':
            shape = (1, in_channels, 28, 28)
        elif dataset == 'cifar10':
            shape = (1, in_channels, 64, 64)

        # Sample from the model
        sampled_img = sample(model, time_dim, betas, shape, device)
        sampled_img = sampled_img[0]

        # Save the image
        save_image(sampled_img, save_dir=f'saved_images_{dataset}', filename=f'{seed}_sampled_image.png')
        sampled_img = transform_range(sampled_img, sampled_img.min(), sampled_img.max(), 0, 1)
        save_image(sampled_img, save_dir=f'saved_images_{dataset}', filename=f'{seed}_sampled_image_trans.png')

if __name__ == "__main__":
    main()

