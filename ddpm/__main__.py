import hydra
from omegaconf import DictConfig

import sys
import torch
import random

import neptune

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
    train_size = cfg.training.train_size
    test_size = cfg.training.test_size
    optimizer = cfg.training.optimizer
    weight_decay = float(cfg.training.weight_decay)
    learning_rate = cfg.training.learning_rate
    lr_scheduler = cfg.training.lr_scheduler
    batch_size = cfg.training.batch_size
    epochs = cfg.training.epochs
    beta_lower = cfg.training.beta_lower
    beta_upper = cfg.training.beta_upper
    early_stopping = cfg.training.early_stopping
    neptune_log = cfg.training.neptune

    torch.manual_seed(seed)
    random.seed(seed)

    # Initialize Neptune
    run = None
    if neptune_log:
        run = neptune.init_run(
        project="s194527/DDPM",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3ZTA1YjFjNS1hNDdmLTQ3OTktOWIxMi01YjNhMWI3NGNmMGEifQ==",
    ) 

    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # initialize the U-net model
    model = UNet(in_channels, out_channels)
    model.to(device)

    if mode_train:
        model.apply(init_weights)

        # Load the training data
        train, test = Preprocess.preprocess_dataset(batch_size, dataset, train_size, test_size)

        # Train the model
        train_model(train, test, model, device, time_dim, beta_lower, beta_upper,\
                     learning_rate, lr_scheduler, epochs, batch_size, early_stopping,\
                         optimizer, weight_decay, run)       
        
        print(f"Saving model weights to main_{time_dim}_{seed}_{learning_rate}_{batch_size}_{epochs}_{dataset}_{weight_decay}.pt")
        torch.save(model.state_dict(),\
                    f'model_weights/main_{early_stopping}_{time_dim}_{seed}_{learning_rate}_{batch_size}_{epochs}_{dataset}_{weight_decay}.pt')

    if mode_sample:
        if early_stopping:
            print(f"predicting with es_{learning_rate}_{batch_size}_{epochs}.pt")
            model.load_state_dict(torch.load(f'model_weights/es_{learning_rate}_{batch_size}_{epochs}.pt',
                                         map_location=torch.device('cuda'),
                                         weights_only=True))

        else:
            print(f"predicting with main_{learning_rate}_{time_dim}_{seed}_{learning_rate}_{batch_size}_{epochs}_{dataset}_{weight_decay}.pt")
            # Load the model weights
            model.load_state_dict(torch.load(\
                f'model_weights/main_{early_stopping}_{time_dim}_{seed}_{learning_rate}_{batch_size}_{epochs}_{dataset}_{weight_decay}.pt',
                                        map_location=torch.device('cuda'),
                                        weights_only=True))
        model.eval()

        betas = torch.linspace(beta_lower, beta_upper, time_dim, device=device)

        if dataset == 'mnist':
            shape = (1, in_channels, 28, 28)
        elif dataset == 'cifar10':
            shape = (1, in_channels, 32, 32)

        # Sample from the model
        sampled_img = sample(model, time_dim, betas, shape, device)
        sampled_img = sampled_img[0]
        sampled_img = transform_range(sampled_img, sampled_img.min(), sampled_img.max(), 0, 1)

        # Save the sampled image       
        save_image(sampled_img, save_dir=f'saved_images_{dataset}', filename=f'{early_stopping}_{seed}_{learning_rate}_{batch_size}_{epochs}_{dataset}_{weight_decay}_sampled_image_trans.png')

if __name__ == "__main__":
    main()

