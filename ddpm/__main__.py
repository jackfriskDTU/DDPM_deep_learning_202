import hydra
from omegaconf import DictConfig

import sys
import torch
import random

import neptune

from .models import UNet, train_model
from .utils import set_project_root, init_weights, get_beta_schedule
from .preprocess import Preprocess 
from .postprocess import sample_and_plot, save_image, transform_range
from .reverse_process import sample

@hydra.main(config_path = "../config_files", config_name = "config", version_base = None)
def main(cfg: DictConfig):

    # Set the project root directory
    set_project_root()

    # Define the mode
    mode_train = cfg.mode.train
    mode_sample = cfg.mode.sample
    dataset = cfg.mode.dataset
    sample_size = cfg.mode.sample_size

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
    beta_scheduler = cfg.training.beta_scheduler
    beta_lower = cfg.training.beta_lower
    beta_upper = cfg.training.beta_upper
    early_stopping = cfg.training.early_stopping
    neptune_log = cfg.training.neptune

    torch.manual_seed(seed)
    random.seed(seed)

    file_name = f'{train_size}_{test_size}_{optimizer}_{weight_decay}_{learning_rate}_{lr_scheduler}_{batch_size}_{epochs}_{beta_scheduler}_{seed}_{time_dim}_{dataset}'

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
    model = UNet(in_channels, out_channels, 0)
    model.to(device)

    if mode_train:
        model.apply(init_weights)

        # Load the training data
        train, test = Preprocess.preprocess_dataset(batch_size, dataset, train_size, test_size)

        # Train the model
        train_model(train, test, model, device, file_name, time_dim, beta_lower, beta_upper,\
                     learning_rate, lr_scheduler, epochs, beta_scheduler, batch_size,\
                        early_stopping, optimizer, weight_decay, run)       

        print(f"Saving model weights to {file_name}_False.pt")
        torch.save(model.state_dict(), f'model_weights/{file_name}_False.pt')
        
        # Remove train and test from memory
        del train
        del test

    if mode_sample:
        if early_stopping:
            print(f"predicting with {file_name}_{early_stopping}.pt")
            model.load_state_dict(torch.load(f'model_weights/{file_name}_{early_stopping}.pt',
                                         map_location=torch.device('cuda'),
                                         weights_only=True))

        else:
            print(f"predicting with {file_name}_False.pt")
            # Load the model weights
            model.load_state_dict(torch.load(f'model_weights/{file_name}_False.pt',
                                        map_location=torch.device('cuda'),
                                        weights_only=True))
        model.eval()

        betas = get_beta_schedule(beta_scheduler, time_dim, device, beta_lower=1e-4, beta_upper=0.02)

        if dataset == 'mnist':
            shape = (sample_size, in_channels, 28, 28)
        elif dataset == 'cifar10':
            shape = (sample_size, in_channels, 32, 32)

        with torch.no_grad():
            if sample_size > 1:
                sample_and_plot(model, betas, shape, device, time_dim, file_name, early_stopping, dataset, beta_scheduler=beta_scheduler)

            else:
                sampled_img = sample(model, time_dim, betas, shape, device, stepwise=False, dataset=dataset, beta_scheduler=beta_scheduler)
                sampled_img = sampled_img[0]
                sampled_img = transform_range(sampled_img, sampled_img.min(), sampled_img.max(), 0, 1)

                # Save the sampled image       
                save_image(sampled_img, save_dir=f'saved_images_{dataset}', filename=f'{file_name}_{early_stopping}.png')

if __name__ == "__main__":
    main()
