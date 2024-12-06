import hydra
from omegaconf import DictConfig

import sys
import torch
import random
import matplotlib.pyplot as plt

import neptune

from model import UNet, train_model
from utils import set_project_root, init_weights, get_optimizer, loss_function
from preprocess import Preprocess 
from postprocess import sample_and_plot, save_image, transform_range
from forward_process import add_noise
from reverse_process import sample
from new_u import ScoreNetwork0, train_model0

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
    model = UNet(in_channels, out_channels, 0)
    model.to(device)

    if mode_train:
        model.apply(init_weights)

        # Load the training data
        train, test = Preprocess.preprocess_dataset(batch_size, dataset, train_size, test_size)

        # Train the model
        train_model(train, test, model, device, time_dim, beta_lower, beta_upper,\
                     learning_rate, lr_scheduler, epochs, batch_size, early_stopping,\
                         optimizer, weight_decay, run)       
        
        print(f"Saving model weights to {train_size}_{test_size}_{optimizer}_{weight_decay}_{learning_rate}_{lr_scheduler}_{batch_size}_{epochs}_{early_stopping}_{seed}_{time_dim}_{dataset}.pt")
        torch.save(model.state_dict(),\
                    f'model_weights/{train_size}_{test_size}_{optimizer}_{weight_decay}_{learning_rate}_{lr_scheduler}_{batch_size}_{epochs}_{early_stopping}_{seed}_{time_dim}_{dataset}.pt')
        
        # Remove train and test from memory
        del train
        del test

    if mode_sample:
        if early_stopping:
            print(f"predicting with es_{learning_rate}_{batch_size}_{epochs}.pt")
            model.load_state_dict(torch.load(f'model_weights/es_{learning_rate}_{batch_size}_{epochs}.pt',
                                         map_location=torch.device('cuda'),
                                         weights_only=True))

        else:
            print(f"predicting with {train_size}_{test_size}_{optimizer}_{weight_decay}_{learning_rate}_{lr_scheduler}_{batch_size}_{epochs}_{early_stopping}_{seed}_{time_dim}_{dataset}.pt")
            # Load the model weights
            model.load_state_dict(torch.load(\
                f'model_weights/{train_size}_{test_size}_{optimizer}_{weight_decay}_{learning_rate}_{lr_scheduler}_{batch_size}_{epochs}_{True}_{seed}_{time_dim}_{dataset}.pt',
                                        map_location=torch.device('cuda'),
                                        weights_only=True))
        model.eval()

        betas = torch.linspace(beta_lower, beta_upper, time_dim, device=device)

        if dataset == 'mnist':
            shape = (sample_size, in_channels, 28, 28)
        elif dataset == 'cifar10':
            shape = (sample_size, in_channels, 32, 32)

        
        if sample_size > 1:# Save the sampled image
            sample_and_plot(model, betas, shape, device, train_size, test_size, optimizer, weight_decay, learning_rate, lr_scheduler, batch_size, epochs, early_stopping, seed, time_dim, dataset)

        else:
            sampled_img = sample(model, time_dim, betas, shape, device, stepwise=False)
            sampled_img = sampled_img[0]
            sampled_img = transform_range(sampled_img, sampled_img.min(), sampled_img.max(), 0, 1)

            # Save the sampled image       
            save_image(sampled_img, save_dir=f'saved_images_{dataset}', filename=f'{train_size}_{test_size}_{optimizer}_{weight_decay}_{learning_rate}_{lr_scheduler}_{batch_size}_{epochs}_{early_stopping}_{seed}_{time_dim}_{dataset}.png')

        # ten_sample = sampled_img[:10]
        # # Plot the 10 sampled images
        # fig, axes = plt.subplots(1, 10, figsize=(15, 3), squeeze=False)
        # axes = axes[0]
        # for i, img in enumerate(ten_sample):
        #     img = transform_range(img, img.min(), img.max(), 0, 1)
        #     img = img.permute(1, 2, 0)
        #     axes[i].imshow(img.detach().cpu().numpy(), cmap='gray')
        #     axes[i].axis('off')
        # fig.savefig(f'saved_images_{dataset}/{early_stopping}_{seed}_{learning_rate}_{batch_size}_{epochs}_{dataset}_{weight_decay}_sampled_image.png')

        # sampled_img = sampled_img[0]
        # sampled_img = transform_range(sampled_img, sampled_img.min(), sampled_img.max(), 0, 1)

        # Save the sampled image       
        # save_image(sampled_img, save_dir=f'saved_images_{dataset}', filename=f'{early_stopping}_{seed}_{learning_rate}_{batch_size}_{epochs}_{dataset}_{weight_decay}_sampled_image_trans.png')

        # ten_sample = sampled_img[:10]
        # # Plot the 10 sampled images
        # fig, axes = plt.subplots(1, 10, figsize=(15, 3), squeeze=False)
        # axes = axes[0]
        # for i, img in enumerate(ten_sample):
        #     img = transform_range(img, img.min(), img.max(), 0, 1)
        #     img = img.permute(1, 2, 0)
        #     axes[i].imshow(img.detach().cpu().numpy(), cmap='gray')
        #     axes[i].axis('off')
        # fig.savefig(f'saved_images_{dataset}/{early_stopping}_{seed}_{learning_rate}_{batch_size}_{epochs}_{dataset}_{weight_decay}_sampled_image.png')

        # sampled_img = sampled_img[0]
        # sampled_img = transform_range(sampled_img, sampled_img.min(), sampled_img.max(), 0, 1)

        # Save the sampled image       
        # save_image(sampled_img, save_dir=f'saved_images_{dataset}', filename=f'{early_stopping}_{seed}_{learning_rate}_{batch_size}_{epochs}_{dataset}_{weight_decay}_sampled_image_trans.png')

if __name__ == "__main__":
    main()

