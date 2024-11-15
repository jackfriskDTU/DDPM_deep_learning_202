import hydra
from omegaconf import DictConfig

import argparse
import os
import sys
import torch

from model import UNet
from utils import set_project_root
from validate import validate

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


    model = UNet(in_channels, out_channels, time_dim)

    # test the model
    x = torch.randn(batch_size, in_channels, input_size, input_size)
    t = torch.randint(0, 10, (1, batch_size))
    y = model(x, t, verbose=False)

    # # Create dummy validation data
    # validation_data = [(torch.randn(1, input_size), torch.tensor([1])) for _ in range(5)]

    # # Validate the model
    # validate(model, validation_data)

if __name__ == "__main__":
    main()

