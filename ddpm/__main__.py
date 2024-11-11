import hydra
from omegaconf import DictConfig

import argparse
import os
import sys
import torch

from model import SimpleNN
from utils import set_project_root
from validate import validate

@hydra.main(config_path = "../config_files", config_name = "config", version_base = None)
def main(cfg: DictConfig):

    # Set the project root directory
    set_project_root()

    # Define the network
    input_size = cfg.input_size
    hidden_size = cfg.hidden_size
    output_size = cfg.output_size
    learning_rate = cfg.learning_rate
    batch_size = cfg.batch_size
    epochs = cfg.epochs

    model = SimpleNN(input_size, hidden_size, output_size)

    # Create a dummy input tensor
    input_tensor = torch.randn(1, input_size)

    # Forward pass
    output = model(input_tensor)
    print(output)

    # Create dummy validation data
    validation_data = [(torch.randn(1, input_size), torch.tensor([1])) for _ in range(5)]

    # Validate the model
    validate(model, validation_data)

if __name__ == "__main__":
    main()

