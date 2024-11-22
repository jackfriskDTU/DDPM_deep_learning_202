from pathlib import Path
from torch import nn
from torch import optim

# Define the project root directory
PROJECT_ROOT = Path(__file__).resolve()

def set_project_root():
    global PROJECT_ROOT
    PROJECT_ROOT = PROJECT_ROOT

    while PROJECT_ROOT.name != 'DDPM_deep_learning_202':
        PROJECT_ROOT = PROJECT_ROOT.parent

def init_weights(model):
    if isinstance(model, nn.Conv2d):
        # Initialize weights using He initialization
        nn.init.kaiming_normal_(model.weight, mode='fan_out', nonlinearity='leaky_relu')
        if model.bias is not None:
            nn.init.zeros_(model.bias)
    elif isinstance(model, nn.Linear):
        # Apply He initialization for linear layers as well
        nn.init.kaiming_normal_(model.weight, mode='fan_out', nonlinearity='leaky_relu')
        if model.bias is not None:
            nn.init.zeros_(model.bias)

def loss_function(predicted_noise, noise):
    """
    Computes the MSE loss for predicting the noise added during the forward process.
    Args:
        predicted_noise (torch.Tensor): Predicted noise tensor of shape (batch_size, channels, height, width).
        noise (torch.Tensor): True noise tensor of shape (batch_size, channels, height, width).
    Returns:
        MSE loss between the predicted noise and true noise.
    """

    # Compute MSE loss between predicted noise and true noise
    mse_loss = nn.MSELoss()
    loss = mse_loss(predicted_noise, noise)

    return loss

def get_optimizer(model, learning_rate=1e-3):
    """
    Returns an Adam optimizer with the learning rate specified in the config file.
    """

    # Create an Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    return optimizer