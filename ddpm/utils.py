from pathlib import Path
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau, CosineAnnealingLR

# Define the project root directory
PROJECT_ROOT = Path(__file__).resolve()

def set_project_root():
    global PROJECT_ROOT
    PROJECT_ROOT = PROJECT_ROOT

    while PROJECT_ROOT.name != 'DDPM_deep_learning_202':
        PROJECT_ROOT = PROJECT_ROOT.parent

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)

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
    return nn.MSELoss()(predicted_noise, noise)

def get_optimizer(model, optimizer_type = "Adam", learning_rate=1e-3, weight_decay=0):
    """
    Returns an Adam optimizer with the learning rate specified in the config file.
    """

    if optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)

    return optimizer

def get_scheduler(optimizer, scheduler_type):
    """
    Create a learning rate scheduler for the given optimizer.

    Args:
        optimizer: The optimizer for which to create the scheduler.
        scheduler_config (string): type of learning rate scheduler.

    Returns:
        scheduler: The learning rate scheduler.
    """
    if scheduler_type == 'StepLR':
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    elif scheduler_type == 'ExponentialLR':
        scheduler = ExponentialLR(optimizer, gamma=0.9)
    elif scheduler_type == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            verbose=True
        )
    elif scheduler_type == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=50,
            eta_min=1e-6
        )
    
    return scheduler