from pathlib import Path
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
import math

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

def get_scheduler(optimizer, scheduler_type, num_epochs):
    """
    Create a learning rate scheduler for the given optimizer.

    Args:
        optimizer: The optimizer for which to create the scheduler.
        scheduler_config (string): type of learning rate scheduler.

    Returns:
        scheduler: The learning rate scheduler or None if no scheduler is selected.
    """
    if scheduler_type == 'StepLR':
        scheduler = StepLR(optimizer, step_size=num_epochs//10, gamma=0.5)
    elif scheduler_type == 'ExponentialLR':
        scheduler = ExponentialLR(optimizer, gamma=0.9)
    elif scheduler_type == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=15
        )
    elif scheduler_type == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_epochs - num_epochs//8,
            eta_min=1e-9
        )
    
    elif scheduler_type == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=num_epochs//8,
            T_mult=2,
            eta_min=1e-9
        )
    elif scheduler_type == 'None':
        scheduler = None
    
    return scheduler

def get_beta_schedule(schedule_type, T, device, beta_lower=1e-4, beta_upper=0.02):
    if schedule_type == "Linear":
        # Classic linear schedule from DDPM
        betas = torch.linspace(beta_lower, beta_upper, T, device=device)
        return betas
    
    elif schedule_type == "Cosine":
        # Cosine schedule from "Improved Denoising Diffusion Probabilistic Models"
        s = 0.008  # small offset
        steps = torch.arange(T, device=device, dtype=torch.float)
        
        # Compute alpha_bar at each timestep
        alpha_bar = (torch.cos(((steps/T) + s) / (1 + s) * math.pi/2) ** 2)
        
        alpha_bar_shifted = torch.cat([torch.tensor([1.0], device=device), alpha_bar[:-1]])
        alphas = alpha_bar / alpha_bar_shifted
        betas = 1 - alphas
        # Clip betas to ensure numerical stability
        betas = torch.clamp(betas, min=1e-8, max=0.999)
        return betas
    else:
        raise ValueError(f"Unknown beta scheduler type: {schedule_type}")
    
def sinusoidal_embedding(t, embed_dim):
    """
    Generates sinusoidal embeddings for the time step.
    Args:
        t (torch.Tensor): Input tensor of shape (batch_size,).
        embed_dim (int): Dimensionality of the embedding.
    Returns:
        torch.Tensor: Sinusoidal embeddings of shape (batch_size, embed_dim).
    """
    device = t.device
    half_dim = embed_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    emb = t[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return emb 

class TimeEmbedding(nn.Module):
    def __init__(self, time_embed_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim * 4),
            nn.SiLU(),
        )

    def forward(self, t):
        # t: (batch_size,)
        emb = sinusoidal_embedding(t, self.mlp[0].in_features)
        # emb shape: (batch_size, time_embed_dim)
        emb = self.mlp(emb)
        return emb