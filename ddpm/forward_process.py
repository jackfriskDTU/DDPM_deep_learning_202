import torch
import sys
import numpy as np

def add_noise(df, betas, t, device):
    """
    Adds noise to a tensor `df` for some timestep `t` with `betas` as the beta schedule.

    Parameters:
        df (torch.Tensor): Input tensor of shape (b, c, h, w), where 'b' is batch_size, 'c' is channels, 'h' is height of the data and 'w' is width.
        betas (torch.Tensor): Noise schedule of shape (T), where T is the total number of timesteps.
        t (torch.Tensor): A timestep for each image in the batch. A tensor of shape (b,).

    Returns:
        torch.Tensor: The input tensor, but with added standard normally distributed noise for one time step.
    """
    df = df.to(device)
    # Convert to alpha to allow closed-form calculation of the noise scale (i.e. all noise in one go, not stepwise)
    alphas = 1 - betas
    alphas_cummulative = torch.cumprod(alphas, dim=0).to(device)

    # Get the cummulative alpha for the current timestep
    alpha_t_bar = alphas_cummulative[t]

    # Generate standard normal noise of the same shape as `df`
    noise = torch.randn_like(df, device=device)

    # Scale the values of df (x0) as in equation (4), N(sqrt(a) * mu, (1-a)*I)
    # Return the input tensor with added noise
    df_noise = df * torch.sqrt(alpha_t_bar).view(-1, 1, 1, 1) \
            + (noise * (1 - alpha_t_bar).view(-1, 1, 1, 1))
    return df_noise, noise

if __name__ == "__main__":
    # Set seed to get same answer
    torch.manual_seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Example setup
    B, C, H, W = 2, 1, 3, 3  # Batch size, channels, height, width
    T = 10  # Number of timesteps
    df = torch.randn(B, C, H, W)  # Input tensor
    betas = torch.linspace(1e-4, 0.02, T)  # Example linear beta schedule
    t = torch.randint(0, T, (B,)) # Random timesteps for each batch element

    # Add noise
    df_noised, noise = add_noise(df, betas, t, device)
    print('df:', df)
    print('noise:', noise)
    print('df_noised:', df_noised)