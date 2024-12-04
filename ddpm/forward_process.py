import torch
import sys
import numpy as np

from preprocess import *
import matplotlib.pyplot as plt

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
    ### Test Example ###
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


    ### Example with MNIST data ###
    # Load data #
    train_loader, test_loader = Preprocess.preprocess_dataset(64, 'mnist')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Get a sample image and label
    img, _ = train_loader.dataset[1] # Set to one to get a nice zero
    img.to(device)
 
    # Add batch dimension to img to make it work with add_noise()
    img = img.unsqueeze(0)

    T = 500
    betas = torch.linspace(1e-4, 0.02, T, device=device)

    # Img, means and std as tensor, placeholders
    images = torch.zeros((6, 28, 28))
    times = torch.zeros(6)
    means = torch.zeros(T, device=device)
    stds = torch.zeros(T, device=device)
    counter = 0

    # Mean and std for original image
    means[0] = img.mean()
    stds[0] = img.std()

    # Loop over all timesteps, but skip the first one
    for T_t in range(1, T):
        t = torch.tensor([T_t], device=device)

        # Add noise to the image, remove batch dimension
        img_noisy, _ = add_noise(img, betas, t, device=device)
        img_noisy = img_noisy.squeeze(0)

        # Save mean and std before transforming
        means[T_t] = img_noisy.mean()
        stds[T_t] = img_noisy.std()

        # At certain timesteps, save the noisy image
        if T_t in [9, 19, 49, 99, 249, 499]:
            # Transform the image to [0, 1] to save image
            img_noisy = transform_range(img_noisy, img_noisy.min(), img_noisy.max(), 0, 1)

            # Append the image to the tensor
            images[counter] = img_noisy
            # Add 1 and turn into int
            times[counter] = T_t + 1
            counter += 1

            # Save the noisy image
            save_image(img_noisy, save_dir='poster', filename=f'noisy_{T_t+1}_image.png')

    # Plot the means and stds over time side by side
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(means.detach().cpu().numpy())
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=2)
    plt.xlabel('Timestep')
    plt.ylabel('Mean')
    plt.title('Mean of Noisy Image over Time')
    plt.subplot(1, 2, 2)
    plt.plot(stds.detach().cpu().numpy())
    plt.axhline(y=1, color='gray', linestyle='--', linewidth=2)
    plt.xlabel('Timestep')
    plt.ylabel('Standard Deviation')
    plt.title('Standard Deviation of Noisy Image over Time')
    plt.savefig('poster/mean_std_over_time.png')
    plt.close
    
    # Plot the noisy images in a grid
    fig, axes = plt.subplots(2, 3, figsize=(5, 4), squeeze=False)
    for i, (image, time) in enumerate(zip(images, times)):
        row, col = divmod(i, 3)  # Map index to grid position (row, column)
        
        # Plot the image
        axes[row, col].imshow(image.detach().cpu().numpy(), cmap='gray')
        axes[row, col].axis('off')  # Turn off the axes
        
        # Set the main title
        axes[row, col].set_title(f"Noise to time {int(time)}", fontsize=10, pad=10)
        
        # Add a subtitle below the main title
        axes[row, col].text(0.5, 1.05,
                    f"Mean: {means[int(time) - 1]:.2f}, Std: {stds[int(time) - 1]:.2f}",
                    ha='center', va='center', 
                    transform=axes[row, col].transAxes, fontsize=8, color='#404040')
    plt.savefig('poster/progressive_noise.png')
    plt.close
    