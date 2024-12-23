import torch
import pandas as pd

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
               + noise * torch.sqrt(1 - alpha_t_bar).view(-1, 1, 1, 1)

    return df_noise, noise

if __name__ == "__main__":
    from postprocess import *
    from preprocess import *
    from utils import get_beta_schedule
    
    ### Test Example ###
    torch.manual_seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### To generate plots ###
    beta_scheduler = 'Linear'
    dataset = 'cifar10'
    T = 750
    betas = get_beta_schedule(beta_scheduler, T, device, beta_lower=1e-4, beta_upper=0.02)

    # Load data #
    train_loader, test_loader = Preprocess.preprocess_dataset(64, dataset)
    img, _ = train_loader.dataset[1] # Set to one to get a nice zero
    img.to(device)
 
    # Add batch dimension to img to make it work with add_noise()
    img = img.unsqueeze(0)
    
    # Img, means and std as tensor, placeholders
    if dataset == 'mnist':
        images = torch.zeros((6, 28, 28))
    elif dataset == 'cifar10':
        images = torch.zeros((6, 3, 32, 32))
    times = torch.zeros(6)
    means = torch.zeros(T, device=device)
    stds = torch.zeros(T, device=device)
    counter = 0

    # Add the original image
    images[counter] = transform_range(img.squeeze(0), img.squeeze(0).min(), img.squeeze(0).max(), 0, 1)
    times[counter] = 0
    means[counter] = img.mean()
    stds[counter] = img.std()

    # These are the timesteps to save the images
    timestamps = [99, 199, 299, 499, 749] # [19, 49, 99, 249, 499], [99, 199, 299, 499, 749]

    # Loop over all timesteps
    for T_t in range(T):
        t = torch.tensor([T_t], device=device)

        # Add noise to the image, remove batch dimension
        img_noisy, _ = add_noise(img, betas, t, device=device)
        img_noisy = img_noisy.squeeze(0)

        # Save mean and std before transforming
        means[T_t] = img_noisy.mean()
        stds[T_t] = img_noisy.std()

        # At certain timesteps, save the noisy image
        if T_t in timestamps:
            counter += 1

            # Transform the image to [0, 1] to save image
            img_noisy = transform_range(img_noisy, img_noisy.min(), img_noisy.max(), 0, 1)

            # Append the image to the tensor
            images[counter] = img_noisy

            # Add 1 to nullify the 0-indexing
            times[counter] = T_t + 1

    # Read in other set of means and stds from .csv file (requires that referse_process.py has been run)
    data = pd.read_csv(f'poster/mean_std_over_time_denoising_{dataset}_{beta_scheduler}.csv')

    # Extract the columns into variables
    mean_denoising = data['Mean'].values
    std_denoising = data['Std'].values

    # Plot the means and stds over time side by side
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(means.detach().cpu().numpy())
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=2)
    plt.xlabel('Timestep', fontsize=14)
    plt.ylabel('Mean', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.title('Mean of Image over Time', fontsize=16)
    plt.subplot(1, 2, 2)
    plt.plot(stds.detach().cpu().numpy())
    plt.axhline(y=1, color='gray', linestyle='--', linewidth=2)
    plt.xlabel('Timestep', fontsize=14)
    plt.ylabel('Standard Deviation', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.title('Std Dev of Image over Time', fontsize=16)
    plt.savefig(f'poster/mean_std_over_time_diffusion_{dataset}_{beta_scheduler}.png')
    plt.close

    # Plot the means and stds over time side by side with denoising as well
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(means.detach().cpu().numpy(), label = 'Forward')
    plt.plot(mean_denoising, label = 'Reverse')
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=2)
    plt.xlabel('Timestep', fontsize=14)
    plt.ylabel('Mean', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.title('Mean of Image over Time', fontsize=16)
    plt.legend(fontsize=12)
    plt.subplot(1, 2, 2)
    plt.plot(stds.detach().cpu().numpy(), label = 'Forward')
    plt.plot(std_denoising, label = 'Reverse')
    plt.axhline(y=1, color='gray', linestyle='--', linewidth=2)
    plt.xlabel('Timestep', fontsize=14)
    plt.ylabel('Standard Deviation', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.title('Std Dev of Image over Time', fontsize=16)
    plt.legend(fontsize=12)
    plt.savefig(f'poster/mean_std_over_time_both_{dataset}_{beta_scheduler}.png')
    plt.close
    
    # Plot the noisy images in a grid
    fig, axes = plt.subplots(2, 3, figsize=(5, 4), squeeze=False)
    for i, (image, time) in enumerate(zip(images, times)):
        row, col = divmod(i, 3)  # Map index to grid position (row, column)

        # Permute the image to (H, W, C) for plotting
        if dataset == 'cifar10':
            image = image.permute(1, 2, 0)
        
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
    plt.savefig(f'poster/progressive_noise_diffusion_{dataset}_{beta_scheduler}_{T}.png')
    plt.close
    