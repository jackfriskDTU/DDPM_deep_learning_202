import torch
import model
from preprocess import *

import matplotlib.pyplot as plt
import numpy as np
def sample(model, timesteps, betas, shape, device, stepwise):
    """
    Samples a new image from the learned reverse process.
    Args:
        model: The trained U-Net model.
        timesteps: Number of diffusion steps.
        betas: Noise schedule of shape (T), where T is the total number of timesteps.
        shape: Shape of the output image, e.g. (1, 28, 28) for MNIST.
    Returns:
        A denoised generated image.
    """
    # Convert to alpha to allow closed-form calculation of the noise scale
    alphas = 1 - betas
    alphas_cumulative = torch.cumprod(alphas, dim=0).to(device)

    # Make df of shape shape
    df = torch.randn(shape, device=device)

    # Start from Gaussian noise
    x_t = torch.randn_like(df, device=device)

    if stepwise: # This if is not needed, but to clarify it is used for stepwise plotting
        # Placeholders for saving at different timesteps
        images = torch.zeros((6, shape[2], shape[3]), device=device)
        times = torch.zeros(6)
        means = torch.zeros(timesteps, device=device)
        stds = torch.zeros(timesteps, device=device)
        counter = 0

        # Mean and std for gaussian noise
        means[0] = x_t.mean()
        stds[0] = x_t.std()

    for t in reversed(range(timesteps)):
        # Ensure t is a tensor
        t_tensor = torch.tensor([t] * shape[0], device=device)

        if t > 0:
            z = torch.randn_like(x_t, device=device)
        else:
            z = torch.zeros(shape, device=device)

        sigma_t = torch.sqrt(betas[t])

        # Predict the noise
        predicted_noise = model.forward(x_t, t_tensor)

        # # Calculate the posterior mean estimate for x_{t-1}
        alpha_t = alphas[t]
        alpha_t_bar = alphas_cumulative[t] #if t > 0 else torch.tensor(1.0)

        frac = (betas[t]) / torch.sqrt(1 - alpha_t_bar)   

        x_t = (1 / torch.sqrt(alpha_t)) * (x_t - frac * predicted_noise) + (sigma_t * z)

        if stepwise:
            # Add the mean and std to the tensors to plot
            means[t] = x_t.mean()
            stds[t] = x_t.std()

            if t in [499, 199, 149, 99, 49, 0]:
                # Transform the image to [0, 1] to save image
                img_denoise = transform_range(x_t, x_t.min(), x_t.max(), 0, 1)

                # Remove batch dimension
                img_denoise = img_denoise.squeeze(0)

                # Append the image to the tensor
                images[counter] = img_denoise

                # Add 1 to nullify the 0-indexing
                times[counter] = t + 1

                counter += 1

                # Save the noisy image
                save_image(img_denoise, save_dir='poster', filename=f'denoise_{t+1}_image.png')
        torch.cuda.empty_cache()

    if stepwise:
        # Plot the means and stds over time side by side
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(means.detach().cpu().numpy())
        plt.axhline(y=0, color='gray', linestyle='--', linewidth=2)
        plt.xlabel('Timestep', fontsize=14)
        plt.ylabel('Mean', fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.gca().invert_xaxis()
        plt.title('Mean of denoising over Time', fontsize=16)

        plt.subplot(1, 2, 2)
        plt.plot(stds.detach().cpu().numpy())
        plt.axhline(y=1, color='gray', linestyle='--', linewidth=2)
        plt.xlabel('Timestep', fontsize=14)
        plt.ylabel('Standard Deviation', fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.gca().invert_xaxis()
        plt.title('Std Dev of denoising over Time', fontsize=16)
        plt.savefig('poster/mean_std_over_time_denoising.png')
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
        plt.savefig('poster/progressive_noise_denoising.png')
        plt.close
    return x_t

if __name__ == "__main__":
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set seed to get same answer
    seed = 1
    torch.manual_seed(seed)

    # Example setup
    B, C, H, W = 1, 1, 28, 28  # Batch size, channels, height, width
    T = 500  # Number of timesteps
    betas = torch.linspace(1e-4, 0.02, T)  # Example linear beta schedule
    shape = (B, C, H, W)
    
    # Load the model weights
    model = model.UNet(C, C)  # Adjust the parameters as needed
    model.to(device)
    model.load_state_dict(torch.load('model_weights/main_False_500_1_0.001_100_40_mnist_0.0.pt', map_location=torch.device('cuda'), weights_only=True))
    model.eval()

    # Sample from the model
    sampled_img = sample(model, T, betas, shape, device, stepwise=True)
    # sampled_img = sampled_img[0]

    # sampled_img = transform_range(sampled_img, sampled_img.min(), sampled_img.max(), 0, 1)

    ## Save the image
    # save_image(sampled_img, save_dir='saved_images_reverse', filename=f'{seed}_3updated_sampled_image.png')