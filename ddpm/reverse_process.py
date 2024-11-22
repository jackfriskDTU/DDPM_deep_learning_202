import torch
import model
from preprocess import *
def sample(model, timesteps, betas, shape, device):
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

    # Start from Gaussian noise
    x_t = torch.randn(shape, device=device)

    for t in reversed(range(timesteps)):
        # Ensure t is a tensor
        t_tensor = torch.tensor([t] * shape[0], device=device)

        if t > 0:
            z = torch.randn(shape, device=device)
        else:
            z = torch.zeros(shape, device=device)

        sigma_t = torch.sqrt(betas[t])

        # Predict the noise
        predicted_noise = model.forward(x_t, t_tensor)

        # Calculate the posterior mean estimate for x_{t-1}
        alpha_t = alphas[t]
        alpha_t_bar = alphas_cumulative[t] #if t > 0 else torch.tensor(1.0)

        frac = (1 - alpha_t) / torch.sqrt(1- alpha_t_bar)

        x_t = (1 / torch.sqrt(alpha_t)) * (x_t - frac * predicted_noise) + sigma_t * z

        # if t % 10 == 0:
        #     sampled_img = x_t[0]
        #     save_image(sampled_img, save_dir=f'saved_images_sample', filename=f'{t}_sampled_image.png')
        #     sampled_img = transform_range(sampled_img, sampled_img.min(), sampled_img.max(), 0, 1)
        #     save_image(sampled_img, save_dir=f'saved_images_sample', filename=f'{t}_sampled_image_trans.png')

    return x_t

if __name__ == "__main__":
    # Set seed to get same answer
    seed = 1
    torch.manual_seed(seed)

    # Example setup
    B, C, H, W = 64, 1, 28, 28  # Batch size, channels, height, width
    T = 100  # Number of timesteps
    betas = torch.linspace(1e-4, 0.02, T)  # Example linear beta schedule
    shape = (B, C, H, W)
    
    # Load the model weights
    model = model.UNet(C, C, B)  # Adjust the parameters as needed
    model.load_state_dict(torch.load('model_weights/model_e06.pt', map_location=torch.device('cuda'), weights_only=True))
    model.eval()

    # Sample from the model
    sampled_img = sample(model, T, betas, shape)
    sampled_img = sampled_img[0]

    sampled_img = transform_range(sampled_img, sampled_img.min(), sampled_img.max(), 0, 1)

    # Save the image
    save_image(sampled_img, save_dir='saved_images', filename=f'{seed}_3updated_sampled_image.png')