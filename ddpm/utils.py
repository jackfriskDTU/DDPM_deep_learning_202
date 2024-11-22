from pathlib import Path

# Define the project root directory
PROJECT_ROOT = Path(__file__).resolve()

def set_project_root():
    global PROJECT_ROOT
    PROJECT_ROOT = PROJECT_ROOT

    while PROJECT_ROOT.name != 'DDPM_deep_learning_202':
        PROJECT_ROOT = PROJECT_ROOT.parent

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
    loss = nn.MSELoss(predicted_noise, noise)

    return loss