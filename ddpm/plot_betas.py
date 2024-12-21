import torch
import math
import matplotlib.pyplot as plt
# Parameters
T = 1000  # Number of timesteps
s = 0.008  # Small offset
device = 'cpu'  # Device

# Generate steps
steps = torch.arange(T, device=device, dtype=torch.float)

# Compute alpha_bar at each timestep
alpha_bar = (torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2)

# Compute shifted alpha_bar and alphas
alpha_bar_shifted = torch.cat([torch.tensor([1.0], device=device), alpha_bar[:-1]])
alphas = alpha_bar / alpha_bar_shifted

# Compute betas
betas = 1 - alphas

# Clip betas for numerical stability
betas = torch.clamp(betas, min=1e-8, max=0.999)

betas_linear = torch.linspace(1e-4, 0.02, T, device=device)

# Plot betas
plt.figure(figsize=(10, 6))
plt.plot(betas.numpy(), label='Betas Cosine')
plt.plot(betas_linear.numpy(), label='Betas Linear')
plt.title('Betas over Timesteps')
plt.xlabel('Timesteps')
plt.ylabel('Beta')
plt.legend()
plt.grid(True)
plt.savefig('poster/betas.png')
