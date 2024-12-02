import matplotlib.pyplot as plt
import os
from PIL import Image
from torchvision import transforms
from reverse_process import sample

def transform_range(tensor, source_min=None, source_max=None, target_min=0, target_max=255):
    """
    Transform tensor from [source_min, source_max] to [target_min, target_max] range.
    If source_min/max are None, they are taken from the tensor.
    
    Args:
        tensor (torch.Tensor): Input tensor
        source_min (float): Current minimum value. If None, uses tensor.min()
        source_max (float): Current maximum value. If None, uses tensor.max()
        target_min (float): Desired minimum value (default: 0)
        target_max (float): Desired maximum value (default: 255)
    
    Returns:
        torch.Tensor: Transformed tensor
    """
    # Get source range if not provided
    if source_min is None:
        source_min = tensor.min()
    if source_max is None:
        source_max = tensor.max()
        
    # First normalize to [0,1]
    normalized = (tensor - source_min) / (source_max - source_min)
    
    # Then scale to target range
    transformed = normalized * (target_max - target_min) + target_min
    
    return transformed

def sample_and_plot(model, time_dim, betas, shape, device, dataset, early_stopping, seed, learning_rate, batch_size, epochs, weight_decay):
    # Sample from the model
    sampled_img = sample(model, time_dim, betas, shape, device)

    ten_sample = sampled_img[:10]
    # Plot the 10 sampled images
    fig, axes = plt.subplots(1, 10, figsize=(15, 3), squeeze=False)
    axes = axes[0]
    for i, img in enumerate(ten_sample):
        img = transform_range(img, img.min(), img.max(), 0, 1)
        img = img.permute(1, 2, 0)
        axes[i].imshow(img.detach().cpu().numpy(), cmap='gray')
        axes[i].axis('off')
    fig.savefig(f'saved_images_{dataset}/{early_stopping}_{seed}_{learning_rate}_{batch_size}_{epochs}_{dataset}_{weight_decay}_sampled_image.png')

def save_image(image_tensor, save_dir, filename=None, index=0):
    try:
        os.makedirs(save_dir, exist_ok=True)
        
        image_pil = transforms.ToPILImage()(image_tensor)
        
        if filename is None:
            filename = f'image_{index}.png'
            
        save_path = os.path.join(save_dir, filename)
        
        image_pil.save(save_path)
        print(f"Image saved successfully to {save_path}")
        
        return save_path
        
    except Exception as e:
        print(f"Error saving image: {str(e)}")
        return None