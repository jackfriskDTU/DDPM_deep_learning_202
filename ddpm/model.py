import torch
import torch.nn as nn

import sys
import utils
from preprocess import *
from forward_process import *

import matplotlib.pyplot as plt

class UNet(nn.Module):
    """
    U-Net model for image-to-image translation tasks with time embedding.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        batch size (int): Dimension of the time embedding.
    Methods:
        conv_block(in_channels, out_channels):
            Creates a convolutional block with two convolutional layers followed by ReLU activations.
        forward(x, t, verbose=False):
            Forward pass of the U-Net model.
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
                t (torch.Tensor): Time embedding tensor of shape (batch_size).
                verbose (bool): If True, prints the shape of intermediate tensors in the U-net
            Returns:
                torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
            Steps:
                1. Pass the input through the first encoder block.
                2. Pass the result through the pooling layer and then the second encoder block.
                3. Repeat step 2 for the third and fourth encoder blocks.
                4. Embed the time dimension and add it to the output of the fourth encoder block.
                5. Pass the result through the first upconvolutional layer and concatenate with the third encoder block output.
                6. Pass the result through the first decoder block.
                7. Repeat steps 5 and 6 for the second and third decoder blocks.
                8. Pass the result through the fourth upconvolutional layer.
                9. Concatenate the result with the input tensor and pass through the final decoder block.
                10. Return the final output tensor.
    """

    def __init__(self, in_channels, out_channels, batch_size):
        super(UNet, self).__init__()
        
        #  conv_block consists of two 
        # convolutional layers followed by ReLU activations.
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        
        # This line defines a linear layer to embed
        #  the time dimension into a 512-dimensional vector.
        self.time_embed_layer_4 = nn.Linear(batch_size, 512)
        self.time_embed_layer_3 = nn.Linear(batch_size, 256)
        
        # The decoder consists of four conv_block layers
        # followed by a final convolutional layer to output the final image.
        self.decoder4 = self.conv_block(512 + 256, 256)
        self.decoder3 = self.conv_block(256 + 128, 128)
        self.decoder2 = self.conv_block(128 + 64, 64)
        self.decoder1 = self.conv_block(64 + in_channels, out_channels)
        
        # The pooling layer downsamples the input by a factor of 2.
        # The upconvolutional layer upsamples the input by a factor of 2.
        self.pool = nn.MaxPool2d(2)
        self.upconv4 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(64, 64, 2, stride=2)

    def conv_block(self, in_channels, out_channels):
        """A convolutional block consists of two convolutional layers
        followed by ReLU activations."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x, t, verbose=False, layers=3):
        """Forward pass of the U-Net model.
        Verbose mode prints the shape of intermediate tensors."""

        # Initialize the time embedding
        t_emb = self.time_embed_layer_3(t.float())

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        if (layers == 4):
            enc4 = self.encoder4(self.pool(enc3))
            t_emb = self.time_embed_layer_4(t.float())
            

        # Unsqueeze the time embedding to match the dimensions of the last encoding level
        if (layers == 4):
            while len(t_emb.shape) + 1 < len(enc4.shape):
                t_emb = t_emb.unsqueeze(-1)
            enc4 = enc4 + t_emb

            dec4 = self.upconv4(enc4)
            dec4 = torch.cat((dec4, enc3), dim=1)
            dec4 = self.decoder4(dec4)

            dec3 = self.upconv3(dec4)
            dec3 = torch.cat((dec3, enc2), dim=1)
            dec3 = self.decoder3(dec3)
        else:
            while len(t_emb.shape) + 1 < len(enc3.shape):
                t_emb = t_emb.unsqueeze(-1)
            enc3 = enc3 + t_emb

            dec3 = self.upconv3(enc3)
            dec3 = torch.cat((dec3, enc2), dim=1)
            dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc1), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)

        # Crop dec1 to match the dimensions of x
        if dec1.size(2) > x.size(2) or dec1.size(3) > x.size(3):
            dec1 = dec1[:, :, :x.size(2), :x.size(3)]

        dec1 = torch.cat((dec1, x), dim=1)
        dec1 = self.decoder1(dec1)

        if verbose:
            print(f'x shape: {x.shape}')
            print(f'enc1 shape: {enc1.shape}')
            print('enc1.size() =', enc1.size())
            print(f'enc2 shape: {enc2.shape}')
            print('enc2.size() =', enc2.size())
            print(f'enc3 shape: {enc3.shape}')
            print('enc3.size() =', enc3.size())
            if (layers == 4):
                print(f'enc4 shape: {enc4.shape}')
                print('enc4.size() =', enc4.size())
            print(f'dec1 shape: {dec1.shape}')
            print('dec1.size() =', dec1.size())
            print(f'dec2 shape: {dec2.shape}')
            print('dec2.size() =', dec2.size())
            print(f'dec3 shape: {dec3.shape}')
            print('dec3.size() =', dec3.size())
            if (layers == 4):
                print(f'dec4 shape: {dec4.shape}')
                print('dec4.size() =', dec4.size())

        return dec1

def train_model(train_loader, model, device, T=1000, beta_lower=1e-4, beta_upper=0.02, learning_rate=1e-3, num_epochs=4, batch_size = 64):
    # Move to device
    #model.to(device)

    # Define the beta schedule
    betas = torch.linspace(beta_lower, beta_upper, T, device=device)

    # Get the optimizer
    optimizer = utils.get_optimizer(model, learning_rate)

    # Set the model to training mode
    model.train()

    # Placeholder to save loss
    losses = []

    # Start training
    for epoch in range(num_epochs):

        # Iterate over batches
        for batch, _ in train_loader:
            # Send to device
            batch = batch.to(device, non_blocking=True)

            # Generate random timesteps for each image in the batch
            t = torch.randint(0, T, (batch_size,), device=device)

            # Add noise
            batch_noised, noise = add_noise(batch, betas, t, device)
            batch_noised = batch_noised.to(device)
            noise = noise.to(device)

            # Forward pass
            predicted_noise = model.forward(batch_noised, t, verbose=False)

            # Compute loss
            loss = utils.loss_function(predicted_noise, noise)

            # Clean up gradients from the model.
            optimizer.zero_grad()

            # Compute gradients based on the loss from the current batch (backpropagation).
            loss.backward()

            # Take one optimizer step using the gradients computed in the previous step.
            optimizer.step()

            # Save the loss
            losses.append(loss.item())

            # Num epoch
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

    print("Finished training.")
    return losses

if __name__ == '__main__':
    # Set seed to get same answer
    torch.manual_seed(1)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    BATCH_SIZE = 10
    model = UNet(1, 1, BATCH_SIZE)
    model.apply(utils.init_weights)
    model.to(device)

    train_loader, _ = Preprocess.preprocess_dataset(BATCH_SIZE, 'mnist')

    losses = train_model(train_loader, model, device, T=1000, beta_lower=1e-4, beta_upper=0.02, 
                         learning_rate=1e-4, num_epochs=100, batch_size = BATCH_SIZE)

    # Plot losses
    plt.plot(losses)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('plots/training_loss.png')

    # Save model
    torch.save(model.state_dict(), 'model_weights/model.pt')
