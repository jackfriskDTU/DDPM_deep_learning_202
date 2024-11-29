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

    def __init__(self, in_channels, out_channels, dropout_prob=0.1):
        super(UNet, self).__init__()

        # List of values for the number of parameters in each layer
        self.num_params = [32, 64, 128, 256]
        
        # conv_block consists of two convolutional layers followed by ReLU activations.
        # Adding +1 to the in_channels to account for the time dimension.
        self.encoder1 = self.conv_block(in_channels + 1, self.num_params[0], dropout_prob)
        self.encoder2 = self.conv_block(self.num_params[0] + 1, self.num_params[1], dropout_prob)
        self.encoder3 = self.conv_block(self.num_params[1] + 1, self.num_params[2], dropout_prob)
        self.encoder4 = self.conv_block(self.num_params[2] + 1, self.num_params[3], dropout_prob)
        
        # This line defines a linear layer to embed the time dimension into a 512-dimensional vector.
        # self.time_embed_layer_4 = nn.Linear(1, 512)
        # self.time_embed_layer_3 = nn.Linear(1, self.num_params[2])
        
        # The decoder consists of four conv_block layers followed by a final convolutional layer to output the final image.
        # Adding +1 to the in_channels to account for the time dimension.
        self.decoder4 = self.conv_block(self.num_params[3] + self.num_params[2] + 2, self.num_params[2], dropout_prob)
        self.decoder3 = self.conv_block(self.num_params[2] + self.num_params[1] + 2, self.num_params[1], dropout_prob)
        self.decoder2 = self.conv_block(self.num_params[1] + self.num_params[0] + 2, self.num_params[0], dropout_prob)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(self.num_params[0], out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)#,
            # nn.Dropout(dropout_prob)
        )
        
        # The pooling layer downsamples the input by a factor of 2.
        # The upconvolutional layer upsamples the input by a factor of 2.
        # Adding +1 to the in_channels to account for the time dimension.
        self.pool = nn.MaxPool2d(2)
        self.upconv4 = nn.ConvTranspose2d(self.num_params[3], self.num_params[3], 2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(self.num_params[2], self.num_params[2], 2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(self.num_params[1], self.num_params[1], 2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(self.num_params[0], self.num_params[0], 2, stride=2)

    def conv_block(self, in_channels, out_channels, dropout_prob):
        """A convolutional block consists of two convolutional layers
        followed by ReLU activations."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob)
        )
    
    def add_timestep(self, x, t, batch_size):
        """Add timestep information to the input tensor."""
        t = t.view(batch_size, 1, 1, 1)
        t = t.expand(-1 , 1, x.size(2), x.size(3))

        return torch.cat([x, t], dim=1)

    def forward(self, x, t, verbose=False, layers=3):
        """Forward pass of the U-Net model.
        Verbose mode prints the shape of intermediate tensors."""
        
        # Unsqueeze the time embedding to match the dimensions of the last encoding level
        # while len(t.shape) < len(x.shape):
        #     t = t.unsqueeze(1).float()

        # Concatenate the time embedding as an additional channel
        x = self.add_timestep(x, t, x.size(0))

        if layers == 3:
            enc1 = self.encoder1(x)
            enc1 = self.add_timestep(enc1, t, enc1.size(0))
            enc2 = self.encoder2(self.pool(enc1))
            enc2 = self.add_timestep(enc2, t, enc2.size(0))
            bottleneck = self.encoder3(self.pool(enc2))

            bottleneck_upconv = self.upconv3(bottleneck)
            bottleneck_upconv = torch.cat((bottleneck_upconv, enc2), dim=1)
            bottleneck_upconv = self.add_timestep(bottleneck_upconv, t, bottleneck_upconv.size(0))
            dec2 = self.decoder3(bottleneck_upconv)
            
            dec2_upconv = self.upconv2(dec2)
            dec2_upconv = torch.cat((dec2_upconv, enc1), dim=1)
            dec2_upconv = self.add_timestep(dec2_upconv, t, dec2_upconv.size(0))
            dec1 = self.decoder2(dec2_upconv)
            # We do not add timestep in the 1x1 conv, should we?
            #dec1 = self.add_timestep(dec1, t, dec1.size(0))
            output = self.decoder1(dec1)

        elif layers == 4:
            enc1 = self.encoder1(x)
            enc2 = self.encoder2(self.pool(enc1))
            enc3 = self.encoder3(self.pool(enc2))
            # Practically identical to encoder 4
            bottleneck = self.encoder4(self.pool(enc3))

            bottleneck_upconv = self.upconv4(bottleneck)
            dec3 = self.decoder4(torch.cat((bottleneck_upconv, enc3), dim=1))
            dec3_upconv = self.upconv3(dec3)
            dec2 = self.decoder3(torch.cat((dec3_upconv, enc2), dim=1))
            dec2_upconv = self.upconv2(dec2)
            dec1 = self.decoder2(torch.cat((dec2_upconv, enc1), dim=1))
            output = self.decoder1(dec1)

        

        # Initialize the time embedding
        # t_emb = self.time_embed_layer_3(t)
        # t_emb = t_emb.view(t_emb.size(0), t_emb.size(3), 1, 1)

        # enc1 = self.encoder1(x)
        # enc2 = self.encoder2(self.pool(enc1))
        # enc3 = self.encoder3(self.pool(enc2))
        # if (layers == 4):
        #     enc4 = self.encoder4(self.pool(enc3))

        #     # Embed the time dimension and add it to the output of the fourth encoder block.
        #     t_emb = self.time_embed_layer_4(t)
        #     t_emb = t_emb.view(t_emb.size(0), t_emb.size(3), 1, 1)

        #     enc4 = enc4 + t_emb
        #     dec4 = self.upconv4(enc4)
        #     dec4 = torch.cat((dec4, enc3), dim=1)
        #     dec4 = self.decoder4(dec4)

        #     dec3 = self.upconv3(dec4)
        #     dec3 = torch.cat((dec3, enc2), dim=1)
        #     dec3 = self.decoder3(dec3)
        # else:
        #     enc3 = enc3 + t_emb
        #     dec3 = self.decoder3(enc3)

        # dec2 = self.upconv2(dec3)
        # dec2 = torch.cat((dec2, enc1), dim=1)
        # dec2 = self.decoder2(dec2)
        
        # dec1 = self.upconv1(dec2)

        # # Crop dec1 to match the dimensions of x
        # if dec1.size(2) > x.size(2) or dec1.size(3) > x.size(3):
        #     #diff_x = dec1.size(2) - x.size(2)
        #     print('dec1_before.size() =', dec1.size())

        #     dec1 = dec1[:, :, :x.size(2), :x.size(3)]
        #     print('dec1_after.size() =', dec1.size())

        # dec1 = torch.cat((dec1, x), dim=1)
        # dec1 = self.decoder1(dec1)

        if verbose:
            print('x.size() =', x.size())
            print('enc1.size() =', enc1.size())
            print('enc2.size() =', enc2.size())
            if (layers == 4):
                print('enc3.size() =', enc3.size())
            print('bottleneck.size() =', bottleneck.size())
            if (layers == 4):
                print('dec3.size() =', dec3.size())
            print('dec2.size() =', dec2.size())
            print('dec1.size() =', dec1.size())
            print('output.size() =', output.size())

        return output

def train_model(train_loader, test_loader, model, device, T=1000, beta_lower=1e-4, beta_upper=0.02, learning_rate=1e-3, num_epochs=4, batch_size = 64, early_stopping=False, weight_decay=0.0):
    # Move to device
    #model.to(device)

    # patience
    patience = 10

    # Initiate best_loss
    best_loss = float("inf")

    # best epoch counter
    best_epoch = 0

    # Define the beta schedule
    betas = torch.linspace(beta_lower, beta_upper, T, device=device)

    # Get the optimizer
    optimizer = utils.get_optimizer(model, learning_rate, weight_decay=weight_decay)

    # Set the model to training mode
    model.train()



    # Start training
    for epoch in range(num_epochs):
        # Placeholder to save loss
        losses = []

        # Iterate over batches (image)
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

            # Compute gradients based on the loss from the current batch (backpropagation).
            loss.backward()
            
            # Take one optimizer step using the gradients computed in the previous step.
            optimizer.step()

            # Clean up gradients from the model.
            optimizer.zero_grad()

            # Save the loss
            losses.append(loss.item())

        # Compute the average loss for the epoch
        train_loss = sum(losses) / len(losses)

       # test phase
        model.eval()
        test_loss = []
        with torch.no_grad():
            for batch, _ in test_loader:
                batch = batch.to(device, non_blocking=True)
                t = torch.randint(0, T, (batch_size,), device=device)
                batch_noised, noise = add_noise(batch, betas, t, device)
                batch_noised = batch_noised.to(device)
                noise = noise.to(device)

                predicted_noise = model.forward(batch_noised, t, verbose=False)
                loss = utils.loss_function(predicted_noise, noise)
                test_loss.append(loss.item())

        test_loss = sum(test_loss) / len(test_loss)

        print(f'Epoch {epoch+1}/{num_epochs}, \
              Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, \
                diff: {train_loss - test_loss:.4f}, best_loss: {best_loss:.4f}')

    # Save the model if the validation loss is the best we've seen so far
    # Early stopping
        if early_stopping and test_loss < best_loss:
            best_loss = test_loss
            best_epoch = epoch
            print(f'Validation loss improved. Saving model weights to model_weights/es_{learning_rate}_{batch_size}_{num_epochs}.pt')
            torch.save(model.state_dict(), f'model_weights/es_{learning_rate}_{batch_size}_{num_epochs}.pt')

        elif epoch - best_epoch > 0.1 * num_epochs:
            print(f'Validation loss has not improved for 10 epochs. Best loss: {best_loss:.4f} at epoch {best_epoch+1}')
            patience -= 1
            if patience == 0:
                print('Early stopping')
                break

    print("Finished training.")
    return losses

if __name__ == '__main__':
    # Set seed to get same answer
    torch.manual_seed(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    BATCH_SIZE = 10
    model = UNet(1, 1)
    model.apply(utils.init_weights)
    model.to(device)

    train_loader, _ = Preprocess.preprocess_dataset(BATCH_SIZE, 'mnist')

    losses = train_model(train_loader, model, device, T=1000, beta_lower=1e-4, beta_upper=0.02, 
                         learning_rate=1e-6, num_epochs=10, batch_size = BATCH_SIZE)

    # Plot losses
    plt.plot(losses)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('plots/training_loss.png')

    # Save model
    #torch.save(model.state_dict(), 'model_weights/model_e06.pt')
