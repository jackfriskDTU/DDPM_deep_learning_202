import torch
import torch.nn as nn

import sys
import utils
import math
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

        # If MNIST
        if in_channels == 1:
            self.image_shape = 28
            self.layers = 2
        else:
            self.image_shape = 32
            self.layers = 3

        # List of values for the number of parameters in each layer
        self.num_params = [32, 64, 128, 256]

        # Linear layer to project time embedding to match input channels
        self.time_linear_in = nn.Linear(self.image_shape, in_channels)
        self.time_linear_1 = nn.Linear(self.num_params[0], self.num_params[0])
        self.time_linear_2 = nn.Linear(self.num_params[1], self.num_params[1])
        self.time_linear_3 = nn.Linear(self.num_params[2], self.num_params[2])
        self.time_linear_4 = nn.Linear(self.num_params[3], self.num_params[3])
        
        # conv_block consists of two convolutional layers followed by ReLU activations.
        self.encoder1 = self.conv_block(in_channels, self.num_params[0], dropout_prob)
        self.encoder2 = self.conv_block(self.num_params[0], self.num_params[1], dropout_prob)
        self.encoder3 = self.conv_block(self.num_params[1], self.num_params[2], dropout_prob)
        self.encoder4 = self.conv_block(self.num_params[2], self.num_params[3], dropout_prob)
               
        # The decoder consists of four conv_block layers followed by a final convolutional layer to output the final image.
        self.decoder4 = self.conv_block(self.num_params[3] + self.num_params[2], self.num_params[2], dropout_prob)
        self.decoder3 = self.conv_block(self.num_params[2] + self.num_params[1], self.num_params[1], dropout_prob)
        self.decoder2 = self.conv_block(self.num_params[1] + self.num_params[0], self.num_params[0], dropout_prob)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(self.num_params[0], out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)#,
            # nn.Dropout(dropout_prob)
        )
        
        # The pooling layer downsamples the input by a factor of 2.
        # The upconvolutional layer upsamples the input by a factor of 2.
        self.pool = nn.MaxPool2d(2)
        self.upconv4 = nn.ConvTranspose2d(self.num_params[3], self.num_params[3], kernel_size = 2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(self.num_params[2], self.num_params[2], kernel_size = 2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(self.num_params[1], self.num_params[1], kernel_size = 2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(self.num_params[0], self.num_params[0], kernel_size = 2, stride=2)

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
    
    def sinusoidal_embedding(self, t, embed_dim):
        """
        Generates sinusoidal embeddings for the time step.
        Args:
            t (torch.Tensor): Input tensor of shape (batch_size,).
            embed_dim (int): Dimensionality of the embedding.
        Returns:
            torch.Tensor: Sinusoidal embeddings of shape (batch_size, embed_dim).
        """
        device = t.device
        half_dim = embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

    def forward(self, x, t, verbose=False):
        """Forward pass of the U-Net model.
        Verbose mode prints the shape of intermediate tensors."""
        
        # Generate sinusoidal time embeddings
        time_emb_in = self.sinusoidal_embedding(t, self.image_shape)
        time_emb_in = self.time_linear_in(time_emb_in).unsqueeze(-1).unsqueeze(-1)
        time_emb_1 = self.sinusoidal_embedding(t, self.num_params[0])
        time_emb_1 = self.time_linear_1(time_emb_1).unsqueeze(-1).unsqueeze(-1)
        time_emb_2 = self.sinusoidal_embedding(t, self.num_params[1])
        time_emb_2 = self.time_linear_2(time_emb_2).unsqueeze(-1).unsqueeze(-1)
        time_emb_3 = self.sinusoidal_embedding(t, self.num_params[2])
        time_emb_3 = self.time_linear_3(time_emb_3).unsqueeze(-1).unsqueeze(-1)
        time_emb_4 = self.sinusoidal_embedding(t, self.num_params[3])
        time_emb_4 = self.time_linear_4(time_emb_4).unsqueeze(-1).unsqueeze(-1)

        x = x + time_emb_in

        if self.layers == 2:
            enc1 = self.encoder1(x)
            enc1 = enc1 + time_emb_1
            enc2 = self.encoder2(self.pool(enc1))
            enc2 = enc2 + time_emb_2

            bottleneck = self.encoder3(self.pool(enc2))
            bottleneck_upconv = self.upconv3(bottleneck)
            bottleneck_upconv = torch.cat((bottleneck_upconv, enc2), dim=1)

            dec2 = self.decoder3(bottleneck_upconv)
            dec2_upconv = self.upconv2(dec2)
            dec2_upconv = torch.cat((dec2_upconv, enc1), dim=1)
            dec1 = self.decoder2(dec2_upconv)
            output = self.decoder1(dec1)

        elif self.layers == 3:
            enc1 = self.encoder1(x)
            enc1 = enc1 + time_emb_1
            enc2 = self.encoder2(self.pool(enc1))
            enc2 = enc2 + time_emb_2
            enc3 = self.encoder3(self.pool(enc2))
            enc3 = enc3 + time_emb_3

            bottleneck = self.encoder4(self.pool(enc3))
            bottleneck_upconv = self.upconv4(bottleneck)
            bottleneck_upconv = torch.cat((bottleneck_upconv, enc3), dim=1)
            dec3 = self.decoder4(bottleneck_upconv)
            dec3_upconv = self.upconv3(dec3)
            dec3_upconv = torch.cat((dec3_upconv, enc2), dim=1)
            dec2 = self.decoder3(dec3_upconv)
            dec2_upconv = self.upconv2(dec2)
            dec2_upconv = torch.cat((dec2_upconv, enc1), dim=1)
            dec1 = self.decoder2(dec2_upconv)
            output = self.decoder1(dec1)


        if verbose:
            print('x.size() =', x.size())
            print('enc1.size() =', enc1.size())
            print('enc2.size() =', enc2.size())
            if (self.layers == 3):
                print('enc3.size() =', enc3.size())
            print('bottleneck.size() =', bottleneck.size())
            if (self.layers == 3):
                print('dec3.size() =', dec3.size())
            print('dec2.size() =', dec2.size())
            print('dec1.size() =', dec1.size())
            print('output.size() =', output.size())

        return output

def train_model(train_loader,\
                 test_loader,\
                   model,\
                     device,\
                       T=1000,\
                         beta_lower=1e-4,\
                           beta_upper=0.02,\
                             learning_rate=1e-3,\
                               lr_scheduler = "None",\
                                 num_epochs=4,\
                                   batch_size = 64,\
                                     early_stopping=False,\
                                       optimizer = "Adam",\
                                         weight_decay=0.0,\
                                           neptune_log=False):
    # Move to device
    #model.to(device)

    if neptune_log:
        params = {"learning_rate": learning_rate,
                "lr_scheduler": lr_scheduler,
                "optimizer": optimizer,
                "weight_decay": weight_decay,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                }
        neptune_log["parameters"] = params

    # patience
    patience = 10

    # Initiate best_loss
    best_loss = float("inf")

    # best epoch counter
    best_epoch = 0

    # Define the beta schedule
    betas = torch.linspace(beta_lower, beta_upper, T, device=device)

    # Get the optimizer
    optimizer = utils.get_optimizer(model, optimizer, learning_rate, weight_decay=weight_decay)

    lr_scheduler = utils.get_scheduler(optimizer, lr_scheduler)

    # Start training
    for epoch in range(num_epochs):
        # Set the model to training mode
        model.train()

        # Placeholder to save loss
        losses = []

        # Iterate over batches (image)
        for batch, _ in train_loader:
            # Send to device
            batch = batch.to(device, non_blocking=True)

            # Clean up gradients from the model.
            optimizer.zero_grad()

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

            # Save the loss
            losses.append(loss.item())

        # Compute the average loss for the epoch
        train_loss = sum(losses) / len(losses)
 

        # test phase
        # Set the model to evaluation mode
        model.eval()
        
        # Placeholder to save test loss
        test_loss = []
        
        # Disable gradient calculation for validation
        with torch.no_grad():
            # Iterate over batches in the test loader
            for batch, _ in test_loader:
                # Send batch to device
                batch = batch.to(device, non_blocking=True)
                
                # Generate random timesteps for each image in the batch
                t = torch.randint(0, T, (batch_size,), device=device)
                
                # Add noise to the batch
                batch_noised, noise = add_noise(batch, betas, t, device)
                batch_noised = batch_noised.to(device)
                noise = noise.to(device)

                # Forward pass
                predicted_noise = model.forward(batch_noised, t, verbose=False)
                
                # Compute loss
                loss = utils.loss_function(predicted_noise, noise)
                
                # Save the loss
                test_loss.append(loss.item())

        # Compute the average test loss
        test_loss = sum(test_loss) / len(test_loss)

        # Log learning rate to Neptune
        if neptune_log:
            neptune_log["train/learning_rate"].log(optimizer.param_groups[0]['lr'])
            neptune_log["train/train"].log(train_loss)
            neptune_log["train/test"].log(test_loss)

        # Print the training and test loss for the current epoch
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
            
            # if neptune_log:
            #     neptune_log["model/best_model"].upload(best_epoch)

        # stop if the test loss is not improving after x percentage of total epochs
        elif early_stopping and (epoch - best_epoch > 0.6 * num_epochs):
            print(f'Validation loss has not improved for {0.6 * num_epochs} epochs. Best loss: {best_loss:.4f} at epoch {best_epoch+1}')
            patience -= 1
            if patience == 0:
                print('Early stopping')
                break

        if lr_scheduler == "ReduceLROnPlateau":
            lr_scheduler.step(test_loss)
        elif lr_scheduler != None:
            lr_scheduler.step()

    print("Finished training.")

    if neptune_log:

        neptune_log.stop()

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
