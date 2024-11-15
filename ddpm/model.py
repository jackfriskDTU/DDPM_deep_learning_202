import torch
import torch.nn as nn


class UNet(nn.Module):
    """
    U-Net model for image-to-image translation tasks with time embedding.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        time_dim (int): Dimension of the time embedding.
    Methods:
        conv_block(in_channels, out_channels):
            Creates a convolutional block with two convolutional layers followed by ReLU activations.
        forward(x, t, verbose=False):
            Forward pass of the U-Net model.
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
                t (torch.Tensor): Time embedding tensor of shape (batch_size, time_dim).
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

    def __init__(self, in_channels, out_channels, time_dim):
        super(UNet, self).__init__()
        
        #  conv_block consists of two 
        # convolutional layers followed by ReLU activations.
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        
        # This line defines a linear layer to embed
        #  the time dimension into a 512-dimensional vector.
        self.time_embed = nn.Linear(time_dim, 512)
        
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
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, t, verbose=False):
        """Forward pass of the U-Net model.
        Verbose mode prints the shape of intermediate tensors."""
        if verbose:
            print(f'x shape: {x.shape}')
        enc1 = self.encoder1(x)
        if verbose:
            print(f'enc1 shape: {enc1.shape}')
        enc2 = self.encoder2(self.pool(enc1))
        if verbose:
            print(f'enc2 shape: {enc2.shape}')
        enc3 = self.encoder3(self.pool(enc2))
        if verbose:
            print(f'enc3 shape: {enc3.shape}')
        enc4 = self.encoder4(self.pool(enc3))
        if verbose:
            print(f'enc4 shape: {enc4.shape}')

        # Convert t to float
        t = t.float()

        # Initialize the time embedding
        t_emb = self.time_embed(t)
        # unsqueeze the time embedding to match the dimensions of enc4
        while len(t_emb.shape) < len(enc4.shape):
            t_emb = t_emb.unsqueeze(-1) 
        
        # t_emb = t_emb.long()
        enc4 = enc4 + t_emb
        
        if verbose:
            print("-----------------")
            print(f'enc4 shape: {enc4.shape}')
        dec4 = self.upconv4(enc4)
        dec4 = torch.cat((dec4, enc3), dim=1)
        dec4 = self.decoder4(dec4)
        
        if verbose:
            print(f'dec4 shape: {dec4.shape}')
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc2), dim=1)
        dec3 = self.decoder3(dec3)

        if verbose:
            print(f'dec3 shape: {dec3.shape}')
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc1), dim=1)
        dec2 = self.decoder2(dec2)
        
        if verbose:
            print(f'dec2 shape: {dec2.shape}')
        dec1 = self.upconv1(dec2)
        
        # Crop dec1 to match the dimensions of x
        if dec1.size(2) > x.size(2) or dec1.size(3) > x.size(3):
            dec1 = dec1[:, :, :x.size(2), :x.size(3)]

        dec1 = torch.cat((dec1, x), dim=1)
        dec1 = self.decoder1(dec1)    

        if verbose:
            print(f'dec1 shape: {dec1.shape}')

        return dec1


if __name__ == '__main__':
    model = UNet(3, 3, 2)
    x = torch.rand(2, 3, 64, 64)
    t = torch.randint(1, 5, (1, 2)) 
    print(type(t))
    print(t)
    y = model(x, t, verbose=True)
