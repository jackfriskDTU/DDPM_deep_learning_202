import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
import utils
from forward_process import add_noise
from preprocess import Preprocess

class ScoreNetwork0(torch.nn.Module):
    # takes an input image and time, returns the score function
    def __init__(self):
        super().__init__()
        nch = 2
        chs = [32, 64, 128, 256, 256]
        self._convs = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(2, chs[0], kernel_size=3, padding=1),  # (batch, ch, 28, 28)
                torch.nn.LogSigmoid(),  # (batch, 8, 28, 28)
            ),
            torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=2, stride=2),  # (batch, ch, 14, 14)
                torch.nn.Conv2d(chs[0], chs[1], kernel_size=3, padding=1),  # (batch, ch, 14, 14)
                torch.nn.LogSigmoid(),  # (batch, 16, 14, 14)
            ),
            torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=2, stride=2),  # (batch, ch, 7, 7)
                torch.nn.Conv2d(chs[1], chs[2], kernel_size=3, padding=1),  # (batch, ch, 7, 7)
                torch.nn.LogSigmoid(),  # (batch, 32, 7, 7)
            ),
            torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),  # (batch, ch, 4, 4)
                torch.nn.Conv2d(chs[2], chs[3], kernel_size=3, padding=1),  # (batch, ch, 4, 4)
                torch.nn.LogSigmoid(),  # (batch, 64, 4, 4)
            ),
            torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=2, stride=2),  # (batch, ch, 2, 2)
                torch.nn.Conv2d(chs[3], chs[4], kernel_size=3, padding=1),  # (batch, ch, 2, 2)
                torch.nn.LogSigmoid(),  # (batch, 64, 2, 2)
            ),
        ])
        self._tconvs = torch.nn.ModuleList([
            torch.nn.Sequential(
                # input is the output of convs[4]
                torch.nn.ConvTranspose2d(chs[4], chs[3], kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, 64, 4, 4)
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[3]
                torch.nn.ConvTranspose2d(chs[3] * 2, chs[2], kernel_size=3, stride=2, padding=1, output_padding=0),  # (batch, 32, 7, 7)
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[2]
                torch.nn.ConvTranspose2d(chs[2] * 2, chs[1], kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, chs[2], 14, 14)
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[1]
                torch.nn.ConvTranspose2d(chs[1] * 2, chs[0], kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, chs[1], 28, 28)
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[0]
                torch.nn.Conv2d(chs[0] * 2, chs[0], kernel_size=3, padding=1),  # (batch, chs[0], 28, 28)
                torch.nn.LogSigmoid(),
                torch.nn.Conv2d(chs[0], 1, kernel_size=3, padding=1),  # (batch, 1, 28, 28)
            ),
        ])

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (..., ch0 * 28 * 28), t: (..., 1)
        # x2 = torch.reshape(x, (*x.shape[:-1], 1, 28, 28))  # (..., ch0, 28, 28)
        # tt = t[..., None, None].expand(*t.shape[:-1], 1, 28, 28)  # (..., 1, 28, 28)
        # x2t = torch.cat((x2, tt), dim=-3)
        # signal = x2t

        batch_size, _, height, width = x.shape

        # Ensure that the time tensor has the same batch size
        t = t.view(batch_size, 1, 1, 1).expand(batch_size, 1, height, width)

        # Concatenate input image `x` with the expanded time tensor `t` along the channel dimension
        x2t = torch.cat((x, t), dim=1)  # (batch, 2, height, width)
        signal = x2t
        signals = []
        for i, conv in enumerate(self._convs):
            signal = conv(signal)
            if i < len(self._convs) - 1:
                signals.append(signal)

        for i, tconv in enumerate(self._tconvs):
            if i == 0:
                signal = tconv(signal)
            else:
                signal = torch.cat((signal, signals[-i]), dim=-3)
                signal = tconv(signal)
        # signal = torch.reshape(signal, (*signal.shape[:-3], -1))  # (..., 1 * 28 * 28)
        return signal

def train_model0(train_loader,
                test_loader,
                model,
                device,
                T=1000,
                beta_lower=1e-4,
                beta_upper=0.02,
                learning_rate=1e-3,
                lr_scheduler_type="StepLR",
                num_epochs=4,
                batch_size=64,
                early_stopping=False,
                optimizer_type="Adam",
                weight_decay=0.0,
                neptune_log=False):
    # Define the optimizer
    if optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Optimizer type '{optimizer_type}' not supported.")

    # Define the learning rate scheduler
    if lr_scheduler_type == 'StepLR':
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    elif lr_scheduler_type == 'ExponentialLR':
        scheduler = ExponentialLR(optimizer, gamma=0.9)
    elif lr_scheduler_type == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    elif lr_scheduler_type == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    else:
        scheduler = None

    # Define the beta schedule
    betas = torch.linspace(beta_lower, beta_upper, T, device=device)

    # Set patience for early stopping
    patience = 10
    best_loss = float("inf")
    best_epoch = 0

    for epoch in range(num_epochs):
        model.train()
        train_losses = []

        for batch, _ in train_loader:
            batch = batch.to(device, non_blocking=True)

            # Clean up gradients from the model
            optimizer.zero_grad()

            # Generate random timesteps for each image in the batch
            t = torch.randint(0, T, (batch_size,), device=device)

            # Add noise to the input batch
            batch_noised, noise = add_noise(batch, betas, t, device)

            # Forward pass
            predicted_noise = model(batch_noised, t)

            # Compute loss (Mean Squared Error between predicted and actual noise)
            loss = torch.nn.functional.mse_loss(predicted_noise, noise)

            # Backward pass and optimization step
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # Calculate average train loss for the epoch
        avg_train_loss = sum(train_losses) / len(train_losses)

        # Validation phase
        model.eval()
        test_losses = []

        with torch.no_grad():
            for batch, _ in test_loader:
                batch = batch.to(device, non_blocking=True)
                t = torch.randint(0, T, (batch_size,), device=device)
                batch_noised, noise = add_noise(batch, betas, t, device)

                predicted_noise = model(batch_noised, t)
                loss = torch.nn.functional.mse_loss(predicted_noise, noise)
                test_losses.append(loss.item())

        avg_test_loss = sum(test_losses) / len(test_losses)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Best Loss: {best_loss:.4f}')

        # Early stopping logic
        if early_stopping and avg_test_loss < best_loss:
            best_loss = avg_test_loss
            best_epoch = epoch
            print(f'Validation loss improved. Saving model weights...')
            torch.save(model.state_dict(), f'model_weights/best_model_epoch_{epoch+1}.pt')
            patience = 10  # Reset patience

        elif early_stopping and (epoch - best_epoch > 0.6 * num_epochs):
            patience -= 1
            if patience == 0:
                print('Early stopping due to lack of improvement in validation loss')
                break

        # Scheduler step
        if lr_scheduler_type == "ReduceLROnPlateau":
            scheduler.step(avg_test_loss)
        elif scheduler is not None:
            scheduler.step()

    print("Finished training.")

    return train_losses

# Example usage
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 10
    model = ScoreNetwork0()  # Use ScoreNetwork0 instead of UNet
    model.apply(utils.init_weights)
    model.to(device)

    train_loader, test_loader = Preprocess.preprocess_dataset(BATCH_SIZE, 'mnist')

    losses = train_model(train_loader,
                         test_loader,
                         model,
                         device,
                         T=1000,
                         beta_lower=1e-4,
                         beta_upper=0.02,
                         learning_rate=1e-3,
                         lr_scheduler_type="StepLR",
                         num_epochs=10,
                         batch_size=BATCH_SIZE,
                         early_stopping=True)

    # Plot losses
    import matplotlib.pyplot as plt
    plt.plot(losses)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('plots/training_loss.png')
