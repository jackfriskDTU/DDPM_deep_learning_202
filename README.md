# DDPM Implementation

This repository contains a custom **Denoising Diffusion Probabilistic Model (DDPM)** implementation built in PyTorch.  
We train and sample images on **MNIST** (28×28, grayscale) and **CIFAR-10** (32×32, RGB) datasets using a **U-Net** architecture that adapts to the number of channels and image sizes.

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Training](#training)
- [Sampling](#sampling)
- [Citation](#citation)

---

## Features
- **U-Net Architecture** with variable depth depending on MNIST vs. CIFAR-10.
- **Sinusoidal Time Embeddings** for conditioning the U-Net on the diffusion timestep.
- **Flexible Beta Schedules** (linear and cosine).
- **Learning Rate Schedulers** (e.g., StepLR, ReduceLROnPlateau).
- **Mixed Precision Training** for faster inference and reduced memory usage.
- **Preprocessing** pipelines for MNIST and CIFAR-10, normalizing images to \([-1, 1]\).

---

## Project Structure

```
DDPM_Repo/
├─ ddpm/
│  ├─ __init__.py
│  ├─ __main__.py           # Entry point (if using python -m ddpm)
│  ├─ model.py              # U-Net model, training loop
│  ├─ utils.py              # Helper functions (loss, scheduling, time embedding)
│  ├─ forward_process.py    # Code to add noise to images
│  ├─ reverse_process.py    # Code to sample (reverse diffusion)
│  ├─ preprocess.py         # Loading & normalizing MNIST/CIFAR-10
│  ├─ postprocess.py        # Image saving, plotting, etc.
├─ config_files/
│  └─ config.yaml           # Main configuration (hyperparams, dataset, scheduler)
├─ tutorial.ipynb           # Example notebook showcasing usage
├─ weights/                 # Folder where trained weights are saved
├─ model_weights/           # Folder where new trained model weights are saved
├─ environment.yml          # Python dependencies
└─ README.md                # This readme
```

- **`ddpm/`**: Main Python package with code for training, sampling, and models.
- **`config_files/`**: YAML configs containing hyperparameters, dataset choices, and scheduler settings.
- **`weights/`**: Pretrained weights.
- **`notebooks/`**: Jupyter notebooks for demo.

---

## Installation

1. **Clone the Repo**:
   ```bash
   git clone https://github.com/jackfriskDTU/DDPM_deep_learning_202.git
   cd DDPM_deep_learning_202
   ```

2. **Install Dependencies**:
   ```bash
   conda install -f environment.yml
   ```

---

## Configuration

We use simple YAML files (in `config_files/`) to handle parameters. For example, `config.yaml` contain:

```yaml
defaults:
  - _self_

mode:
  train: False
  sample: True
  dataset: "cifar10" # "mnist" or "cifar10"
  sample_size: 10

model:
  in_channels: 3
  out_channels: 3
  time_dim: 750
  seed: 42

training:
  train_size: 25600 # cifar10: 49920, mnist: 48000
  test_size: 2560 # cifar10: 8320, mnist: 9600
  optimizer: Adam # Adam, SGD
  weight_decay: 0.0 # 1e-4, 1e-2
  learning_rate: 0.01
  lr_scheduler: ReduceLROnPlateau # None, StepLR, ExponentialLR, ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
  batch_size: 128
  epochs: 200
  beta_scheduler: "Cosine" # Linear, Cosine
  beta_lower: 0.0001 
  beta_upper: 0.02
  early_stopping: False
  neptune: False
  ...
```

- **mode**: dataset choice, whether to train, sample, etc.
- **model**: channels, diffusion timesteps, etc.
- **training**: hyperparams like learning rate, optimizer, batch size, epochs, beta schedule.

You can create additional YAML files for different experiments (e.g., a linear beta schedule, different LR).

---

## Usage

### 1. Running as module

- **Train** using the default config:
  ```bash
  python -m ddpm
  ```
  
- **Sample** after training:
  turn off training in the config
  ```bash
  python -m ddpm
  ```

### 3. Notebook Demo
Open `tutorial.ipynb` to see a step-by-step example of:
- Loading the model,
- Training on a small subset,
- Sampling from the trained model,
- Traning a new model.

---

## Training

1. **Prepare Data**  
   The `Preprocess` class automatically downloads MNIST/CIFAR-10 and normalizes images to \([-1,1]\).

2. **Call the `train_model`** function  
   Found in `ddpm/model.py`, it handles epochs, forward pass, noise addition, loss computation, optimizer steps, and optional LR scheduling.

3. **Checkpointing**  
   Model weights are saved to `model_weights/` after each epoch if early stopping is enabled or after final training completion.

---

## Sampling

Use `ddpm.reverse_process.sample` to:
1. Initialize random noise `x_T`.
2. Iteratively denoise until an image `x_0` emerges.
3. Save or visualize the result.  

---

## Citation

This repository is adapted with inspiration from the original DDPM paper:

```
@inproceedings{ho2020denoising,
  title={Denoising Diffusion Probabilistic Models},
  author={Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}
```

---

**Thank you for checking out our DDPM Implementation!** Feel free to open an issue or submit a PR if you have any questions or improvements.
