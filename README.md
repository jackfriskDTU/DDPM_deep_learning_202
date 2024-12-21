Below is a concise **README.md** template for your GitHub repository, giving an overview of the project structure, how to install and run the code, and how the configuration system works. Feel free to adjust folder names, references, and commands to match your actual setup.

---

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
- **Mixed Precision Training** (optional) for faster inference and reduced memory usage.
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
│  ├─ config.yaml           # Main configuration (hyperparams, dataset, scheduler)
│  ├─ other_configs.yaml    # Additional config variations
├─ notebooks/
│  ├─ ddpm_demo.ipynb       # Example notebook showcasing usage
├─ model_weights/           # Folder where trained weights are saved
├─ requirements.txt         # Python dependencies
├─ README.md                # This readme
└─ setup.py (optional)      # For pip install -e .
```

- **`ddpm/`**: Main Python package with code for training, sampling, and models.
- **`config_files/`**: YAML configs containing hyperparameters, dataset choices, and scheduler settings.
- **`model_weights/`**: Pretrained or intermediate weights saved during training.
- **`notebooks/`**: Jupyter notebooks for demos or experiments.

---

## Installation

1. **Clone the Repo**:
   ```bash
   git clone https://github.com/yourusername/DDPM_Repo.git
   cd DDPM_Repo
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   If you use conda or another environment manager, create an environment first, then install.

3. **(Optional) Editable Install**:
   ```bash
   pip install -e .
   ```
   This lets you do `import ddpm` from anywhere, and helps with relative imports.

---

## Configuration

We use simple YAML files (in `config_files/`) to handle parameters. For example, `config.yaml` might contain:

```yaml
mode:
  dataset: cifar10
  train: true
  sample: false
  sample_size: 4

model:
  in_channels: 3
  out_channels: 3
  time_dim: 1000

training:
  train_size: 50000
  test_size: 10000
  optimizer: Adam
  weight_decay: 0.0
  learning_rate: 1e-4
  lr_scheduler: StepLR
  batch_size: 128
  epochs: 50
  beta_lower: 1e-4
  beta_upper: 0.02
  early_stopping: false
  beta_scheduler: cosine
  ...
```

- **mode**: dataset choice, whether to train, sample, etc.
- **model**: channels, diffusion timesteps, etc.
- **training**: hyperparams like learning rate, optimizer, batch size, epochs, beta schedule.

You can create additional YAML files for different experiments (e.g., a linear beta schedule, different LR).

---

## Usage

### 1. Command-Line Execution

- **Train** using the default config:
  ```bash
  python ddpm/__main__.py \
    mode.train=true mode.sample=false \
    model.in_channels=3 training.epochs=50
  ```
  *(If you have Hydra or a similar library, adapt accordingly.)*

- **Sample** after training:
  ```bash
  python ddpm/__main__.py \
    mode.train=false mode.sample=true \
    mode.sample_size=8
  ```

*(Adjust keys to match your config structure.)*

### 2. Running as a Module
From the project root:
```bash
python -m ddpm
```
This entry point typically looks for a `config.yaml` or uses command-line overrides.

### 3. Notebook Demo
Open `notebooks/ddpm_demo.ipynb` to see a step-by-step example of:
- Loading the model,
- Training on a small subset,
- Sampling from the trained model,
- Visualizing results.

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

Use `ddpm.reverse_process.sample` (or the CLI, if you’ve configured your scripts) to:
1. Initialize random noise `x_T`.
2. Iteratively denoise until an image `x_0` emerges.
3. Save or visualize the result.  

Images are typically clamped to `[0,1]` before saving.

---

## Citation

If you build upon or reference this code, please cite the original DDPM paper:

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
