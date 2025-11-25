# Mini Diffusion

Minimal DDPM implementation for sprite generation, upgraded to Latent Diffusion (VAE + DDPM).

## Architecture

The project now uses a **two-stage latent diffusion pipeline**:

1. **VAE Stage**: Encodes 64×64 RGB images to 16×16 latents (4 channels)
2. **LDPM Stage**: Trains DDPM on the latent space instead of pixel space

This is more efficient and follows modern diffusion model architectures (like Stable Diffusion).

## Usage

### Stage 1: Train VAE

First, train the VAE to learn the latent representation:

```bash
python train_vae.py
```

This will:
- Train the encoder/decoder on your sprite dataset
- Save weights to `vae.pth`
- Save reconstruction samples to `samples_vae/`

### Stage 2: Train Latent Diffusion

Once the VAE is trained, train the DDPM on latents:

```bash
python train_ldpm.py
```

This will:
- Load the frozen VAE
- Train DDPM on encoded latents (16×16, 4 channels)
- Save weights to `ldpm_tinyhero.pth`
- Save generation samples to `samples_ldpm/`

### Generate Samples

Generate new sprites using the full pipeline:

```bash
python sample_ldpm.py
```

This will:
- Load both VAE and LDPM models
- Sample latents using DDPM reverse process
- Decode latents to 64×64 images
- Save to `generated_ldpm/sample.png`

## Legacy DDPM (Pixel Space)

The original pixel-space DDPM is still available:

**Train:**
```bash
python train_ddpm.py
```

**Generate:**
```bash
python sample.py
```

## Project Structure

- `vae.py` - VAE encoder/decoder architecture
- `ddpm.py` - DDPM components (UNet, schedules) for latent space
- `train_vae.py` - VAE training script
- `train_ldpm.py` - Latent diffusion training script
- `sample_ldpm.py` - Sampling script with VAE decoder
- `train_ddpm.py` - Original pixel-space DDPM training
- `sample.py` - Original pixel-space sampling

## Requirements

- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- Pillow >= 9.0.0
- tqdm >= 4.65.0

