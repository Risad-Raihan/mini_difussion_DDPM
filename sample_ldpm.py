"""
Sample from Latent Diffusion Model
Loads VAE + LDPM, samples latents, decodes to 64x64 images
"""

import os
import torch
from torchvision import utils
from vae import VAE
from ddpm import UNet, get_beta_schedule


# ============================================================
#                   CONFIG
# ============================================================
class Config:
    image_size = 64
    channels = 3
    latent_channels = 4
    timesteps = 1000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    vae_checkpoint = "vae.pth"
    ldpm_checkpoint = "ldpm_tinyhero.pth"
    save_dir = "generated_ldpm"
    os.makedirs(save_dir, exist_ok=True)

config = Config()


# ============================================================
#               BETA / ALPHA SCHEDULE
# ============================================================
betas = get_beta_schedule(config.timesteps).to(config.device)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)


# ============================================================
#               SAMPLING FUNCTION (FIXED)
# ============================================================
@torch.no_grad()
def sample(model, vae, n=16):
    """
    Correct DDPM reverse process:
    1. Start with noise in latent space
    2. Iteratively denoise using the model
    3. Decode final latents to images
    
    Uses standard DDPM sampling formula:
    z_{t-1} = 1/sqrt(alpha_t) * (z_t - beta_t/sqrt(1-alpha_bar_t) * epsilon_theta) + sqrt(beta_t) * noise
    """
    model.eval()
    vae.eval()
    
    # Start with random noise in latent space: (B, 4, 16, 16)
    z = torch.randn(n, config.latent_channels, 16, 16).to(config.device)
    
    # Reverse diffusion loop: T -> 1
    # betas[0] = beta_1, betas[T-1] = beta_T
    for t in reversed(range(config.timesteps)):
        t_idx = t  # Index into betas/alphas arrays (0-indexed)
        t_batch = torch.full((n,), t_idx, device=config.device, dtype=torch.long)
        noise_pred = model(z, t_batch)
        
        beta_t = betas[t_idx]
        alpha_t = alphas[t_idx]
        alpha_bar_t = alpha_bars[t_idx]
        
        # Standard DDPM reverse step
        # z_{t-1} = 1/sqrt(alpha_t) * (z_t - beta_t/sqrt(1-alpha_bar_t) * epsilon_theta)
        pred_z = (1.0 / torch.sqrt(alpha_t)) * (z - (beta_t / torch.sqrt(1 - alpha_bar_t)) * noise_pred)
        
        # Add noise if not at final step
        if t > 0:
            noise = torch.randn_like(z)
            z = pred_z + torch.sqrt(beta_t) * noise
        else:
            # At t=0 (final step), no noise
            z = pred_z
    
    # Clamp latents to reasonable range
    z = z.clamp(-1, 1)
    
    # Decode latents to images
    imgs = vae.decode(z)
    
    # Normalize to [0, 1] for saving
    imgs = (imgs.clamp(-1, 1) + 1) / 2
    
    return imgs


# ============================================================
#                   MAIN SAMPLING
# ============================================================
def main():
    print("Loading VAE...")
    vae = VAE(in_channels=config.channels, latent_channels=config.latent_channels).to(config.device)
    vae.load_state_dict(torch.load(config.vae_checkpoint, map_location=config.device))
    vae.eval()
    print("VAE loaded!")
    
    print("Loading LDPM...")
    model = UNet(latent_channels=config.latent_channels).to(config.device)
    model.load_state_dict(torch.load(config.ldpm_checkpoint, map_location=config.device))
    model.eval()
    print("LDPM loaded!")
    
    print("Sampling...")
    imgs = sample(model, vae, n=16)
    utils.save_image(imgs, f"{config.save_dir}/sample.png", nrow=4)
    
    print(f"Generated sample saved to {config.save_dir}/sample.png")


if __name__ == "__main__":
    main()

