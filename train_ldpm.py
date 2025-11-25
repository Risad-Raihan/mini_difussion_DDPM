"""
Train Latent Diffusion Model (LDPM)
Stage 2: Freeze VAE, train DDPM on encoded latents
"""

import os
import torch
import torch.nn.functional as F
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from glob import glob
from vae import VAE
from ddpm import UNet, get_beta_schedule, q_sample


# ============================================================
#                     CONFIG
# ============================================================
class Config:
    image_size = 64
    channels = 3
    latent_channels = 4
    batch_size = 32
    num_epochs = 30
    lr = 2e-4
    timesteps = 1000  # T
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    data_path = "data/tinyhero"
    save_dir = "samples_ldpm"
    os.makedirs(save_dir, exist_ok=True)
    
    vae_checkpoint = "vae.pth"
    ldpm_checkpoint = "ldpm_tinyhero.pth"

config = Config()


# ============================================================
#                    DATASET
# ============================================================
class TinyHeroDataset(Dataset):
    def __init__(self, root, transform=None):
        self.paths = sorted(glob(os.path.join(root, "*/*.png")))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


transform = transforms.Compose([
    transforms.Resize((config.image_size, config.image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)  # [-1, 1]
])

dataset = TinyHeroDataset(config.data_path, transform)
loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                    num_workers=4, pin_memory=True)


# ============================================================
#               BETA / ALPHA SCHEDULE
# ============================================================
betas = get_beta_schedule(config.timesteps).to(config.device)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)


# ============================================================
#               TRAINING LOOP
# ============================================================
def train():
    # Load frozen VAE
    print("Loading VAE...")
    vae = VAE(in_channels=config.channels, latent_channels=config.latent_channels).to(config.device)
    vae.load_state_dict(torch.load(config.vae_checkpoint, map_location=config.device))
    vae.eval()  # Freeze VAE
    for param in vae.parameters():
        param.requires_grad = False
    print("VAE loaded and frozen!")
    
    # Initialize DDPM model
    model = UNet(latent_channels=config.latent_channels).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    
    print(f"Training LDPM on {config.device}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Training DDPM on latents: (batch, {config.latent_channels}, 16, 16)")
    
    for epoch in range(1, config.num_epochs + 1):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{config.num_epochs}")
        
        for x0 in pbar:
            x0 = x0.to(config.device)
            bs = x0.size(0)
            
            # Encode images to latents
            with torch.no_grad():
                z0 = vae.encode(x0)  # (B, 4, 16, 16)
            
            # Sample timestep and noise
            t = torch.randint(1, config.timesteps, (bs,), device=config.device).long()
            noise = torch.randn_like(z0)
            
            # Forward diffusion on latents
            z_t = q_sample(z0, t, noise, alpha_bars)
            
            # Predict noise
            noise_pred = model(z_t, t)
            
            # Loss
            loss = F.mse_loss(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({"loss": loss.item()})
        
        # Sample at end of each epoch
        if epoch % 5 == 0 or epoch == config.num_epochs:
            sample_latents(model, vae, n=16, epoch=epoch)
    
    # Save LDPM weights
    torch.save(model.state_dict(), config.ldpm_checkpoint)
    print(f"\nLDPM training complete! Saved to {config.ldpm_checkpoint}")


# ============================================================
#               SAMPLING FUNCTION (for visualization)
# ============================================================
@torch.no_grad()
def sample_latents(model, vae, n=16, epoch=0):
    """Sample latents, decode to images using correct DDPM reverse process"""
    model.eval()
    vae.eval()
    
    # Sample noise in latent space
    z = torch.randn(n, config.latent_channels, 16, 16).to(config.device)
    
    # DDPM reverse process: T -> 1 (0-indexed: timesteps-1 -> 0)
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
    
    # Clamp latents
    z = z.clamp(-1, 1)
    
    # Decode latents to images
    imgs = vae.decode(z)
    imgs = (imgs.clamp(-1, 1) + 1) / 2  # [-1,1] -> [0,1]
    
    utils.save_image(imgs, f"{config.save_dir}/epoch_{epoch:03d}.png", nrow=4)
    print(f"Saved sample to {config.save_dir}/epoch_{epoch:03d}.png")


if __name__ == "__main__":
    train()

