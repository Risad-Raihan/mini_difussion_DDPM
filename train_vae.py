"""
Train the VAE first stage
Reconstruction loss (MSE) on images
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


# ============================================================
#                     CONFIG
# ============================================================
class Config:
    image_size = 64
    channels = 3
    latent_channels = 4
    batch_size = 32
    num_epochs = 20
    lr = 1e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    data_path = "data/tinyhero"
    save_dir = "samples_vae"
    os.makedirs(save_dir, exist_ok=True)
    
    vae_checkpoint = "vae.pth"

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
#               TRAINING LOOP
# ============================================================
def train():
    model = VAE(in_channels=config.channels, latent_channels=config.latent_channels).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    
    print(f"Training VAE on {config.device}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Latent shape: (batch, {config.latent_channels}, 16, 16)")
    
    for epoch in range(1, config.num_epochs + 1):
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{config.num_epochs}")
        
        for x0 in pbar:
            x0 = x0.to(config.device)
            
            # Forward: encode → decode
            x_recon, z = model(x0)
            
            # Reconstruction loss (MSE)
            loss = F.mse_loss(x_recon, x0)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch} - Avg Loss: {avg_loss:.6f}")
        
        # Visualize reconstruction at end of epoch
        model.eval()
        with torch.no_grad():
            sample_batch = next(iter(loader))[:8].to(config.device)
            recon_batch, _ = model(sample_batch)
            
            # Combine original and reconstructed
            comparison = torch.cat([sample_batch, recon_batch], dim=0)
            comparison = (comparison + 1) / 2  # [-1,1] -> [0,1]
            utils.save_image(comparison, f"{config.save_dir}/epoch_{epoch:03d}.png", nrow=4)
    
    # Save VAE weights
    torch.save(model.state_dict(), config.vae_checkpoint)
    print(f"\nVAE training complete! Saved to {config.vae_checkpoint}")


if __name__ == "__main__":
    train()

