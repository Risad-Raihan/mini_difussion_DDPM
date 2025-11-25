import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from glob import glob

# ============================================================
#                     CONFIG
# ============================================================
class Config:
    image_size = 64
    channels = 3
    batch_size = 32
    num_epochs = 30
    lr = 2e-4
    timesteps = 1000  # T
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_path = "data/tinyhero"            # <— put your sprites here
    save_dir = "samples_ddpm"
    os.makedirs(save_dir, exist_ok=True)

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
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = TinyHeroDataset(config.data_path, transform)
loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                    num_workers=4, pin_memory=True)


# ============================================================
#               BETA / ALPHA SCHEDULE
# ============================================================
def get_beta_schedule(timesteps, start=1e-4, end=0.02):
    return torch.linspace(start, end, timesteps)


betas = get_beta_schedule(config.timesteps).to(config.device)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)


# ============================================================
#            Q-SAMPLE (FORWARD NOISING)
# ============================================================
def q_sample(x0, t, noise):
    """
    x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
    """
    sqrt_ab = torch.sqrt(alpha_bars[t])[:, None, None, None]
    sqrt_mab = torch.sqrt(1 - alpha_bars[t])[:, None, None, None]
    return sqrt_ab * x0 + sqrt_mab * noise


# ============================================================
#        SINUSOIDAL TIME EMBEDDINGS (like transformers)
# ============================================================
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# ============================================================
#                  UNET-LITE
# ============================================================
def conv_block(in_c, out_c, time_emb_dim):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.GroupNorm(1, out_c),
        nn.SiLU(),
        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.GroupNorm(1, out_c),
        nn.SiLU(),
    )


class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, time_emb_dim):
        super().__init__()
        self.block = conv_block(in_c, out_c, time_emb_dim)
        self.time_mlp = nn.Linear(time_emb_dim, out_c)
        self.residual = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x, t_emb):
        h = self.block[0](x)
        h = self.block[1](h)
        h = self.block[2](h)

        # time embedding
        t = self.time_mlp(t_emb)[:, :, None, None]
        h = h + t

        h = self.block[3](h)
        h = self.block[4](h)
        h = self.block[5](h)

        return h + self.residual(x)


class UNet(nn.Module):
    def __init__(self, img_channels=3, base_channels=32, time_emb_dim=128):
        super().__init__()

        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )

        self.down1 = ResBlock(img_channels, base_channels, time_emb_dim)
        self.down2 = ResBlock(base_channels, base_channels*2, time_emb_dim)
        self.down3 = ResBlock(base_channels*2, base_channels*4, time_emb_dim)

        self.pool = nn.MaxPool2d(2)

        self.up2 = ResBlock(base_channels*4, base_channels*2, time_emb_dim)
        self.up1 = ResBlock(base_channels*2, base_channels, time_emb_dim)

        self.final_conv = nn.Conv2d(base_channels, img_channels, 1)

    def forward(self, x, t):
        t_emb = self.time_embedding(t)

        # Down
        x1 = self.down1(x, t_emb)
        x2 = self.down2(self.pool(x1), t_emb)
        x3 = self.down3(self.pool(x2), t_emb)

        # Up
        u = F.interpolate(x3, scale_factor=2, mode="nearest")
        u = self.up2(u, t_emb)

        u = F.interpolate(u, scale_factor=2, mode="nearest")
        u = self.up1(u, t_emb)

        return self.final_conv(u)


# ============================================================
#               SAMPLING FUNCTION
# ============================================================
@torch.no_grad()
def sample(model, n=16):
    model.eval()
    img = torch.randn(n, config.channels, config.image_size, config.image_size).to(config.device)

    for t in reversed(range(1, config.timesteps)):
        t_batch = torch.full((n,), t, device=config.device, dtype=torch.long)
        noise_pred = model(img, t_batch)

        beta_t = betas[t]
        alpha_t = alphas[t]
        alpha_bar_t = alpha_bars[t]

        if t > 1:
            noise = torch.randn_like(img)
        else:
            noise = torch.zeros_like(img)

        img = (
            1 / torch.sqrt(alpha_t) * (img - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * noise_pred)
            + torch.sqrt(beta_t) * noise
        )

    img = (img.clamp(-1, 1) + 1) / 2
    return img


# ============================================================
#               TRAINING LOOP
# ============================================================
def train():
    model = UNet().to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    for epoch in range(1, config.num_epochs + 1):
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{config.num_epochs}")

        for x0 in pbar:
            x0 = x0.to(config.device)
            bs = x0.size(0)

            t = torch.randint(1, config.timesteps, (bs,), device=config.device).long()
            noise = torch.randn_like(x0)

            x_t = q_sample(x0, t, noise)
            noise_pred = model(x_t, t)

            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"loss": loss.item()})

        # sample at end of each epoch
        imgs = sample(model, n=16)
        utils.save_image(imgs, f"{config.save_dir}/epoch_{epoch:03d}.png", nrow=4)

    torch.save(model.state_dict(), "ddpm_tinyhero.pth")
    print("Training complete!")


if __name__ == "__main__":
    train()
