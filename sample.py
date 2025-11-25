import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils

# ============================================================
#                   CONFIG
# ============================================================
class Config:
    image_size = 64
    channels = 3
    timesteps = 1000
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = "ddpm_tinyhero.pth"
    save_dir = "generated"
    os.makedirs(save_dir, exist_ok=True)

config = Config()


# ============================================================
#               BETA / ALPHA SCHEDULE
# ============================================================
def get_beta_schedule(timesteps, start=1e-4, end=0.02):
    return torch.linspace(start, end, timesteps)

betas = get_beta_schedule(config.timesteps).to(config.device)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)


# ============================================================
#        SINUSOIDAL TIME EMBEDDINGS
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
#                     UNET-LITE
# ============================================================
def conv_block(in_c, out_c):
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
        self.block = conv_block(in_c, out_c)
        self.time_mlp = nn.Linear(time_emb_dim, out_c)
        self.residual = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x, t_emb):
        h = self.block(x)
        t = self.time_mlp(t_emb)[:, :, None, None]
        return h + t + self.residual(x)

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

        x1 = self.down1(x, t_emb)
        x2 = self.down2(self.pool(x1), t_emb)
        x3 = self.down3(self.pool(x2), t_emb)

        u = F.interpolate(x3, scale_factor=2, mode="nearest")
        u = self.up2(u, t_emb)

        u = F.interpolate(u, scale_factor=2, mode="nearest")
        u = self.up1(u, t_emb)

        return self.final_conv(u)


# ============================================================
#                     SAMPLING
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
            1 / torch.sqrt(alpha_t)
            * (img - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * noise_pred)
            + torch.sqrt(beta_t) * noise
        )

    img = (img.clamp(-1, 1) + 1) / 2
    return img


# ============================================================
#                   MAIN SAMPLING
# ============================================================
def main():
    model = UNet().to(config.device)
    model.load_state_dict(torch.load(config.model_path, map_location=config.device))

    imgs = sample(model, n=16)
    utils.save_image(imgs, f"{config.save_dir}/sample.png", nrow=4)

    print("Generated sample saved to generated/sample.png")


if __name__ == "__main__":
    main()
