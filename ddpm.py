"""
DDPM components for Latent Diffusion
UNet operates on latent space (4 channels, 16x16) instead of pixel space
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
#                  UNET-LITE (for latents)
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
    """
    UNet for latent diffusion
    Input/Output: (batch, latent_channels, 16, 16)
    """
    def __init__(self, latent_channels=4, base_channels=32, time_emb_dim=128):
        super().__init__()

        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )

        # Down path: 16 -> 8 -> 4
        self.down1 = ResBlock(latent_channels, base_channels, time_emb_dim)
        self.down2 = ResBlock(base_channels, base_channels*2, time_emb_dim)
        self.down3 = ResBlock(base_channels*2, base_channels*4, time_emb_dim)

        self.pool = nn.MaxPool2d(2)

        # Up path: 4 -> 8 -> 16
        self.up2 = ResBlock(base_channels*4, base_channels*2, time_emb_dim)
        self.up1 = ResBlock(base_channels*2, base_channels, time_emb_dim)

        # Output same channels as input (latent_channels)
        self.final_conv = nn.Conv2d(base_channels, latent_channels, 1)

    def forward(self, x, t):
        """
        x: (B, latent_channels, 16, 16) - noisy latents
        t: (B,) - timestep indices
        Returns: (B, latent_channels, 16, 16) - predicted noise
        """
        t_emb = self.time_embedding(t)

        # Down
        x1 = self.down1(x, t_emb)  # 16x16
        x2 = self.down2(self.pool(x1), t_emb)  # 8x8
        x3 = self.down3(self.pool(x2), t_emb)  # 4x4

        # Up
        u = F.interpolate(x3, scale_factor=2, mode="nearest")  # 4 -> 8
        u = self.up2(u, t_emb)  # 8x8

        u = F.interpolate(u, scale_factor=2, mode="nearest")  # 8 -> 16
        u = self.up1(u, t_emb)  # 16x16

        return self.final_conv(u)


# ============================================================
#               BETA / ALPHA SCHEDULE
# ============================================================
def get_beta_schedule(timesteps, start=1e-4, end=0.02):
    return torch.linspace(start, end, timesteps)


# ============================================================
#            Q-SAMPLE (FORWARD NOISING)
# ============================================================
def q_sample(x0, t, noise, alpha_bars):
    """
    Forward diffusion: add noise to latents
    x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
    """
    sqrt_ab = torch.sqrt(alpha_bars[t])[:, None, None, None]
    sqrt_mab = torch.sqrt(1 - alpha_bars[t])[:, None, None, None]
    return sqrt_ab * x0 + sqrt_mab * noise

