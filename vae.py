"""
Mini-VAE for Latent Diffusion
Encoder → Latent → Decoder
Output latents: (batch, 4, 16, 16) for 64x64 input images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Encodes 64x64 RGB images to 16x16 latent space"""
    def __init__(self, in_channels=3, latent_channels=4):
        super().__init__()
        # Downsample: 64 -> 32 -> 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),  # 64 -> 32
            nn.GroupNorm(4, 32),
            nn.SiLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 32 -> 16
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),  # 16 -> 16
            nn.GroupNorm(16, 128),
            nn.SiLU(),
        )
        self.final = nn.Conv2d(128, latent_channels, 3, padding=1)  # 16 -> 16
        
    def forward(self, x):
        # x: (B, 3, 64, 64)
        h = self.conv1(x)  # (B, 32, 32, 32)
        h = self.conv2(h)  # (B, 64, 16, 16)
        h = self.conv3(h)  # (B, 128, 16, 16)
        z = self.final(h)  # (B, 4, 16, 16)
        return z


class Decoder(nn.Module):
    """Decodes 16x16 latents back to 64x64 RGB images"""
    def __init__(self, latent_channels=4, out_channels=3):
        super().__init__()
        # Upsample: 16 -> 32 -> 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(latent_channels, 128, 3, padding=1),  # 16 -> 16
            nn.GroupNorm(16, 128),
            nn.SiLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),  # 16 -> 16
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 16 -> 32
            nn.GroupNorm(4, 32),
            nn.SiLU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(32, out_channels, 4, stride=2, padding=1),  # 32 -> 64
            nn.Tanh(),  # Output in [-1, 1]
        )
        
    def forward(self, z):
        # z: (B, 4, 16, 16)
        h = self.conv1(z)  # (B, 128, 16, 16)
        h = self.conv2(h)  # (B, 64, 16, 16)
        h = self.up1(h)  # (B, 32, 32, 32)
        x_recon = self.up2(h)  # (B, 3, 64, 64)
        return x_recon


class VAE(nn.Module):
    """Simple VAE: Encoder → Latent → Decoder"""
    def __init__(self, in_channels=3, latent_channels=4):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_channels)
        self.decoder = Decoder(latent_channels, in_channels)
        
    def encode(self, x):
        """Encode image to latent"""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode latent to image"""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass: encode → decode"""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

