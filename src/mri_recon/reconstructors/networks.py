"""Supervised CNNs (ResidualCNN, U-Net + DC, SRCNN + DC) from Week 5 / Week 10."""

from __future__ import annotations

import torch
import torch.nn as nn

from mri_recon.kspace_ops import data_consistency

RESIDUAL_CNN_ID = "residual_cnn"
UNET_ID = "unet"
SRCNN_ID = "srcnn"


class ResidualCNN(nn.Module):
    """Undersampled -> GT mapping (Week 5)."""

    def __init__(self, ch: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ZFResidualCNN_DC(nn.Module):
    """ZF + small residual CNN + data consistency (Experiment Spec Tier 1)."""

    def __init__(self, ch: int = 32) -> None:
        super().__init__()
        self.core = ResidualCNN(ch=ch)

    def forward(self, x: torch.Tensor, y_obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = self.core(x)
        x_dc = data_consistency(out, y_obs, mask)
        return torch.clamp(x_dc, 0.0, 1.0)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """U-Net with data consistency (Week 10)."""

    def __init__(self, ch: int = 32) -> None:
        super().__init__()
        self.enc1 = ConvBlock(1, ch)
        self.enc2 = ConvBlock(ch, ch * 2)
        self.enc3 = ConvBlock(ch * 2, ch * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(ch * 4, ch * 8)
        self.up3 = nn.ConvTranspose2d(ch * 8, ch * 4, 2, stride=2)
        self.dec3 = ConvBlock(ch * 8, ch * 4)
        self.up2 = nn.ConvTranspose2d(ch * 4, ch * 2, 2, stride=2)
        self.dec2 = ConvBlock(ch * 4, ch * 2)
        self.up1 = nn.ConvTranspose2d(ch * 2, ch, 2, stride=2)
        self.dec1 = ConvBlock(ch * 2, ch)
        self.head = nn.Conv2d(ch, 1, 1)

    def forward(self, x: torch.Tensor, y_obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        out = self.head(d1)
        x_refined = x + out
        x_dc = data_consistency(x_refined, y_obs, mask)
        return torch.clamp(x_dc, 0.0, 1.0)


class SRCNN_DC(nn.Module):
    """SRCNN + residual + data consistency (Week 10)."""

    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.layer2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.layer3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, y_obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.layer1(x))
        out = self.relu(self.layer2(out))
        out = self.layer3(out)
        x_refined = x + out
        x_dc = data_consistency(x_refined, y_obs, mask)
        return torch.clamp(x_dc, 0.0, 1.0)
