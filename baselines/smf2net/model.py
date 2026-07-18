#!/usr/bin/env python3
"""
PyTorch **BCHW** SMF2Net-style scale-4 fusion baseline (paper2 adapter).

Lightweight analogue of ``third_party/baselines/SMF2Net`` (TF/Keras, CAVE, 32×): bicubic
upsample LR-HSI to HR-MSI resolution, dual branches, spectral–spatial fusion, residual output
``pred = h_up + head(fusion)``. No TensorFlow dependency.

Typical PaviaU: ``lr_hsi`` [B,103,h,w], ``hr_msi`` [B,3,H,W] with ``H/h == W/w == scale`` (default 4).
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Conv2d → ReLU → Conv2d with residual skip; input/output channels match."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.relu(self.conv1(x)))
        return x + out


class SpectralSpatialFusionBlock(nn.Module):
    """
    SMF2Net-style fusion (lightweight): concat HR-MSI / upsampled HSI branch features,
    mix, then residual conv stack.

    ``f_msi``, ``f_hsi``: [B, hidden_channels, H, W] (same spatial size).
    """

    def __init__(self, hidden_channels: int, num_blocks: int = 4) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=1, bias=True)
        self.mix = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_channels) for _ in range(num_blocks)])

    def forward(self, f_msi: torch.Tensor, f_hsi: torch.Tensor) -> torch.Tensor:
        x = torch.cat([f_msi, f_hsi], dim=1)
        x = self.reduce(x)
        x = self.relu(self.mix(x))
        x = self.blocks(x)
        return x


class SMF2NetScale4(nn.Module):
    """
    BCHW fusion model. ``forward`` upsamples ``lr_hsi`` to ``hr_msi`` spatial size (bicubic).

    ``pred = h_up + reconstruction_head(fusion(MSI_branch, HSI_branch))``.
    """

    def __init__(
        self,
        hsi_channels: int = 103,
        msi_channels: int = 3,
        hidden_channels: int = 64,
        num_blocks: int = 4,
        scale: int = 4,
    ) -> None:
        super().__init__()
        self.hsi_channels = int(hsi_channels)
        self.msi_channels = int(msi_channels)
        self.hidden_channels = int(hidden_channels)
        self.num_blocks = int(num_blocks)
        self.scale = int(scale)

        self.hsi_stem = nn.Conv2d(self.hsi_channels, self.hidden_channels, kernel_size=3, padding=1, bias=True)
        self.hsi_trunk = nn.Sequential(*[ResidualBlock(self.hidden_channels) for _ in range(self.num_blocks)])

        self.msi_stem = nn.Conv2d(self.msi_channels, self.hidden_channels, kernel_size=3, padding=1, bias=True)
        self.msi_trunk = nn.Sequential(*[ResidualBlock(self.hidden_channels) for _ in range(self.num_blocks)])

        self.fusion = SpectralSpatialFusionBlock(self.hidden_channels, num_blocks=self.num_blocks)
        self.recon_head = nn.Conv2d(self.hidden_channels, self.hsi_channels, kernel_size=1, bias=True)

    def forward(self, lr_hsi: torch.Tensor, hr_msi: torch.Tensor) -> torch.Tensor:
        if lr_hsi.dim() != 4 or hr_msi.dim() != 4:
            raise ValueError(f"expected BCHW tensors, got lr_hsi={lr_hsi.shape} hr_msi={hr_msi.shape}")
        _, _, H_hr, W_hr = hr_msi.shape
        h_up = F.interpolate(
            lr_hsi,
            size=(H_hr, W_hr),
            mode="bicubic",
            align_corners=False,
        )
        f_hsi = self.hsi_trunk(self.hsi_stem(h_up))
        f_msi = self.msi_trunk(self.msi_stem(hr_msi))
        fused = self.fusion(f_msi, f_hsi)
        residual = self.recon_head(fused)
        return h_up + residual


def smoke_test() -> Tuple[torch.Tensor, torch.Size]:
    """Sanity: B=2, LR 32×32, HR 128×128 → pred [2,103,128,128]."""
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = SMF2NetScale4(
        hsi_channels=103,
        msi_channels=3,
        hidden_channels=64,
        num_blocks=4,
        scale=4,
    ).to(device)
    lr = torch.randn(2, 103, 32, 32, device=device)
    hr = torch.randn(2, 3, 128, 128, device=device)
    pred = model(lr, hr)
    assert pred.shape == (2, 103, 128, 128), pred.shape
    assert torch.isfinite(pred).all(), "non-finite outputs"
    print(f"[pytorch_smf2net_scale4 smoke] pred shape={tuple(pred.shape)} mean={float(pred.mean()):.6f}")
    return pred, pred.shape


if __name__ == "__main__":
    smoke_test()
