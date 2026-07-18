"""Lightweight building blocks for DCSR-Net (coarse backbone and re-learning branches)."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_norm(num_channels: int, norm: Literal["gn", "bn"]) -> nn.Module:
    if norm == "bn":
        return nn.BatchNorm2d(num_channels)
    g = min(8, num_channels)
    while g > 1 and num_channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, num_channels)


class ConvBlock(nn.Module):
    """Conv2d -> norm -> GELU.

    Shapes:
        Input:  ``[B, in_ch, H, W]``
        Output: ``[B, out_ch, H, W]`` (same H, W for odd ``kernel_size`` and padding ``k//2``).
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        norm: Literal["gn", "bn"] = "gn",
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False)
        self.norm = _make_norm(out_ch, norm)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_ch, H, W] -> [B, out_ch, H, W]
        return self.act(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    """Two ``ConvBlock``s with a residual skip; 1x1 projection if channel width changes.

    Shapes:
        Input:  ``[B, in_ch, H, W]``
        Output: ``[B, out_ch, H, W]``
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        norm: Literal["gn", "bn"] = "gn",
    ) -> None:
        super().__init__()
        self.conv1 = ConvBlock(in_ch, out_ch, kernel_size=kernel_size, norm=norm)
        self.conv2 = ConvBlock(out_ch, out_ch, kernel_size=kernel_size, norm=norm)
        self.proj = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_ch, H, W]
        y = self.conv2(self.conv1(x))
        return y + self.proj(x)


class SmallUNetBackbone(nn.Module):
    """Compact encoder–decoder with two stride-2 downsamples and matching upsamples.

    Shapes:
        Input:  ``x`` of shape ``[B, C_in, H, W]`` with ``H % 4 == 0`` and ``W % 4 == 0``.
        Output: ``(feature, coarse)``, each ``[B, base_channels, H, W]``.
                ``coarse`` is an early fusion map; ``feature`` is the refined main representation.
    """

    def __init__(self, in_ch: int, base_channels: int, norm: Literal["gn", "bn"] = "gn") -> None:
        super().__init__()
        bc = base_channels
        self.stem = ConvBlock(in_ch, bc, kernel_size=3, norm=norm)

        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(bc, bc, kernel_size=3, norm=norm))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(bc, bc, kernel_size=3, norm=norm))

        self.mid = ConvBlock(bc, bc, kernel_size=3, norm=norm)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(bc * 2, bc, kernel_size=3, norm=norm),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(bc * 2, bc, kernel_size=3, norm=norm),
        )
        self.final_fuse = ConvBlock(bc * 2, bc, kernel_size=3, norm=norm)

        self.coarse_head = nn.Sequential(
            nn.Conv2d(bc, bc, kernel_size=1, bias=False),
            _make_norm(bc, norm),
            nn.GELU(),
        )
        self.refine = ConvBlock(bc, bc, kernel_size=3, norm=norm)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, C_in, H, W]
        b, _, h, w = x.shape
        if h % 4 != 0 or w % 4 != 0:
            raise ValueError(f"H and W must be divisible by 4, got H={h}, W={w}")

        s0 = self.stem(x)  # [B, bc, H, W]
        s1 = self.down1(s0)  # [B, bc, H/2, W/2]
        s2 = self.down2(s1)  # [B, bc, H/4, W/4]

        u0 = self.mid(s2)  # [B, bc, H/4, W/4]
        u1 = self.up1(torch.cat([u0, s2], dim=1))  # [B, bc, H/2, W/2]
        u2 = self.up2(torch.cat([u1, s1], dim=1))  # [B, bc, H, W]
        u2 = self.final_fuse(torch.cat([u2, s0], dim=1))  # [B, bc, H, W]

        coarse = self.coarse_head(u2)  # [B, bc, H, W]
        feature = self.refine(u2)  # [B, bc, H, W]
        return feature, coarse


class SpectralRelearningBranch(nn.Module):
    """Pointwise mixing + depthwise spatial mixing (per-channel 3x3), biased toward spectral re-mixing.

    Shapes:
        Input:  ``feat`` ``[B, channels, H, W]``
        Output: ``delta_z`` ``[B, hsi_channels, H, W]``
    """

    def __init__(
        self,
        channels: int,
        hsi_channels: int,
        mid_channels: int | None = None,
        norm: Literal["gn", "bn"] = "gn",
    ) -> None:
        super().__init__()
        mid = mid_channels if mid_channels is not None else max(channels // 2, 8)
        self.pw1 = nn.Conv2d(channels, mid, kernel_size=1, bias=False)
        self.norm1 = _make_norm(mid, norm)
        self.dw = nn.Conv2d(mid, mid, kernel_size=3, padding=1, groups=mid, bias=False)
        self.norm2 = _make_norm(mid, norm)
        self.pw2 = nn.Conv2d(mid, hsi_channels, kernel_size=1, bias=True)
        self.act = nn.GELU()

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: [B, C, H, W] -> delta_z: [B, C_h, H, W]
        x = self.act(self.norm1(self.pw1(feat)))
        x = self.act(self.norm2(self.dw(x)))
        return self.pw2(x)


class SpatialRelearningBranch(nn.Module):
    """3x3 conv stack + residual block for spatial-detail refinement.

    Shapes:
        Input:  ``feat`` ``[B, channels, H, W]``
        Output: ``delta_z`` ``[B, hsi_channels, H, W]``
    """

    def __init__(
        self,
        channels: int,
        hsi_channels: int,
        norm: Literal["gn", "bn"] = "gn",
    ) -> None:
        super().__init__()
        self.body = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, norm=norm),
            ResidualBlock(channels, channels, kernel_size=3, norm=norm),
        )
        self.head = nn.Conv2d(channels, hsi_channels, kernel_size=1, bias=True)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: [B, C, H, W] -> delta_z: [B, C_h, H, W]
        return self.head(self.body(feat))


FEEDBACK_OBSERVATION_MODE_FULL = "full"
FEEDBACK_OBSERVATION_MODE_ZERO_BP = "zero_backprojection"
FEEDBACK_OBSERVATION_MODE_MSI_ONLY = "msi_only"
FEEDBACK_OBSERVATION_MODE_LRHSI_ONLY = "lrhsi_only"
FEEDBACK_OBSERVATION_MODES = (
    FEEDBACK_OBSERVATION_MODE_FULL,
    FEEDBACK_OBSERVATION_MODE_ZERO_BP,
    FEEDBACK_OBSERVATION_MODE_MSI_ONLY,
    FEEDBACK_OBSERVATION_MODE_LRHSI_ONLY,
)


class DualObservationResidualFeedbackCell(nn.Module):
    """Dual-observation consistency residual feedback (B10-DCRF-2).

    Reprojects MSI / LR-HSI observation residuals onto HR-HSI and applies a small correction.
    Nominal ``phi0`` / ``k0`` are used for reprojection (not calibrated tilde operators).
    """

    def __init__(
        self,
        hsi_channels: int,
        base_channels: int,
        scale: int,
        *,
        norm: Literal["gn", "bn"] = "gn",
        residual_scale: float = 0.1,
        observation_mode: str = FEEDBACK_OBSERVATION_MODE_FULL,
    ) -> None:
        super().__init__()
        if hsi_channels < 1 or base_channels < 8 or scale < 1:
            raise ValueError("hsi_channels >= 1, base_channels >= 8, scale >= 1 required")
        mode = str(observation_mode).strip().lower()
        if mode not in FEEDBACK_OBSERVATION_MODES:
            raise ValueError(
                f"observation_mode must be one of {FEEDBACK_OBSERVATION_MODES}, got {observation_mode!r}"
            )
        self.observation_mode = mode
        self.hsi_channels = int(hsi_channels)
        self.base_channels = int(base_channels)
        self.scale = int(scale)
        self.residual_scale = float(residual_scale)
        bc = base_channels
        c_h = hsi_channels
        self.proj_z = nn.Conv2d(c_h, bc, kernel_size=1, bias=False)
        self.proj_m = nn.Conv2d(c_h, bc // 2, kernel_size=1, bias=False)
        self.proj_h = nn.Conv2d(c_h, bc // 2, kernel_size=1, bias=False)
        self.proj_abs_m = nn.Conv2d(c_h, max(bc // 4, 4), kernel_size=1, bias=False)
        self.proj_abs_h = nn.Conv2d(c_h, max(bc // 4, 4), kernel_size=1, bias=False)
        fuse_in = bc + bc // 2 + bc // 2 + max(bc // 4, 4) + max(bc // 4, 4)
        self.fuse = ConvBlock(fuse_in, bc, kernel_size=3, norm=norm)
        self.body = nn.Sequential(
            ResidualBlock(bc, bc, kernel_size=3, norm=norm),
            ResidualBlock(bc, bc, kernel_size=3, norm=norm),
        )
        self.head = nn.Conv2d(bc, c_h, kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        z_t: torch.Tensor,
        m: torch.Tensor,
        h: torch.Tensor,
        phi0: torch.Tensor,
        k0: torch.Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        from utils.degradation_ops import spectral_backproject, spatial_degrade, spectral_degrade

        b, _, H, W = z_t.shape
        m_hat_t = spectral_degrade(z_t, phi0)
        h_hat_t = spatial_degrade(z_t, k0, self.scale)
        e_m = m - m_hat_t
        e_h = h - h_hat_t
        bp_m = spectral_backproject(e_m, phi0)
        bp_h = F.interpolate(e_h, size=(H, W), mode="bicubic", align_corners=False)
        if self.observation_mode == FEEDBACK_OBSERVATION_MODE_ZERO_BP:
            bp_m = torch.zeros_like(bp_m)
            bp_h = torch.zeros_like(bp_h)
        elif self.observation_mode == FEEDBACK_OBSERVATION_MODE_MSI_ONLY:
            bp_h = torch.zeros_like(bp_h)
        elif self.observation_mode == FEEDBACK_OBSERVATION_MODE_LRHSI_ONLY:
            bp_m = torch.zeros_like(bp_m)

        fz = self.proj_z(z_t)
        fm = self.proj_m(bp_m)
        fh = self.proj_h(bp_h)
        fam = self.proj_abs_m(bp_m.abs())
        fah = self.proj_abs_h(bp_h.abs())
        feat = self.fuse(torch.cat([fz, fm, fh, fam, fah], dim=1))
        feat = self.body(feat)
        delta_z = self.residual_scale * torch.tanh(self.head(feat))
        z_next = (z_t + delta_z).clamp(0.0, 1.0)

        diag: dict[str, torch.Tensor] = {}
        if return_diagnostics:
            diag = {
                "m_hat_t": m_hat_t,
                "h_hat_t": h_hat_t,
                "e_m": e_m,
                "e_h": e_h,
                "bp_m": bp_m,
                "bp_h": bp_h,
                "delta_z": delta_z,
                "e_m_l1_mean": e_m.abs().mean().detach().expand(b),
                "e_h_l1_mean": e_h.abs().mean().detach().expand(b),
                "bp_m_l1_mean": bp_m.abs().mean().detach().expand(b),
                "bp_h_l1_mean": bp_h.abs().mean().detach().expand(b),
                "delta_z_l1_mean": delta_z.abs().mean().detach().expand(b),
            }
        return z_next, diag


class MixedRelearningBranch(nn.Module):
    """A few residual blocks mixing channels and spatial context (moderate capacity).

    Shapes:
        Input:  ``feat`` ``[B, channels, H, W]``
        Output: ``delta_z`` ``[B, hsi_channels, H, W]``
    """

    def __init__(
        self,
        channels: int,
        hsi_channels: int,
        num_blocks: int = 3,
        norm: Literal["gn", "bn"] = "gn",
    ) -> None:
        super().__init__()
        if num_blocks < 1:
            raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")
        blocks: list[nn.Module] = []
        for _ in range(num_blocks):
            blocks.append(ResidualBlock(channels, channels, kernel_size=3, norm=norm))
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Conv2d(channels, hsi_channels, kernel_size=1, bias=True)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: [B, C, H, W] -> delta_z: [B, C_h, H, W]
        return self.head(self.blocks(feat))
