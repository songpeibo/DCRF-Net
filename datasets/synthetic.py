"""Minimal synthetic HSI–MSI fusion pairs for smoke-testing the DCSR-Net pipeline."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from utils.degradation_ops import (
    apply_perturb_kernel,
    apply_perturb_phi,
    make_gaussian_kernel,
    make_simple_srf,
    spatial_degrade,
    spectral_degrade,
)


def _depthwise_blur(x: torch.Tensor, kernel_11kk: torch.Tensor) -> torch.Tensor:
    """Blur each channel of ``x`` ([C, H, W]) with the same spatial ``kernel`` ([1,1,K,K])."""
    if x.ndim != 3:
        raise ValueError(f"x must be [C,H,W], got {tuple(x.shape)}")
    c, _, _ = x.shape
    kk = kernel_11kk.shape[-1]
    pad = kk // 2
    wgt = kernel_11kk[0:1].expand(c, 1, kk, kk).contiguous()
    return F.conv2d(x.unsqueeze(0), wgt, padding=pad, groups=c).squeeze(0)


def _synthesize_z_gt(
    *,
    hsi_channels: int,
    patch_size: int,
    num_endmembers: int = 5,
) -> torch.Tensor:
    """Abundance maps (smooth) × spectral signatures + mild high-pass texture; values in [0, 1]."""
    r = num_endmembers
    sh = max(4, patch_size // 12)
    sw = max(4, patch_size // 12)
    low = torch.randn(r, sh, sw, dtype=torch.float32)
    low = F.interpolate(low.unsqueeze(0), size=(patch_size, patch_size), mode="bilinear", align_corners=False).squeeze(0)
    abund = torch.softmax(low, dim=0)
    signatures = torch.rand(r, hsi_channels, dtype=torch.float32)
    z = torch.einsum("rhw,rc->chw", abund, signatures)

    hf_noise = torch.randn(hsi_channels, patch_size, patch_size, dtype=torch.float32)
    ks = min(9, patch_size)
    if ks % 2 == 0:
        ks -= 1
    ks = max(ks, 3)
    blur_k = make_gaussian_kernel(ks, sigma=1.75, dtype=torch.float32)
    blurred = _depthwise_blur(hf_noise, blur_k)
    z = z + 0.12 * (hf_noise - blurred)
    return z.clamp(0.0, 1.0)


class SyntheticHSIMSIFusionDataset(Dataset):
    """CPU-only synthetic semi-blind fusion samples."""

    _OBS_NOISE_STD = 0.002

    def __init__(
        self,
        num_samples: int,
        hsi_channels: int,
        msi_channels: int,
        patch_size: int,
        scale: int,
        kernel_size: int,
        seed: int,
        semi_blind: bool = True,
    ) -> None:
        super().__init__()
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")
        if hsi_channels < 1 or msi_channels < 1:
            raise ValueError(f"hsi_channels and msi_channels must be >= 1, got {hsi_channels}, {msi_channels}")
        if patch_size < 1 or scale < 1:
            raise ValueError(f"patch_size and scale must be >= 1, got patch_size={patch_size}, scale={scale}")
        if patch_size % scale != 0:
            raise ValueError(f"patch_size ({patch_size}) must be divisible by scale ({scale}) for LR-HSI sizing.")
        if kernel_size < 1:
            raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")

        self.num_samples = int(num_samples)
        self.hsi_channels = int(hsi_channels)
        self.msi_channels = int(msi_channels)
        self.patch_size = int(patch_size)
        self.scale = int(scale)
        self.kernel_size = int(kernel_size)
        self.seed = int(seed)
        self.semi_blind = bool(semi_blind)

        self._phi0 = make_simple_srf(self.hsi_channels, self.msi_channels, dtype=torch.float32)
        self._k0 = make_gaussian_kernel(self.kernel_size, sigma=float(scale) / 2.0, dtype=torch.float32)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(idx)
        torch.manual_seed(self.seed + idx)

        z_gt = _synthesize_z_gt(hsi_channels=self.hsi_channels, patch_size=self.patch_size)
        z_gt = z_gt.to(dtype=torch.float32)

        phi0 = self._phi0.clone().to(dtype=torch.float32)
        k0 = self._k0.clone().to(dtype=torch.float32)

        if self.semi_blind:
            phi_gt = apply_perturb_phi(phi0.clone())
            k_gt = apply_perturb_kernel(k0.clone())
        else:
            phi_gt = phi0.clone()
            k_gt = k0.clone()

        z_b = z_gt.unsqueeze(0)
        m_obs = spectral_degrade(z_b, phi_gt)[0]
        h_obs = spatial_degrade(z_b, k_gt, self.scale)[0]

        m_obs = (m_obs + torch.randn_like(m_obs) * self._OBS_NOISE_STD).clamp(0.0, 1.0)
        h_obs = (h_obs + torch.randn_like(h_obs) * self._OBS_NOISE_STD).clamp(0.0, 1.0)

        return {
            "gt": z_gt,
            "h": h_obs,
            "m": m_obs,
            "phi0": phi0,
            "k0": k0,
            "phi_gt": phi_gt.to(dtype=torch.float32),
            "k_gt": k_gt.to(dtype=torch.float32),
        }


if __name__ == "__main__":
    ds = SyntheticHSIMSIFusionDataset(
        num_samples=4,
        hsi_channels=31,
        msi_channels=3,
        patch_size=64,
        scale=4,
        kernel_size=7,
        seed=0,
        semi_blind=True,
    )
    one = ds[0]
    print("single sample:")
    for key, tensor in one.items():
        print(f"  {key}: shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}")

    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    print("batch:")
    for key, tensor in batch.items():
        print(f"  {key}: shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}")
