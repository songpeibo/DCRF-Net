"""Spectral and spatial degradation operators for semi-blind HSI–MSI fusion."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def normalize_srf(phi: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Non-negative SRF; each MSI band (row) sums to 1 over HSI bands (last dim)."""
    if phi.ndim not in (2, 3):
        raise ValueError(f"phi must be 2D or 3D, got shape {tuple(phi.shape)}")
    x = phi.relu() + eps
    return x / x.sum(dim=-1, keepdim=True)


def normalize_kernel(k: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Non-negative PSF; spatial entries sum to 1 (per batch if batched)."""
    if k.ndim == 2:
        if k.shape[0] != k.shape[1]:
            raise ValueError(f"kernel must be square [K,K], got {tuple(k.shape)}")
        x = k.relu() + eps
        return x / x.sum()
    if k.ndim == 4:
        kh, kw = k.shape[-2], k.shape[-1]
        if kh != kw:
            raise ValueError(f"kernel spatial dims must be square, got {tuple(k.shape)}")
        x = k.relu() + eps
        return x / x.sum(dim=(-2, -1), keepdim=True)
    raise ValueError(f"k must be [K,K] or 4D [1,1,K,K] / [B,1,K,K], got shape {tuple(k.shape)}")


def spectral_degrade(z: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Spectral downsampling: HR-HSI → synthetic MSI (same spatial resolution)."""
    if z.ndim != 4:
        raise ValueError(f"z must be [B,C_h,H,W], got shape {tuple(z.shape)}")
    b, c_h, _, _ = z.shape
    if phi.ndim == 2:
        c_m, c_h_phi = phi.shape
        if c_h_phi != c_h:
            raise ValueError(f"z channels {c_h} != phi dim (-1) {c_h_phi}")
        return torch.einsum("bchw,mc->bmhw", z, phi)
    if phi.ndim == 3:
        if phi.shape[0] != b:
            raise ValueError(f"z batch {b} != phi batch {phi.shape[0]}")
        c_m, c_h_phi = phi.shape[1], phi.shape[2]
        if c_h_phi != c_h:
            raise ValueError(f"z channels {c_h} != phi dim (-1) {c_h_phi}")
        return torch.einsum("bchw,bmc->bmhw", z, phi)
    raise ValueError(f"phi must be [C_m,C_h] or [B,C_m,C_h], got shape {tuple(phi.shape)}")


def spatial_degrade(z: torch.Tensor, k: torch.Tensor, scale: int) -> torch.Tensor:
    """Depthwise blur then sub-sampling by ``scale`` (stride along H, W)."""
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")
    if z.ndim != 4:
        raise ValueError(f"z must be [B,C_h,H,W], got shape {tuple(z.shape)}")
    b, c_h, h, w = z.shape
    if h % scale != 0 or w % scale != 0:
        raise ValueError(
            f"H and W must be divisible by scale (got H={h}, W={w}, scale={scale}) "
            "so that [::scale] matches H//scale, W//scale."
        )

    k_t = _format_depthwise_kernel(k, b=b, c_h=c_h)
    kk = k_t.shape[-1]
    padding = kk // 2

    if k_t.shape[0] == 1:
        wgt = k_t[0:1].expand(c_h, 1, kk, kk).contiguous()
        conv = F.conv2d(z, wgt, padding=padding, groups=c_h)
        out = conv[:, :, ::scale, ::scale]
    else:
        outs: list[torch.Tensor] = []
        for bi in range(b):
            wgt = k_t[bi : bi + 1].expand(c_h, 1, kk, kk).contiguous()
            conv = F.conv2d(z[bi : bi + 1], wgt, padding=padding, groups=c_h)
            outs.append(conv[:, :, ::scale, ::scale])
        out = torch.cat(outs, dim=0)
    assert out.shape == (b, c_h, h // scale, w // scale), (out.shape, b, c_h, h, w, scale)
    return out


def _format_depthwise_kernel(k: torch.Tensor, *, b: int, c_h: int) -> torch.Tensor:
    """Return kernel tensor of shape [B or 1, 1, K, K] for batch-wise or shared PSF."""
    if k.ndim == 2:
        kh, kw = k.shape
        if kh != kw:
            raise ValueError(f"kernel must be square [K,K], got {tuple(k.shape)}")
        return k.view(1, 1, kh, kw)
    if k.ndim == 4:
        if k.shape[1] != 1:
            raise ValueError(f"kernel 4D must have singleton channel dim [*,1,K,K], got {tuple(k.shape)}")
        kh, kw = k.shape[-2], k.shape[-1]
        if kh != kw:
            raise ValueError(f"kernel spatial dims must be square, got {tuple(k.shape)}")
        if k.shape[0] == 1:
            return k
        if k.shape[0] != b:
            raise ValueError(f"batch kernel B={k.shape[0]} != z batch B={b}")
        return k
    raise ValueError(f"k must be [K,K] or [1,1,K,K] or [B,1,K,K], got shape {tuple(k.shape)}")


def make_gaussian_kernel(
    kernel_size: int,
    sigma: float,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """2D isotropic Gaussian PSF, shape [1,1,K,K] with unit mass."""
    if kernel_size < 1:
        raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    ax = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) * 0.5
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    g = torch.exp(-(xx.pow(2) + yy.pow(2)) / (2.0 * sigma**2))
    g = g / (g.sum() + 1e-8)
    return g.view(1, 1, kernel_size, kernel_size)


def make_simple_srf(
    hsi_channels: int,
    msi_channels: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Nominal Phi0 [C_m, C_h]: Gaussian rows covering the HSI-band axis."""
    if hsi_channels < 1 or msi_channels < 1:
        raise ValueError(f"hsi_channels and msi_channels must be >= 1, got {hsi_channels}, {msi_channels}")
    band = torch.arange(hsi_channels, device=device, dtype=dtype).view(1, hsi_channels)
    centers = torch.linspace(0.0, float(hsi_channels - 1), msi_channels, device=device, dtype=dtype).view(
        msi_channels, 1
    )
    sigma = max(float(hsi_channels) / max(msi_channels * 2, 1), 0.5)
    phi = torch.exp(-0.5 * ((band - centers) / sigma) ** 2)
    return normalize_srf(phi)


def apply_perturb_phi(phi0: torch.Tensor, max_shift: int = 2, noise_std: float = 0.01) -> torch.Tensor:
    """Random band-axis roll + Gaussian noise; renormalize rows."""
    if max_shift < 0:
        raise ValueError(f"max_shift must be >= 0, got {max_shift}")
    if phi0.ndim not in (2, 3):
        raise ValueError(f"phi0 must be 2D or 3D, got shape {tuple(phi0.shape)}")
    lo, hi = int(-max_shift), int(max_shift)
    shift = int(torch.randint(lo, hi + 1, (1,), device=phi0.device).item())
    phi = torch.roll(phi0, shifts=shift, dims=-1)
    noise = torch.randn_like(phi) * noise_std
    return normalize_srf(phi + noise)


def apply_perturb_kernel(k0: torch.Tensor, noise_std: float = 0.02) -> torch.Tensor:
    """Additive noise then spatial mass normalization."""
    noise = torch.randn_like(k0) * noise_std
    return normalize_kernel(k0 + noise)


def spectral_adjoint(e_m: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Adjoint of ``spectral_degrade``: MSI residual -> HR-HSI correction map.

    For ``M = Phi @ Z`` (per pixel), the adjoint maps ``E_m`` with ``G = Phi^T E_m``.
    """
    if e_m.ndim != 4:
        raise ValueError(f"e_m must be [B,C_m,H,W], got {tuple(e_m.shape)}")
    b, c_m, _, _ = e_m.shape
    if phi.ndim == 2:
        if phi.shape[0] != c_m:
            raise ValueError(f"phi rows {phi.shape[0]} != e_m channels {c_m}")
        return torch.einsum("mc,bmhw->bchw", phi, e_m)
    if phi.ndim == 3:
        if phi.shape[0] != b or phi.shape[1] != c_m:
            raise ValueError(f"batched phi shape {tuple(phi.shape)} incompatible with e_m batch {b}")
        return torch.einsum("bmc,bmhw->bchw", phi, e_m)
    raise ValueError(f"phi must be [C_m,C_h] or [B,C_m,C_h], got {tuple(phi.shape)}")


def spectral_backproject(e_m: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """MSI observation residual back-projection to HSI space (alias of :func:`spectral_adjoint`)."""
    return spectral_adjoint(e_m, phi)


def spatial_degrade_adjoint(
    r_h: torch.Tensor,
    k: torch.Tensor,
    target_size: tuple[int, int],
) -> torch.Tensor:
    """Approximate adjoint of blur + stride subsample: upsample LR map then transpose blur.

    Uses bilinear upsample (matching export_support) and flipped-kernel depthwise conv.
    Documented for B10 posterior back-projection only.
    """
    if r_h.ndim != 4:
        raise ValueError(f"r_h must be [B,C,h,w], got {tuple(r_h.shape)}")
    b, c, _, _ = r_h.shape
    h_hr, w_hr = int(target_size[0]), int(target_size[1])
    if h_hr < 1 or w_hr < 1:
        raise ValueError(f"target_size must be positive, got {target_size}")

    r_up = F.interpolate(r_h, size=(h_hr, w_hr), mode="bilinear", align_corners=False)
    k_n = normalize_kernel(k.to(device=r_h.device, dtype=r_h.dtype))
    if k_n.ndim == 2:
        kh, kw = k_n.shape
        w_dw = k_n.view(1, 1, kh, kw).expand(c, 1, kh, kw)
        batched_k = False
    elif k_n.ndim == 4:
        kh, kw = k_n.shape[-2], k_n.shape[-1]
        if k_n.shape[0] == 1:
            w_dw = k_n.expand(c, 1, kh, kw)
            batched_k = False
        elif k_n.shape[0] == b:
            w_dw = k_n
            batched_k = True
        else:
            raise ValueError(f"kernel batch {k_n.shape[0]} != r_h batch {b}")
    else:
        raise ValueError(f"k must be [K,K] or [1,1,K,K] or [B,1,K,K], got {tuple(k.shape)}")

    k_adj = torch.flip(w_dw, dims=(-2, -1))
    pad_h, pad_w = kh // 2, kw // 2
    if not batched_k:
        return F.conv2d(r_up, k_adj, bias=None, stride=1, padding=(pad_h, pad_w), groups=c)
    parts = []
    for bi in range(b):
        wb = k_adj[bi : bi + 1].expand(c, 1, kh, kw)
        parts.append(
            F.conv2d(r_up[bi : bi + 1], wb, bias=None, stride=1, padding=(pad_h, pad_w), groups=c)
        )
    return torch.cat(parts, dim=0)
