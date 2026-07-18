"""Image-quality metrics for HSI reconstruction (batch-averaged scalars)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _check_bchw(x: torch.Tensor, name: str) -> None:
    if x.ndim != 4:
        raise ValueError(f"{name} must be a 4D tensor [B,C,H,W], got shape {tuple(x.shape)}")


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0, eps: float = 1e-10) -> float:
    """Peak SNR over all batch, channel, and spatial elements (global MSE)."""
    _check_bchw(pred, "pred")
    _check_bchw(target, "target")
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}")
    mse = F.mse_loss(pred, target, reduction="mean").clamp(min=eps)
    return float((10.0 * torch.log10((max_val**2) / mse)).item())


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Root mean squared error over all elements."""
    _check_bchw(pred, "pred")
    _check_bchw(target, "target")
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}")
    return float(torch.sqrt(F.mse_loss(pred, target, reduction="mean")).item())


def sam(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Mean spectral angle (degrees) over batch and pixels."""
    _check_bchw(pred, "pred")
    _check_bchw(target, "target")
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}")
    b, c, h, w = pred.shape
    p = pred.reshape(b, c, -1).transpose(1, 2).contiguous()
    t = target.reshape(b, c, -1).transpose(1, 2).contiguous()
    dot = (p * t).sum(dim=-1)
    na = p.norm(dim=-1).clamp(min=eps)
    nb = t.norm(dim=-1).clamp(min=eps)
    cos = (dot / (na * nb)).clamp(-1.0 + eps, 1.0 - eps)
    ang = torch.acos(cos).mean() * (180.0 / math.pi)
    return float(ang.item())


def ergas(pred: torch.Tensor, target: torch.Tensor, scale: int, eps: float = 1e-8) -> float:
    """Global relative dimensionless synthesis error (lower is better).

    Uses the common form::

        ERGAS = (100 / scale) * sqrt(mean_c ( (RMSE_c / mu_c)^2 ))

    where ``RMSE_c`` is per-channel RMSE over (batch, height, width) and ``mu_c`` is the mean absolute
    reference level for channel ``c``.
    """
    _check_bchw(pred, "pred")
    _check_bchw(target, "target")
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}")
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")
    err = pred - target
    mse_c = err.pow(2).mean(dim=(0, 2, 3))
    rmse_c = torch.sqrt(mse_c + eps)
    mu_c = target.abs().mean(dim=(0, 2, 3)).clamp(min=eps)
    rel = rmse_c / mu_c
    val = (100.0 / float(scale)) * torch.sqrt((rel**2).mean() + eps)
    return float(val.item())


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, max_val: float = 1.0, eps: float = 1e-8) -> float:
    """Simplified global SSIM: per-channel SSIM with uniform pooling, then mean over channels and batch.

    This is a lightweight stand-in (box filter statistics) suitable for monitoring curves, not for
    reporting publication-grade multi-scale SSIM.
    """
    _check_bchw(pred, "pred")
    _check_bchw(target, "target")
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}")
    if window_size % 2 == 0 or window_size < 3:
        raise ValueError(f"window_size must be an odd int >= 3, got {window_size}")
    pad = window_size // 2
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    def _local_stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = F.avg_pool2d(x, window_size, stride=1, padding=pad)
        mu2 = F.avg_pool2d(x * x, window_size, stride=1, padding=pad)
        var = (mu2 - mu * mu).clamp(min=0.0)
        return mu, var

    b, c, _, _ = pred.shape
    vals: list[torch.Tensor] = []
    for ci in range(c):
        x = pred[:, ci : ci + 1, :, :]
        y = target[:, ci : ci + 1, :, :]
        mu_x, var_x = _local_stats(x)
        mu_y, var_y = _local_stats(y)
        cov_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=pad) - mu_x * mu_y
        num = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
        den = (mu_x.pow(2) + mu_y.pow(2) + c1) * (var_x + var_y + c2).clamp(min=eps)
        vals.append((num / den).mean())
    return float(torch.stack(vals).mean().item())


def evaluate_reconstruction(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    scale: int = 4,
    max_val: float = 1.0,
) -> dict[str, float]:
    """Compute validation metrics on aligned ``[B,C,H,W]`` tensors (same as ``train.py`` ``_evaluate``).

    Returns keys ``PSNR``, ``SSIM``, ``SAM_deg``, ``RMSE``, ``ERGAS`` for CSV / table export.
    """
    _check_bchw(pred, "pred")
    _check_bchw(target, "target")
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}")
    return {
        "PSNR": psnr(pred, target, max_val=max_val),
        "SSIM": ssim(pred, target, max_val=max_val),
        "SAM_deg": sam(pred, target),
        "RMSE": rmse(pred, target),
        "ERGAS": ergas(pred, target, scale=scale),
    }
