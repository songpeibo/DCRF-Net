"""Training losses for DCSR-Net."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.degradation_ops import spatial_degrade, spectral_degrade


def _broadcast_phi0(phi0: torch.Tensor, batch: int) -> torch.Tensor:
    if phi0.ndim == 2:
        return phi0.unsqueeze(0).expand(batch, -1, -1).contiguous()
    if phi0.ndim == 3:
        if phi0.shape[0] == 1 and batch > 1:
            return phi0.expand(batch, -1, -1).contiguous()
        if phi0.shape[0] != batch:
            raise ValueError(f"phi0 batch {phi0.shape[0]} must be 1 or B={batch}, got shape {tuple(phi0.shape)}")
        return phi0
    raise ValueError(f"phi0 must be [C_m, C_h] or [B, C_m, C_h], got shape {tuple(phi0.shape)}")


def _broadcast_k0(k0: torch.Tensor, batch: int) -> torch.Tensor:
    x = k0
    if x.ndim == 5 and x.shape[1] == 1 and x.shape[2] == 1:
        x = x.squeeze(2)
    if x.ndim == 2:
        x = x.view(1, 1, x.shape[0], x.shape[1])
    if x.ndim != 4 or x.shape[1] != 1:
        raise ValueError(f"k0 must be [1,1,K,K] or [B,1,K,K], got shape {tuple(k0.shape)}")
    if x.shape[0] == 1 and batch > 1:
        return x.expand(batch, -1, -1, -1).contiguous()
    if x.shape[0] != batch:
        raise ValueError(f"k0 batch {x.shape[0]} must be 1 or B={batch}, got shape {tuple(x.shape)}")
    return x


def _sam_mean_radians(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean spectral angle (radians) over batch and pixels; ``pred``/``target`` are ``[B, C, H, W]``."""
    b, c, h, w = pred.shape
    p = pred.reshape(b, c, -1).transpose(1, 2).contiguous()  # [B, HW, C]
    t = target.reshape(b, c, -1).transpose(1, 2).contiguous()
    dot = (p * t).sum(dim=-1)
    na = p.norm(dim=-1).clamp(min=eps)
    nb = t.norm(dim=-1).clamp(min=eps)
    cos = (dot / (na * nb)).clamp(-1.0 + eps, 1.0 - eps)
    return torch.acos(cos).mean()


def _read_lambdas(cfg: dict[str, Any] | None, defaults: dict[str, float]) -> dict[str, float]:
    if cfg is None:
        return dict(defaults)
    merged: dict[str, Any] = dict(cfg)
    loss_section = cfg.get("loss")
    if isinstance(loss_section, dict):
        merged.update(loss_section)
    out = dict(defaults)
    for k in defaults:
        if k in merged:
            out[k] = float(merged[k])
    return out


class DCSRLoss(nn.Module):
    """Composite loss: reconstruction, observation consistency, SAM, degradation prior, gate sparsity.

    Primary observation terms use ``pred['m_hat']`` / ``pred['h_hat']`` (final fusion) vs ``batch['m']`` /
    ``batch['h']``. Optional coarse auxiliary uses ``pred['m_hat_coarse']`` / ``pred['h_hat_coarse']``
    weighted by ``lambda_obs_coarse`` (default 0.1). Optional coarse reconstruction uses
    ``pred['z_coarse']`` vs ``batch['gt']`` weighted by ``lambda_rec_coarse`` (default 0.0). Falls back
    to degrading ``z_hat`` / ``z_coarse`` when those keys are absent.

    ``loss_sam`` is the **mean spectral angle in radians** (weighted by ``lambda_sam``); logged scalars use
    the same unit. Use :func:`utils.metrics.sam` if you need degrees for evaluation.
    """

    def __init__(
        self,
        scale: int,
        lambda_rec: float = 1.0,
        lambda_rec_coarse: float = 0.0,
        lambda_obs: float = 0.2,
        lambda_obs_coarse: float = 0.1,
        lambda_sam: float = 0.05,
        lambda_deg: float = 0.001,
        lambda_sel: float = 0.0001,
        lambda_feedback_intermediate: float = 0.0,
    ) -> None:
        super().__init__()
        if scale < 1:
            raise ValueError(f"scale must be >= 1, got {scale}")
        self.scale = int(scale)
        self._defaults = {
            "lambda_rec": float(lambda_rec),
            "lambda_rec_coarse": float(lambda_rec_coarse),
            "lambda_obs": float(lambda_obs),
            "lambda_obs_coarse": float(lambda_obs_coarse),
            "lambda_sam": float(lambda_sam),
            "lambda_deg": float(lambda_deg),
            "lambda_sel": float(lambda_sel),
            "lambda_feedback_intermediate": float(lambda_feedback_intermediate),
        }

    def forward(
        self,
        pred: dict[str, torch.Tensor],
        batch: dict[str, Any],
        cfg: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor | float]:
        if "gt" not in batch:
            raise KeyError("batch must contain 'gt' for DCSRLoss")
        if "z_hat" not in pred:
            raise KeyError("pred must contain 'z_hat'")

        lam = _read_lambdas(cfg, self._defaults)

        z_hat = pred["z_hat"]
        gt = batch["gt"]
        if z_hat.shape != gt.shape:
            raise ValueError(f"z_hat shape {tuple(z_hat.shape)} != gt shape {tuple(gt.shape)}")

        loss_rec = F.l1_loss(z_hat, gt)

        loss_rec_coarse = z_hat.new_tensor(0.0)
        if lam["lambda_rec_coarse"] > 0.0 and "z_coarse" in pred:
            z_coarse = pred["z_coarse"]
            if z_coarse.shape != gt.shape:
                raise ValueError(f"z_coarse shape {tuple(z_coarse.shape)} != gt shape {tuple(gt.shape)}")
            loss_rec_coarse = F.l1_loss(z_coarse, gt)

        m = batch["m"]
        h = batch["h"]
        if "m_hat" in pred and "h_hat" in pred:
            m_hat = pred["m_hat"]
            h_hat = pred["h_hat"]
        else:
            phi_t = pred.get("phi_tilde")
            k_t = pred.get("k_tilde")
            if phi_t is None or k_t is None:
                raise KeyError(
                    "pred must contain 'm_hat' and 'h_hat', or 'phi_tilde' and 'k_tilde' for observation loss"
                )
            m_hat = spectral_degrade(z_hat, phi_t)
            h_hat = spatial_degrade(z_hat, k_t, self.scale)
        loss_obs = F.l1_loss(m_hat, m) + F.l1_loss(h_hat, h)

        loss_obs_coarse = z_hat.new_tensor(0.0)
        if lam["lambda_obs_coarse"] > 0.0:
            if "m_hat_coarse" in pred and "h_hat_coarse" in pred:
                loss_obs_coarse = F.l1_loss(pred["m_hat_coarse"], m) + F.l1_loss(pred["h_hat_coarse"], h)
            elif "z_coarse" in pred:
                phi_t = pred.get("phi_tilde")
                k_t = pred.get("k_tilde")
                if phi_t is None or k_t is None:
                    raise KeyError(
                        "pred must contain 'm_hat_coarse'/'h_hat_coarse' or 'z_coarse' with 'phi_tilde'/'k_tilde'"
                    )
                m_hat_coarse = spectral_degrade(pred["z_coarse"], phi_t)
                h_hat_coarse = spatial_degrade(pred["z_coarse"], k_t, self.scale)
                loss_obs_coarse = F.l1_loss(m_hat_coarse, m) + F.l1_loss(h_hat_coarse, h)

        loss_sam = _sam_mean_radians(z_hat, gt, eps=1e-8)

        phi_t = pred.get("phi_tilde")
        k_t = pred.get("k_tilde")

        loss_deg = z_hat.new_tensor(0.0)
        if "phi0" in batch and "k0" in batch and phi_t is not None and k_t is not None:
            b = z_hat.shape[0]
            phi0_b = _broadcast_phi0(batch["phi0"], b).to(dtype=phi_t.dtype, device=phi_t.device)
            k0_b = _broadcast_k0(batch["k0"], b).to(dtype=k_t.dtype, device=k_t.device)
            loss_deg = (phi_t - phi0_b).pow(2).mean() + (k_t - k0_b).pow(2).mean()

        loss_sel = z_hat.new_tensor(0.0)
        if "gates" in pred:
            g = pred["gates"]
            if g.ndim != 4 or g.shape[1] < 4:
                raise ValueError(f"gates must be [B,4,H,W], got {tuple(g.shape)}")
            # Sum of spectral / spatial / mixed gate mass per pixel, averaged over space & batch.
            loss_sel = g[:, 1:].sum(dim=1).mean()

        loss_feedback_intermediate = z_hat.new_tensor(0.0)
        if lam["lambda_feedback_intermediate"] > 0.0 and "z_feedback_intermediate" in pred:
            z_fb = pred["z_feedback_intermediate"]
            if z_fb.shape != gt.shape:
                raise ValueError(
                    f"z_feedback_intermediate shape {tuple(z_fb.shape)} != gt shape {tuple(gt.shape)}"
                )
            loss_feedback_intermediate = F.l1_loss(z_fb, gt)

        total = (
            lam["lambda_rec"] * loss_rec
            + lam["lambda_rec_coarse"] * loss_rec_coarse
            + lam["lambda_obs"] * loss_obs
            + lam["lambda_obs_coarse"] * loss_obs_coarse
            + lam["lambda_sam"] * loss_sam
            + lam["lambda_deg"] * loss_deg
            + lam["lambda_sel"] * loss_sel
            + lam["lambda_feedback_intermediate"] * loss_feedback_intermediate
        )

        def _f(x: torch.Tensor) -> float:
            return float(x.detach().item())

        return {
            "loss": total,
            "loss_rec": _f(loss_rec),
            "loss_rec_coarse": _f(loss_rec_coarse),
            "loss_obs": _f(loss_obs),
            "loss_obs_coarse": _f(loss_obs_coarse),
            "loss_sam": _f(loss_sam),
            "loss_deg": _f(loss_deg),
            "loss_sel": _f(loss_sel),
            "loss_feedback_intermediate": _f(loss_feedback_intermediate),
        }
