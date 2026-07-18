"""DCSR-Net: coarse fusion, semi-blind calibration, consistency gates, selective re-learning."""

from __future__ import annotations

from typing import Any, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.degradation_ops import normalize_kernel, normalize_srf, spatial_degrade, spectral_degrade
from .blocks import (
    ConvBlock,
    DualObservationResidualFeedbackCell,
    MixedRelearningBranch,
    SmallUNetBackbone,
    SpatialRelearningBranch,
    SpectralRelearningBranch,
)


def _broadcast_phi0(phi0: torch.Tensor, batch: int) -> torch.Tensor:
    """``[C_m, C_h]`` -> ``[B, C_m, C_h]``; leave ``[B, C_m, C_h]`` batched (with broadcast from 1)."""
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
    """``[1,1,K,K]`` -> ``[B,1,K,K]``; accept per-batch ``[B,1,K,K]``."""
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


class _GuidanceProj(nn.Module):
    """Conv + GELU projection for signed / detail residuals."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        kernel_size: int = 3,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _BranchFuse(nn.Module):
    """1x1 fuse: concat(feat, guidance...) -> base_channels."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DCSRNet(nn.Module):
    """Semi-blind HSI–MSI fusion with degradation-aware selective refinement.

    Forward expects a ``batch`` dict with keys ``h``, ``m``, ``phi0``, ``k0``:

    - ``h``: LR-HSI ``[B, C_h, h, w]``
    - ``m``: HR-MSI ``[B, C_m, H, W]``
    - ``phi0``: nominal SRF ``[C_m, C_h]`` or ``[B, C_m, C_h]`` (or broadcast-from-1 batch)
    - ``k0``: nominal PSF ``[1, 1, K, K]`` or ``[B, 1, K, K]``

    Spatial sizes must satisfy ``H = h * scale``, ``W = w * scale``, and ``H``, ``W`` divisible by ``4``
    (U-Net pooling). ``K`` must match ``kernel_size``.
    """

    def __init__(
        self,
        hsi_channels: int,
        msi_channels: int,
        scale: int,
        kernel_size: int,
        base_channels: int,
        delta_phi_scale: float = 0.05,
        delta_k_scale: float = 0.05,
        norm: Literal["gn", "bn"] = "gn",
        *,
        guidance_channels: int = 8,
        use_signed_residual_guidance: bool = False,
        use_msi_detail_guidance: bool = False,
        use_reliability_gated_coarse_fusion: bool = False,
        use_dual_observation_feedback: bool = False,
        num_feedback_steps: int = 2,
        share_feedback_weights: bool = True,
        feedback_residual_scale: float = 0.1,
        feedback_observation_mode: str = "full",
    ) -> None:
        super().__init__()
        if hsi_channels < 1 or msi_channels < 1 or scale < 1 or kernel_size < 1 or base_channels < 1:
            raise ValueError("hsi_channels, msi_channels, scale, kernel_size, base_channels must be >= 1")
        self.hsi_channels = int(hsi_channels)
        self.msi_channels = int(msi_channels)
        self.scale = int(scale)
        self.kernel_size = int(kernel_size)
        self.base_channels = int(base_channels)
        self.delta_phi_scale = float(delta_phi_scale)
        self.delta_k_scale = float(delta_k_scale)
        self.guidance_channels = int(guidance_channels)
        self.use_signed_residual_guidance = bool(use_signed_residual_guidance)
        self.use_msi_detail_guidance = bool(use_msi_detail_guidance)
        self.use_reliability_gated_coarse_fusion = bool(use_reliability_gated_coarse_fusion)
        self.use_dual_observation_feedback = bool(use_dual_observation_feedback)
        self.num_feedback_steps = max(0, int(num_feedback_steps))
        self.share_feedback_weights = bool(share_feedback_weights)
        self.feedback_residual_scale = float(feedback_residual_scale)
        self.feedback_observation_mode = str(feedback_observation_mode).strip().lower()
        self.use_guidance = self.use_signed_residual_guidance or self.use_msi_detail_guidance

        coarse_extra = 1 if self.use_reliability_gated_coarse_fusion else 0
        in_ch = self.hsi_channels + self.msi_channels + coarse_extra
        self.backbone = SmallUNetBackbone(in_ch, self.base_channels, norm=norm)
        self.z_coarse_head = nn.Conv2d(self.base_channels, self.hsi_channels, kernel_size=1, bias=True)

        hidden = max(self.base_channels // 2, 128)
        flat_phi = self.msi_channels * self.hsi_channels
        self.phi_mlp = nn.Sequential(
            nn.Linear(self.base_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, flat_phi),
        )
        flat_k = 1 * self.kernel_size * self.kernel_size
        self.k_mlp = nn.Sequential(
            nn.Linear(self.base_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, flat_k),
        )

        r_in = self.base_channels + 2
        self.router = nn.Sequential(
            ConvBlock(r_in, r_in, kernel_size=3, norm=norm),
            nn.Conv2d(r_in, 4, kernel_size=1, bias=True),
        )

        self.branch_spe = SpectralRelearningBranch(self.base_channels, self.hsi_channels, norm=norm)
        self.branch_spa = SpatialRelearningBranch(self.base_channels, self.hsi_channels, norm=norm)
        self.branch_mix = MixedRelearningBranch(self.base_channels, self.hsi_channels, norm=norm)

        gc = self.guidance_channels
        bc = self.base_channels
        if self.use_signed_residual_guidance:
            self.m_res_proj = _GuidanceProj(self.msi_channels, gc, kernel_size=3, padding=1)
            self.h_res_proj = _GuidanceProj(self.hsi_channels, gc, kernel_size=1, padding=0)
        if self.use_msi_detail_guidance:
            self.m_detail_proj = _GuidanceProj(self.msi_channels, gc, kernel_size=3, padding=1)
        if self.use_guidance:
            if self.use_signed_residual_guidance:
                self.spe_fuse = _BranchFuse(bc + gc, bc)
                self.spa_fuse = _BranchFuse(bc + 2 * gc, bc)
                self.mix_fuse = _BranchFuse(bc + 3 * gc, bc)
            elif self.use_msi_detail_guidance:
                self.spa_fuse = _BranchFuse(bc + gc, bc)
                self.mix_fuse = _BranchFuse(bc + 2 * gc, bc)

        if self.use_reliability_gated_coarse_fusion:
            rel_in = 1 + self.msi_channels + self.msi_channels
            self.reliability_estimator = nn.Sequential(
                ConvBlock(rel_in, rel_in, kernel_size=3, norm=norm),
                nn.Conv2d(rel_in, self.msi_channels, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )

        if self.use_dual_observation_feedback:
            if self.num_feedback_steps < 1:
                raise ValueError("num_feedback_steps must be >= 1 when use_dual_observation_feedback is True")
            cell_kw = dict(
                hsi_channels=self.hsi_channels,
                base_channels=self.base_channels,
                scale=self.scale,
                norm=norm,
                residual_scale=self.feedback_residual_scale,
                observation_mode=self.feedback_observation_mode,
            )
            if self.share_feedback_weights:
                self.feedback_cell = DualObservationResidualFeedbackCell(**cell_kw)
                self.feedback_cells = None
            else:
                self.feedback_cell = None
                self.feedback_cells = nn.ModuleList(
                    [DualObservationResidualFeedbackCell(**cell_kw) for _ in range(self.num_feedback_steps)]
                )
        else:
            self.feedback_cell = None
            self.feedback_cells = None

        self.res_scale = 0.1
        self._init_stable()

    def _init_stable(self) -> None:
        """Near-identity start: zero residual heads, nominal φ/k deltas, reliable-biased gates."""
        for mlp in (self.phi_mlp, self.k_mlp):
            last = mlp[-1]
            assert isinstance(last, nn.Linear)
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

        nn.init.zeros_(self.branch_spe.pw2.weight)
        nn.init.zeros_(self.branch_spe.pw2.bias)
        nn.init.zeros_(self.branch_spa.head.weight)
        nn.init.zeros_(self.branch_spa.head.bias)
        nn.init.zeros_(self.branch_mix.head.weight)
        nn.init.zeros_(self.branch_mix.head.bias)

        router_last = self.router[-1]
        assert isinstance(router_last, nn.Conv2d)
        nn.init.zeros_(router_last.weight)
        nn.init.constant_(router_last.bias, 0.0)
        with torch.no_grad():
            router_last.bias.copy_(torch.tensor([2.0, -2.0, -2.0, -2.0], dtype=router_last.bias.dtype))

        if self.use_reliability_gated_coarse_fusion:
            rel_last = self.reliability_estimator[-2]
            assert isinstance(rel_last, nn.Conv2d)
            nn.init.zeros_(rel_last.weight)
            nn.init.constant_(rel_last.bias, 2.0)

    def _coarse_fusion(
        self,
        h: torch.Tensor,
        m: torch.Tensor,
        phi0_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Return ``(feat, z_coarse, coarse_diag)``; ``coarse_diag`` may be empty."""
        b, _, H, W = m.shape
        h_up = F.interpolate(h, size=(H, W), mode="bicubic", align_corners=False)
        coarse_diag: dict[str, torch.Tensor] = {}

        if not self.use_reliability_gated_coarse_fusion:
            x_in = torch.cat([h_up, m], dim=1)
            feat, _ = self.backbone(x_in)
            z_coarse = torch.sigmoid(self.z_coarse_head(feat))
            return feat, z_coarse, coarse_diag

        m0 = spectral_degrade(h_up, phi0_b)
        r0 = (m - m0).abs()
        r0_mean = r0.mean(dim=1, keepdim=True)
        rel_in = torch.cat([r0_mean, m, m0], dim=1)
        a = self.reliability_estimator(rel_in)
        m_safe = a * m + (1.0 - a) * m0
        x_in = torch.cat([h_up, m_safe, r0_mean], dim=1)
        feat, _ = self.backbone(x_in)
        z_coarse = torch.sigmoid(self.z_coarse_head(feat))
        coarse_diag = {
            "coarse_m0": m0,
            "coarse_r0": r0,
            "coarse_reliability_a": a,
            "coarse_m_safe": m_safe,
        }
        return feat, z_coarse, coarse_diag

    def _branch_feat(
        self,
        feat: torch.Tensor,
        *,
        h_res_guidance: Optional[torch.Tensor],
        m_res_guidance: Optional[torch.Tensor],
        m_detail_guidance: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.use_guidance:
            return self.branch_spe(feat), self.branch_spa(feat), self.branch_mix(feat)

        if self.use_signed_residual_guidance:
            assert h_res_guidance is not None and m_res_guidance is not None
            spe_in = torch.cat([feat, h_res_guidance], dim=1)
            spa_in = torch.cat([feat, m_res_guidance], dim=1)
            mix_in = torch.cat([feat, h_res_guidance, m_res_guidance], dim=1)
            if self.use_msi_detail_guidance:
                assert m_detail_guidance is not None
                spa_in = torch.cat([spa_in, m_detail_guidance], dim=1)
                mix_in = torch.cat([mix_in, m_detail_guidance], dim=1)
            spe_feat = self.spe_fuse(spe_in)
            spa_feat = self.spa_fuse(spa_in)
            mix_feat = self.mix_fuse(mix_in)
        else:
            assert m_detail_guidance is not None
            spa_in = torch.cat([feat, m_detail_guidance], dim=1)
            mix_in = torch.cat([feat, m_detail_guidance], dim=1)
            spe_feat = feat
            spa_feat = self.spa_fuse(spa_in)
            mix_feat = self.mix_fuse(mix_in)

        return (
            self.branch_spe(spe_feat),
            self.branch_spa(spa_feat),
            self.branch_mix(mix_feat),
        )

    def forward(
        self,
        batch: dict[str, Any],
        return_intermediates: bool = False,
    ) -> dict[str, torch.Tensor]:
        h: torch.Tensor = batch["h"]
        m: torch.Tensor = batch["m"]
        phi0: torch.Tensor = batch["phi0"]
        k0: torch.Tensor = batch["k0"]

        if h.ndim != 4 or m.ndim != 4:
            raise ValueError(f"h and m must be 4D, got h={tuple(h.shape)}, m={tuple(m.shape)}")
        b, c_h, h_lr, w_lr = h.shape
        b_m, c_m, H, W = m.shape
        if b != b_m:
            raise ValueError(f"batch mismatch: h B={b}, m B={b_m}")
        if c_h != self.hsi_channels or c_m != self.msi_channels:
            raise ValueError(
                f"channel mismatch: expected C_h={self.hsi_channels}, C_m={self.msi_channels}, "
                f"got C_h={c_h}, C_m={c_m}"
            )
        if H != h_lr * self.scale or W != w_lr * self.scale:
            raise ValueError(
                f"LR/HR spatial mismatch: h is {h_lr}x{w_lr}, m is {H}x{W}, scale={self.scale} "
                f"(expected H={h_lr * self.scale}, W={w_lr * self.scale})."
            )
        if H % 4 != 0 or W % 4 != 0:
            raise ValueError(f"H and W must be divisible by 4 for the U-Net backbone, got H={H}, W={W}")

        phi0_b = _broadcast_phi0(phi0, b)
        if phi0_b.shape[1:] != (self.msi_channels, self.hsi_channels):
            raise ValueError(
                f"phi0 must end with shape [C_m, C_h]=[{self.msi_channels}, {self.hsi_channels}], "
                f"got {tuple(phi0_b.shape)}"
            )

        k0_b = _broadcast_k0(k0, b)
        kk = k0_b.shape[-1]
        if kk != self.kernel_size or k0_b.shape[-2] != self.kernel_size:
            raise ValueError(
                f"k0 spatial size must be [{self.kernel_size}, {self.kernel_size}], got {tuple(k0_b.shape)}"
            )

        # --- 1) Coarse fusion ---
        m_coarse = m
        _msi_alpha = batch.get("diagnostic_msi_alpha")
        if _msi_alpha is not None:
            alpha = float(_msi_alpha)
            if alpha < 0.0 or alpha > 1.0:
                raise ValueError(f"diagnostic_msi_alpha must be in [0, 1], got {alpha}")
            if alpha < 1.0 - 1e-9:
                h_up = F.interpolate(h, size=(H, W), mode="bicubic", align_corners=False)
                m0 = spectral_degrade(h_up, phi0_b)
                m_coarse = alpha * m + (1.0 - alpha) * m0
        feat, z_coarse, coarse_diag = self._coarse_fusion(h, m_coarse, phi0_b)

        # --- 2) Semi-blind calibration ---
        vec = feat.mean(dim=(2, 3))
        d_phi_raw = self.phi_mlp(vec).view(b, self.msi_channels, self.hsi_channels)
        delta_phi = self.delta_phi_scale * torch.tanh(d_phi_raw)
        phi_tilde = normalize_srf(phi0_b + delta_phi)

        d_k_raw = self.k_mlp(vec).view(b, 1, self.kernel_size, self.kernel_size)
        delta_k = self.delta_k_scale * torch.tanh(d_k_raw)
        k_tilde = normalize_kernel(k0_b + delta_k)

        # --- 3) Degradation-consistency diagnosis ---
        m_hat_coarse = spectral_degrade(z_coarse, phi_tilde)
        h_hat_coarse = spatial_degrade(z_coarse, k_tilde, self.scale)

        signed_m_residual = m - m_hat_coarse
        signed_h_residual = h - h_hat_coarse
        signed_h_residual_up = F.interpolate(
            signed_h_residual,
            size=z_coarse.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        # Router: absolute inconsistency (unchanged diagnosis semantics)
        r_spe = (m_hat_coarse - m).abs().mean(dim=1, keepdim=True)
        r_spa_lr = (h_hat_coarse - h).abs().mean(dim=1, keepdim=True)
        r_spa = F.interpolate(r_spa_lr, size=(H, W), mode="bilinear", align_corners=False)

        # --- 4) Signed / detail guidance (optional) ---
        m_res_guidance: Optional[torch.Tensor] = None
        h_res_guidance: Optional[torch.Tensor] = None
        m_detail_guidance: Optional[torch.Tensor] = None
        detail_m_residual: Optional[torch.Tensor] = None

        if self.use_msi_detail_guidance:
            m_residual_low = F.avg_pool2d(signed_m_residual, kernel_size=5, stride=1, padding=2)
            detail_m_residual = signed_m_residual - m_residual_low
            m_detail_guidance = self.m_detail_proj(detail_m_residual)

        if self.use_signed_residual_guidance:
            m_res_guidance = self.m_res_proj(signed_m_residual)
            h_res_guidance = self.h_res_proj(signed_h_residual_up)

        # Diagnostic-only guidance ablation (E4.2; ignored unless batch key is set).
        _guidance_iv = batch.get("diagnostic_guidance_intervention")
        if _guidance_iv is not None:
            _iv = str(_guidance_iv).strip().lower()
            if _iv in ("zero_m_detail_guidance", "zero_m_detail"):
                if m_detail_guidance is not None:
                    m_detail_guidance = torch.zeros_like(m_detail_guidance)
            elif _iv in ("zero_signed_m_guidance", "zero_signed_m", "zero_m_res"):
                if m_res_guidance is not None:
                    m_res_guidance = torch.zeros_like(m_res_guidance)
            elif _iv in ("zero_all_msi_guidance", "zero_all_msi"):
                if m_detail_guidance is not None:
                    m_detail_guidance = torch.zeros_like(m_detail_guidance)
                if m_res_guidance is not None:
                    m_res_guidance = torch.zeros_like(m_res_guidance)
            elif _iv not in ("full_b8", "full", "none", ""):
                raise ValueError(
                    f"unknown diagnostic_guidance_intervention {_guidance_iv!r}; "
                    "expected full_b8, zero_m_detail_guidance, zero_signed_m_guidance, or zero_all_msi_guidance"
                )

        # --- 5) Router ---
        router_in = torch.cat([feat, r_spe, r_spa], dim=1)
        logits = self.router(router_in)
        gates = F.softmax(logits, dim=1)

        # --- 6) Selective re-learning ---
        delta_spe, delta_spa, delta_mix = self._branch_feat(
            feat,
            h_res_guidance=h_res_guidance,
            m_res_guidance=m_res_guidance,
            m_detail_guidance=m_detail_guidance,
        )
        weighted_residual = (
            gates[:, 1:2] * delta_spe
            + gates[:, 2:3] * delta_spa
            + gates[:, 3:4] * delta_mix
        )
        z_hat_raw = z_coarse + self.res_scale * torch.tanh(weighted_residual)
        if self.training:
            z_hat = z_hat_raw
        else:
            z_hat = z_hat_raw.clamp(0.0, 1.0)

        z_stage1 = z_hat
        z_feedback_intermediate: torch.Tensor | None = None
        want_inter = return_intermediates or bool(batch.get("return_intermediates"))
        export_diag = bool(batch.get("diagnostic_export")) or want_inter
        feedback_diag: dict[str, torch.Tensor] = {}
        feedback_step_tensors: list[dict[str, torch.Tensor]] = []
        z_feedback_steps: list[torch.Tensor] = []
        if self.use_dual_observation_feedback:
            z_t = z_stage1
            if want_inter:
                z_feedback_steps.append(z_t)
            for step in range(self.num_feedback_steps):
                cell = (
                    self.feedback_cell
                    if self.share_feedback_weights
                    else self.feedback_cells[step]  # type: ignore[index]
                )
                z_t, step_diag = cell(
                    z_t,
                    m,
                    h,
                    phi0_b,
                    k0_b,
                    return_diagnostics=export_diag,
                )
                if step == 0:
                    z_feedback_intermediate = z_t
                if want_inter:
                    z_feedback_steps.append(z_t)
                    if step_diag:
                        feedback_step_tensors.append(
                            {k: v for k, v in step_diag.items() if isinstance(v, torch.Tensor)}
                        )
                if export_diag and not want_inter:
                    for k, v in step_diag.items():
                        feedback_diag[f"feedback_s{step}_{k}"] = v
            z_hat = z_t if self.training else z_t.clamp(0.0, 1.0)

        z_hat_for_loss = z_hat
        m_hat = spectral_degrade(z_hat_for_loss, phi_tilde)
        h_hat = spatial_degrade(z_hat_for_loss, k_tilde, self.scale)

        out: dict[str, torch.Tensor] = {
            "z_hat": z_hat,
            "z_coarse": z_coarse,
            "phi_tilde": phi_tilde,
            "k_tilde": k_tilde,
            "m_hat_coarse": m_hat_coarse,
            "h_hat_coarse": h_hat_coarse,
            "m_hat": m_hat,
            "h_hat": h_hat,
            "r_spe": r_spe,
            "r_spa": r_spa,
            "gates": gates,
            "delta_spe": delta_spe,
            "delta_spa": delta_spa,
            "delta_mix": delta_mix,
            "signed_m_residual": signed_m_residual,
            "signed_h_residual": signed_h_residual,
            "signed_h_residual_up": signed_h_residual_up,
            "weighted_residual": weighted_residual,
        }
        if detail_m_residual is not None:
            out["detail_m_residual"] = detail_m_residual
        if m_res_guidance is not None:
            out["m_res_guidance"] = m_res_guidance
        if h_res_guidance is not None:
            out["h_res_guidance"] = h_res_guidance
        if m_detail_guidance is not None:
            out["m_detail_guidance"] = m_detail_guidance
        if self.use_dual_observation_feedback:
            out["z_stage1"] = z_stage1
            if z_feedback_intermediate is not None:
                out["z_feedback_intermediate"] = z_feedback_intermediate
        if want_inter and z_feedback_steps:
            out["z_feedback_steps"] = torch.stack(z_feedback_steps, dim=1)
            if feedback_step_tensors:
                out["feedback_step_tensors"] = feedback_step_tensors
        if export_diag and want_inter and feedback_step_tensors:
            for step, step_diag in enumerate(feedback_step_tensors):
                for k, v in step_diag.items():
                    if k.endswith("_l1_mean"):
                        feedback_diag[f"feedback_s{step}_{k}"] = v
                    else:
                        feedback_diag[f"feedback_s{step}_{k}"] = v
        if batch.get("diagnostic_export"):
            out["z_hat_raw"] = z_hat_raw
            if self.use_dual_observation_feedback:
                out["z_before_feedback"] = z_stage1
                out["z_after_feedback"] = z_hat
        if coarse_diag:
            out.update(coarse_diag)
        if feedback_diag:
            out.update(feedback_diag)
        return out
