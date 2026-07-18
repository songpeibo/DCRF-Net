"""
Faithful PyTorch CMHF-net **HSInet** (BCHW), aligned with
``docs/baselines/mhfnet_pytorch_port_spec.md`` and ``third_party/baselines/MHF-net/CMHF-net/MHFnet.py``.

Training I/O (``run_mhfnet_baseline.py``) may wrap this module with ``torch.compile`` when requested; the
``nn.Module`` graph is unchanged—compile is an optional runtime optimization only.

CMHF **does not** use ``torch.nn.PixelShuffle`` / depth-to-space; upsampling follows TF
``conv2d_transpose`` (``UpsumLevel2``). ``MHFPixelShuffleScale`` chains those 2× steps to
reach ``scale`` when ``scale`` is a power of two.

Public entry: ``MHFNetFaithful.forward(lr_hsi, hr_msi, srf=None, psf=None)``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "MHFTransposeUpsample2x",
    "MHFPhaseShift",
    "MHFPixelShuffleScale",
    "MHFInitModule",
    "MHFResLevel",
    "MHFResLevelAddF",
    "MHFSubNet",
    "MHFUpSamBranch",
    "MHFDownSamStack",
    "MHFStage",
    "MHFNetFaithful",
    "MHFNetFaithfulOutputs",
]


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _depthwise_same(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x [B,C,H,W], w [C,1,k,k] depthwise."""
    kh, kw = w.shape[2], w.shape[3]
    pad_h = (kh - 1) // 2
    pad_w = (kw - 1) // 2
    return F.conv2d(x, w, None, 1, (pad_h, pad_w), 1, x.size(1))


def _down_stride4(x: torch.Tensor) -> torch.Tensor:
    """TF ``[:, 1:-1:4, 1:-1:4]`` on H,W (BCHW)."""
    return x[:, :, 1:-1:4, 1:-1:4]


def _down_stride2_last(x: torch.Tensor) -> torch.Tensor:
    """TF ``[:, 0:-1:2, 0:-1:2]``."""
    return x[:, :, 0:-1:2, 0:-1:2]


class MHFTransposeUpsample2x(nn.Module):
    """
    TF ``UpsumLevel2``: ``ConvTranspose2d`` k×k, stride 2, SAME output size ≈ 2× spatial.
    Weight init: ``ini_tile / 4`` with ``ini_tile`` shape ``[k,k,C,C]`` (CAVE ``iniUp1`` layout).
    """

    def __init__(
        self,
        channels: int,
        ini_tile: torch.Tensor,
        *,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for symmetric SAME-style transpose")
        self.kernel_size = int(kernel_size)
        pad = kernel_size // 2
        self.deconv = nn.ConvTranspose2d(
            channels, channels, kernel_size=self.kernel_size, stride=2, padding=pad, bias=False
        )
        it = ini_tile.detach().float()
        if it.shape != (self.kernel_size, self.kernel_size, channels, channels):
            raise ValueError(
                f"ini_tile must be [{self.kernel_size},{self.kernel_size},{channels},{channels}], got {tuple(it.shape)}"
            )
        w_pt = (it / 4.0).permute(2, 3, 0, 1).contiguous()
        with torch.no_grad():
            self.deconv.weight.copy_(w_pt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h2 = x.size(-2) * 2
        w2 = x.size(-1) * 2
        return self.deconv(x, output_size=(x.size(0), x.size(1), h2, w2))


# CMHF uses transpose conv for sub-pixel style upsampling, not literal phase-shift in frequency domain.
MHFPhaseShift = MHFTransposeUpsample2x


class MHFPixelShuffleScale(nn.Module):
    """
    Spatial upsampling by repeated **2×** ``MHFTransposeUpsample2x`` blocks so total factor is ``scale``,
    when ``scale`` is a power of two (e.g. 4 → two blocks). This matches CMHF ``UpSam`` tail (``Cfilter4``/``5``).
    **Not** ``nn.PixelShuffle``.
    """

    def __init__(
        self,
        channels: int,
        ini_tile: torch.Tensor,
        *,
        scale: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if not _is_power_of_two(scale):
            raise ValueError(f"MHFPixelShuffleScale only supports power-of-2 scale, got {scale}")
        self.scale = int(scale)
        n = int(round(math.log2(self.scale)))
        self.blocks = nn.ModuleList(
            [MHFTransposeUpsample2x(channels, ini_tile, kernel_size=kernel_size) for _ in range(n)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return x


class MHFBatchNormMoments2d(nn.Module):
    """BN with batch+spatial moments only (``track_running_stats=False``)."""

    def __init__(self, num_features: int, *, scale_init: float) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, eps=1e-5, affine=True, track_running_stats=False)
        nn.init.constant_(self.bn.weight, scale_init)
        nn.init.constant_(self.bn.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)


class MHFResLevel(nn.Module):
    """TF ``resLevel`` (three convs + residual); order weights1 → weights3 → weights2."""

    def __init__(
        self,
        channel: int,
        *,
        kernel_size: int,
        feature_mid_delta: int,
        bn_scale_div: int = 20,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        p = kernel_size // 2
        c = channel
        d = int(feature_mid_delta)
        self.conv1 = nn.Conv2d(c, c + d, kernel_size, padding=p, bias=True)
        self.bn1 = MHFBatchNormMoments2d(c + d, scale_init=1.0 / bn_scale_div)
        self.conv2 = nn.Conv2d(c + d, c + d, kernel_size, padding=p, bias=True)
        self.bn2 = MHFBatchNormMoments2d(c + d, scale_init=1.0 / bn_scale_div)
        self.conv3 = nn.Conv2d(c + d, c, kernel_size, padding=p, bias=True)
        self.bn3 = MHFBatchNormMoments2d(c, scale_init=1.0 / bn_scale_div)
        for m in (self.conv1, self.conv2, self.conv3):
            nn.init.trunc_normal_(m.weight, std=0.1)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.bn1(y)
        y = F.relu(y, inplace=False)
        y = self.conv2(y)
        y = self.bn2(y)
        y = F.relu(y, inplace=False)
        y = self.conv3(y)
        y = self.bn3(y)
        y = F.relu(y, inplace=False)
        return x + y


class MHFResLevelAddF(nn.Module):
    """TF ``resLevel_addF`` — concat ``X`` and scaled prior ``Y`` on channel axis."""

    def __init__(
        self,
        channel_x: int,
        channel_y: int,
        *,
        kernel_size: int,
        feature_mid_delta: int,
        bn_scale_div: int = 100,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        p = kernel_size // 2
        d = int(feature_mid_delta)
        ch = channel_x + channel_y
        self.conv1 = nn.Conv2d(ch, ch + d, kernel_size, padding=p, bias=True)
        self.bn1 = MHFBatchNormMoments2d(ch + d, scale_init=1.0 / bn_scale_div)
        self.conv2 = nn.Conv2d(ch + d, ch + d, kernel_size, padding=p, bias=True)
        self.bn2 = MHFBatchNormMoments2d(ch + d, scale_init=1.0 / bn_scale_div)
        self.conv3 = nn.Conv2d(ch + d, channel_x, kernel_size, padding=p, bias=True)
        self.bn3 = MHFBatchNormMoments2d(channel_x, scale_init=1.0 / bn_scale_div)
        for m in (self.conv1, self.conv2, self.conv3):
            nn.init.trunc_normal_(m.weight, std=0.1)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, y_scaled: torch.Tensor) -> torch.Tensor:
        z = torch.cat([x, y_scaled], dim=1)
        o = self.conv1(z)
        o = self.bn1(o)
        o = F.relu(o, inplace=False)
        o = self.conv2(o)
        o = self.bn2(o)
        o = F.relu(o, inplace=False)
        o = self.conv3(o)
        o = self.bn3(o)
        o = F.relu(o, inplace=False)
        return x + o


class MHFSubNet(nn.Module):
    """
    TF ``resCNNnet``: ``(level_n - 1)`` copies of ``MHFResLevel`` at ``channel``.
    Proximal nets use ``channel=up_rank``, ``level_n=subnet_l``; ``FinalAjust`` uses ``out_dim``, ``level_n=5``.
    """

    def __init__(
        self,
        channel: int,
        level_n: int,
        *,
        kernel_size: int,
        feature_mid_delta: int,
        bn_scale_div: int = 20,
    ) -> None:
        super().__init__()
        n_blocks = max(0, int(level_n) - 1)
        self.blocks = nn.ModuleList(
            [
                MHFResLevel(channel, kernel_size=kernel_size, feature_mid_delta=feature_mid_delta, bn_scale_div=bn_scale_div)
                for _ in range(n_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


class MHFDepthwiseBlur(nn.Module):
    """Learnable depthwise with shared spatial filter per channel (``getCs`` / ``Blur``)."""

    def __init__(self, kernel_size: int, init_value: float) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for symmetric padding")
        self.kernel_size = int(kernel_size)
        w = torch.full((1, 1, self.kernel_size, self.kernel_size), float(init_value))
        self.weight = nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.size(1)
        dw = self.weight.expand(c, 1, self.kernel_size, self.kernel_size).contiguous()
        return _depthwise_same(x, dw)


class MHFDownSamStack(nn.Module):
    """TF ``getCs`` + ``downSam`` (shared C filters across calls).

    TF uses a **6×6** depthwise PSF; this port uses **7×7** (odd) so ``MHFDepthwiseBlur`` can use symmetric padding.
    Init uses ``1/49`` so tile mass matches ``1/36`` intent at similar scale.
    """

    def __init__(
        self,
        ratio: int,
        *,
        psf_kernel_large: int = 7,
        psf_kernel_small: int = 3,
        init_large: float = 1.0 / 49.0,
        init_small: float = 1.0 / 9.0,
    ) -> None:
        super().__init__()
        self.ratio = int(ratio)
        if psf_kernel_large % 2 == 0 or psf_kernel_small % 2 == 0:
            raise ValueError(
                "PSF kernels must be odd (symmetric padding in MHFDepthwiseBlur); "
                f"got psf_kernel_large={psf_kernel_large} psf_kernel_small={psf_kernel_small}"
            )
        self.psf_kernel_large = int(psf_kernel_large)
        self.psf_kernel_small = int(psf_kernel_small)
        self.f1 = MHFDepthwiseBlur(self.psf_kernel_large, init_large)
        self.f2: Optional[MHFDepthwiseBlur] = None
        self.f3: Optional[MHFDepthwiseBlur] = None
        if self.ratio > 4:
            self.f2 = MHFDepthwiseBlur(self.psf_kernel_large, init_large)
        if self.ratio > 16:
            self.f3 = MHFDepthwiseBlur(self.psf_kernel_small, init_small)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c = x.size(1)
        w1 = self.f1.weight.expand(c, 1, self.psf_kernel_large, self.psf_kernel_large).contiguous()
        x = _depthwise_same(x, w1)
        d4 = _down_stride4(x)
        if self.ratio == 4:
            return d4, d4, d4
        assert self.f2 is not None
        w2 = self.f2.weight.expand(c, 1, self.psf_kernel_large, self.psf_kernel_large).contiguous()
        x = _depthwise_same(d4, w2)
        d16 = _down_stride4(x)
        if self.ratio == 16:
            return d4, d16, d16
        assert self.f3 is not None
        w3 = self.f3.weight.expand(c, 1, self.psf_kernel_small, self.psf_kernel_small).contiguous()
        x = _depthwise_same(d16, w3)
        d32 = _down_stride2_last(x)
        return d4, d16, d32


class MHFUpSamBranch(nn.Module):
    """One TF ``UpSam`` scope: ratio-dependent ladder + ``Blur`` (depthwise; TF uses **4×4** @ ``1/16``).

    Default ``blur_kernel_size=5`` (odd) for symmetric padding in ``MHFDepthwiseBlur``; init ``1/25`` normalizes tile sum ≈1 like TF.
    """

    def __init__(
        self,
        out_dim: int,
        msi_channels: int,
        ratio: int,
        ini_up_tile: torch.Tensor,
        *,
        kernel_size: int,
        feature_mid_delta: int,
        blur_kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.ratio = int(ratio)
        self.out_dim = int(out_dim)
        self.msi_channels = int(msi_channels)

        if int(blur_kernel_size) % 2 == 0:
            raise ValueError(
                f"blur_kernel_size must be odd (MHFDepthwiseBlur / symmetric padding); got {blur_kernel_size}"
            )

        self.ups = nn.ModuleList()
        self.addfs = nn.ModuleList()
        if self.ratio == 32:
            self.ups.append(MHFTransposeUpsample2x(out_dim, ini_up_tile, kernel_size=kernel_size))
            self.addfs.append(
                MHFResLevelAddF(out_dim, msi_channels, kernel_size=kernel_size, feature_mid_delta=feature_mid_delta)
            )
        if self.ratio >= 16:
            self.ups.append(MHFTransposeUpsample2x(out_dim, ini_up_tile, kernel_size=kernel_size))
            self.ups.append(MHFTransposeUpsample2x(out_dim, ini_up_tile, kernel_size=kernel_size))
            self.addfs.append(
                MHFResLevelAddF(out_dim, msi_channels, kernel_size=kernel_size, feature_mid_delta=feature_mid_delta)
            )
        self.ups.append(MHFTransposeUpsample2x(out_dim, ini_up_tile, kernel_size=kernel_size))
        self.ups.append(MHFTransposeUpsample2x(out_dim, ini_up_tile, kernel_size=kernel_size))
        self.addfs.append(
            MHFResLevelAddF(out_dim, msi_channels, kernel_size=kernel_size, feature_mid_delta=feature_mid_delta)
        )
        self.blur = MHFDepthwiseBlur(blur_kernel_size, 1.0 / float(blur_kernel_size * blur_kernel_size))

    def forward(self, x: torch.Tensor, y_hr: torch.Tensor, down_y4: torch.Tensor, down_y16: torch.Tensor) -> torch.Tensor:
        y_s = y_hr / 10.0
        d4 = down_y4 / 10.0
        d16 = down_y16 / 10.0
        ui = 0
        ai = 0
        if self.ratio == 32:
            x = self.ups[ui](x)
            ui += 1
            x = self.addfs[ai](x, d16)
            ai += 1
        if self.ratio >= 16:
            x = self.ups[ui](x)
            ui += 1
            x = self.ups[ui](x)
            ui += 1
            x = self.addfs[ai](x, d4)
            ai += 1
        x = self.ups[ui](x)
        ui += 1
        x = self.ups[ui](x)
        ui += 1
        x = self.addfs[ai](x, y_s)
        ai += 1
        assert ui == len(self.ups) and ai == len(self.addfs)
        return self.blur(x)


class MHFInitModule(nn.Module):
    """
    Builds **iniUp** (``[k,k,C,C]`` tile of identity / 4 for transpose init) and **iniA** (1×1 MSI→HSI).

    - **ini_up**: faithful to CAVE ``iniUp1 = tile(eye(C), [k,k,1,1])`` (not bicubic).
    - **ini_a**: optional ``srf.npy`` array shape ``[msi_channels, hsi_channels]``; else zeros (TF ``test`` scalar 0).
    """

    def __init__(
        self,
        *,
        hsi_channels: int,
        msi_channels: int,
        kernel_size: int = 3,
        srf_path: Optional[Path] = None,
    ) -> None:
        super().__init__()
        self.hsi_channels = int(hsi_channels)
        self.msi_channels = int(msi_channels)
        self.kernel_size = int(kernel_size)
        self.debug: Dict[str, Any] = {
            "ini_up_source": "identity_tile_per_cave_iniUp1",
            "ini_up_note": "CMHF uses identity tiled [K,K,C,C] / 4 for ConvTranspose2d init, not bicubic.",
            "ini_a_source": "zeros",
            "srf_path": str(srf_path) if srf_path else None,
        }
        eye = torch.eye(self.hsi_channels)
        ini_up = eye.view(1, 1, self.hsi_channels, self.hsi_channels).expand(
            self.kernel_size, self.kernel_size, self.hsi_channels, self.hsi_channels
        ).contiguous()
        self.register_buffer("ini_up_template", ini_up.clone())

        ini_a = torch.zeros(self.msi_channels, self.hsi_channels)
        if srf_path is not None:
            p = Path(srf_path)
            if p.is_file():
                arr = np.load(str(p))
                if arr.shape != (self.msi_channels, self.hsi_channels):
                    self.debug["ini_a_error"] = f"srf shape {arr.shape} != ({self.msi_channels},{self.hsi_channels})"
                else:
                    ini_a = torch.from_numpy(arr.astype(np.float32))
                    self.debug["ini_a_source"] = "srf_npy"
            else:
                self.debug["ini_a_error"] = f"missing file {p}"
        self.register_buffer("ini_a_template", ini_a.clone())

    def get_ini_up_tile(self) -> torch.Tensor:
        return self.ini_up_template.clone()

    def build_my_conv_a(self) -> nn.Conv2d:
        conv = nn.Conv2d(self.msi_channels, self.hsi_channels, kernel_size=1, bias=False)
        with torch.no_grad():
            conv.weight.copy_(self.ini_a_template.view(self.hsi_channels, self.msi_channels, 1, 1))
        return conv

    def export_debug_stats(self) -> Dict[str, Any]:
        return dict(self.debug)


class MHFStage(nn.Module):
    """
    One HSInet **inner-loop** iteration (after ``YA`` is fixed): ``UpSam`` branch for stage ``E{j+2}``.

    Proximal subnet ``Pri{j+2}`` is **shared from** ``MHFNetFaithful.prox_subnets[j+1]`` (must match TF
    separate variable scopes but identical architecture).
    """

    def __init__(
        self,
        *,
        hsi_channels: int,
        msi_channels: int,
        ratio: int,
        ini_up_tile: torch.Tensor,
        kernel_size: int,
        feature_mid_delta: int,
        blur_kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.up_branch = MHFUpSamBranch(
            hsi_channels,
            msi_channels,
            ratio,
            ini_up_tile,
            kernel_size=kernel_size,
            feature_mid_delta=feature_mid_delta,
            blur_kernel_size=blur_kernel_size,
        )

    def forward_iter(
        self,
        hy: torch.Tensor,
        ya: torch.Tensor,
        y_hr: torch.Tensor,
        z_lr: torch.Tensor,
        down_y4: torch.Tensor,
        down_y16: torch.Tensor,
        cstack: MHFDownSamStack,
        b_hy_to_out: nn.Conv2d,
        g_out_to_up: nn.Conv2d,
        prox_subnet: MHFSubNet,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hyb = b_hy_to_out(hy)
        x_hat = ya + hyb
        _, _, dx = cstack(x_hat)
        e = dx - z_lr
        g = self.up_branch(e, y_hr, down_y4, down_y16)
        g = g_out_to_up(g)
        hy_next = hy - g
        hy_next = prox_subnet(hy_next)
        return hy_next, x_hat


class MHFNetFaithfulOutputs(NamedTuple):
    pred: torch.Tensor
    list_x: List[torch.Tensor]
    ya: torch.Tensor
    e_final: torch.Tensor
    hy: torch.Tensor
    debug_stats: Dict[str, Any]


class MHFNetFaithful(nn.Module):
    """
    CMHF ``HSInet`` in BCHW.

    ``scale`` replaces TF ``ratio`` (total spatial factor between ``hr_msi`` and ``lr_hsi``).
    Must be a power of two (4, 8, 16, 32, …) to match the chained 2× transpose ladder in ``UpSam``.
    """

    def __init__(
        self,
        *,
        hsi_channels: int = 103,
        msi_channels: int = 3,
        scale: int = 4,
        up_rank: int = 12,
        hsinet_l: int = 20,
        subnet_l: int = 2,
        feature_mid_delta: int = 3,
        kernel_size: int = 3,
        psf_kernel_large: int = 7,
        psf_kernel_small: int = 3,
        blur_kernel_size: int = 5,
        final_level_n: int = 5,
        srf_path: Optional[Path] = None,
    ) -> None:
        super().__init__()
        if hsinet_l < 2:
            raise ValueError("hsinet_l must be >= 2")
        if subnet_l < 2:
            raise ValueError("subnet_l must be >= 2 (resCNNnet uses level_n-1 >= 1)")
        if not _is_power_of_two(scale):
            raise ValueError(f"scale must be a power of 2, got {scale}")
        self.hsi_channels = int(hsi_channels)
        self.msi_channels = int(msi_channels)
        self.scale = int(scale)
        self.up_rank = int(up_rank)
        self.hsinet_l = int(hsinet_l)
        self.subnet_l = int(subnet_l)
        self.feature_mid_delta = int(feature_mid_delta)
        self.kernel_size = int(kernel_size)
        self.final_level_n = int(final_level_n)
        self.blur_kernel_size = int(blur_kernel_size)

        self.init_mod = MHFInitModule(
            hsi_channels=self.hsi_channels,
            msi_channels=self.msi_channels,
            kernel_size=self.kernel_size,
            srf_path=srf_path,
        )
        ini_tile = self.init_mod.get_ini_up_tile()
        self.my_conv_a = self.init_mod.build_my_conv_a()

        self.c_stack = MHFDownSamStack(self.scale, psf_kernel_large=psf_kernel_large, psf_kernel_small=psf_kernel_small)

        self.b_hy_to_out = nn.Conv2d(self.up_rank, self.hsi_channels, 1, bias=False)
        nn.init.trunc_normal_(self.b_hy_to_out.weight, std=0.1)
        self.g_out_to_up = nn.Conv2d(self.hsi_channels, self.up_rank, 1, bias=False)
        nn.init.trunc_normal_(self.g_out_to_up.weight, std=0.1)

        n_up = self.hsinet_l - 1
        self.up_branches = nn.ModuleList(
            [
                MHFUpSamBranch(
                    self.hsi_channels,
                    self.msi_channels,
                    self.scale,
                    ini_tile.clone(),
                    kernel_size=self.kernel_size,
                    feature_mid_delta=self.feature_mid_delta,
                    blur_kernel_size=self.blur_kernel_size,
                )
                for _ in range(n_up)
            ]
        )

        n_prox = self.hsinet_l - 1
        self.prox_subnets = nn.ModuleList(
            [
                MHFSubNet(
                    self.up_rank,
                    self.subnet_l,
                    kernel_size=self.kernel_size,
                    feature_mid_delta=self.feature_mid_delta,
                )
                for _ in range(n_prox)
            ]
        )

        self.stages = nn.ModuleList(
            [
                MHFStage(
                    hsi_channels=self.hsi_channels,
                    msi_channels=self.msi_channels,
                    ratio=self.scale,
                    ini_up_tile=ini_tile.clone(),
                    kernel_size=self.kernel_size,
                    feature_mid_delta=self.feature_mid_delta,
                    blur_kernel_size=self.blur_kernel_size,
                )
                for _ in range(max(0, self.hsinet_l - 2))
            ]
        )

        self.final_subnet = MHFSubNet(
            self.hsi_channels,
            self.final_level_n,
            kernel_size=self.kernel_size,
            feature_mid_delta=self.feature_mid_delta,
        )

    def forward(
        self,
        lr_hsi: torch.Tensor,
        hr_msi: torch.Tensor,
        srf: Optional[torch.Tensor] = None,
        psf: Optional[torch.Tensor] = None,
    ) -> MHFNetFaithfulOutputs:
        """
        ``lr_hsi``: ``[B,C_hsi,h,w]``, ``hr_msi``: ``[B,C_msi,H,W]`` with ``H==h*scale``, ``W==w*scale``.
        ``srf`` / ``psf``: reserved; if ``srf`` is ``[msi, hsi]`` it could override ``ini_a`` in a future revision
        (currently use ``MHFInitModule`` / constructor ``srf_path`` only). ``psf`` is recorded in ``debug_stats`` only.
        """
        dbg = self.init_mod.export_debug_stats()
        if psf is not None:
            dbg["psf_passed"] = True
            dbg["psf_shape"] = list(psf.shape)
            dbg["psf_note"] = "CMHF HSInet does not ingest PSF tensors; not applied in this faithful port."

        b, cc, hh, ww = lr_hsi.shape
        B, cy, H, W = hr_msi.shape
        assert b == B
        assert cc == self.hsi_channels, f"lr_hsi channels {cc} != hsi_channels={self.hsi_channels}"
        assert cy == self.msi_channels, f"hr_msi channels {cy} != msi_channels={self.msi_channels}"
        assert H == hh * self.scale and W == ww * self.scale, (
            f"spatial mismatch lr ({hh},{ww}) hr ({H},{W}) scale={self.scale}"
        )

        if srf is not None:
            sr = srf.detach().float().to(device=lr_hsi.device, dtype=lr_hsi.dtype)
            if sr.ndim == 2:
                if sr.shape == (self.hsi_channels, self.msi_channels):
                    w = sr.view(self.hsi_channels, self.msi_channels, 1, 1)
                elif sr.shape == (self.msi_channels, self.hsi_channels):
                    w = sr.T.contiguous().view(self.hsi_channels, self.msi_channels, 1, 1)
                else:
                    w = None
                    dbg["srf_error"] = f"expected ({self.msi_channels},{self.hsi_channels}) or transpose, got {tuple(sr.shape)}"
                if w is not None:
                    w = w.to(device=self.my_conv_a.weight.device, dtype=self.my_conv_a.weight.dtype)
                    with torch.no_grad():
                        self.my_conv_a.weight.copy_(w)
                    dbg["ini_a_source"] = "forward_srf_tensor"

        y = hr_msi
        z = lr_hsi
        down_y4, down_y16, _ = self.c_stack(y)
        ya = self.my_conv_a(y)

        _, _, dx0 = self.c_stack(ya)
        e0 = dx0 - z
        g0 = self.up_branches[0](e0, y, down_y4, down_y16)
        g0 = self.g_out_to_up(g0)
        hy = -g0
        hy = self.prox_subnets[0](hy)

        list_x: List[torch.Tensor] = []
        for j in range(self.hsinet_l - 2):
            hy, x_hat = self.stages[j].forward_iter(
                hy,
                ya,
                y,
                z,
                down_y4,
                down_y16,
                self.c_stack,
                self.b_hy_to_out,
                self.g_out_to_up,
                self.prox_subnets[j + 1],
            )
            list_x.append(x_hat)

        hyb = self.b_hy_to_out(hy)
        x_last = ya + hyb
        list_x.append(x_last)

        pred = self.final_subnet(list_x[-1])
        _, _, cx = self.c_stack(list_x[-1])
        e_final = cx - z

        assert pred.shape == (B, self.hsi_channels, H, W)
        assert pred.shape[2:] == hr_msi.shape[2:]
        assert pred.shape[1] == self.hsi_channels == lr_hsi.shape[1]

        dbg_out = dict(dbg)
        dbg_out["pred_shape"] = list(pred.shape)
        dbg_out["lr_shape"] = list(lr_hsi.shape)
        dbg_out["hr_msi_shape"] = list(hr_msi.shape)
        if self.blur_kernel_size != 4:
            dbg_out["blur_kernel_note"] = (
                "TF UpSam Blur is 4×4 (even); this port uses odd kernels only — see blur_kernel_size default (5)."
            )

        return MHFNetFaithfulOutputs(
            pred=pred,
            list_x=list_x,
            ya=ya,
            e_final=e_final,
            hy=hy,
            debug_stats=dbg_out,
        )


if __name__ == "__main__":
    def _shape_bracket(t: torch.Tensor) -> str:
        return "[" + ",".join(str(int(x)) for x in t.shape) + "]"

    torch.manual_seed(0)
    dev = torch.device("cpu")
    m = MHFNetFaithful(
        hsi_channels=103,
        msi_channels=3,
        scale=4,
        up_rank=12,
        hsinet_l=20,
        subnet_l=2,
    ).to(dev)
    lr = torch.randn(2, 103, 32, 32, device=dev)
    hr = torch.randn(2, 3, 128, 128, device=dev)
    print(f"input lr_hsi: {_shape_bracket(lr)}")
    print(f"input hr_msi: {_shape_bracket(hr)}")
    out = m(lr, hr)
    pred = out.pred
    print(f"output pred: {_shape_bracket(pred)}")
    assert pred.shape == (2, 103, 128, 128), pred.shape
    assert pred.shape[2] == hr.shape[2] and pred.shape[3] == hr.shape[3]
    assert pred.shape[1] == 103
    assert lr.shape[2] * 4 == hr.shape[2] and lr.shape[3] * 4 == hr.shape[3]
    p = Path(__file__).resolve().parent / "mhfnet_faithful_smoke_debug_stats.json"
    p.write_text(json.dumps(out.debug_stats, indent=2), encoding="utf-8")
    print("smoke_test_ok wrote", p)
