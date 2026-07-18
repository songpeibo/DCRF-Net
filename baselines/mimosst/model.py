# Copyright paper2 baseline port — MIMO-SST style fusion (device-neutral, scale/C configurable).
"""PyTorch reimplementation of the local ``third_party/baselines/MIMO-SST/Model.py`` ``Net`` graph.

Does **not** import upstream ``Model.Net`` (hard-coded 31 / scale 8 / missing ``PixelUnshuffle``).

Public model: :class:`MIMOSSTScale4` with ``forward(lr_hsi, hr_msi) -> dict`` keys
``pred_quarter``, ``pred_half``, ``pred`` (final HR-HSI).
"""
from __future__ import annotations

import numbers
import sys
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ---------------------------------------------------------------------------#
# Layer norm + attention (ported from upstream Model.py, device-neutral)
# ---------------------------------------------------------------------------#


def to_3d(x: torch.Tensor) -> torch.Tensor:
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int | Tuple[int, ...]) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int | Tuple[int, ...]) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm2d(nn.Module):
    def __init__(self, dim: int, layer_norm_type: str) -> None:
        super().__init__()
        if layer_norm_type == "BiasFree":
            self.body: nn.Module = BiasFreeLayerNorm(dim)
        else:
            self.body = WithBiasLayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class Attention1(nn.Module):
    def __init__(self, dim: int, num_heads: int, bias: bool) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, _c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        return k, q, v


class Attention2(nn.Module):
    def __init__(self, dim: int, num_heads: int, bias: bool) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(
        self, x: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        b, _c, h, w = x.shape
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w)
        return self.project_out(out)


class Attention3(nn.Module):
    """Upstream uses ``view(b, c, h, w)`` — safe when ``num_heads == 1`` (default)."""

    def __init__(self, dim: int, num_heads: int, bias: bool) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias
        )

    def forward(
        self, x: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        b, c, h, w = x.shape
        attn = torch.matmul(q / self.temperature, k.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)
        return output.view(b, c, h, w)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_expansion_factor: float,
        hidden_dim: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        use_eca: bool = False,
    ) -> None:
        super().__init__()
        hidden_features = dim if hidden_dim is None else hidden_dim
        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, hidden_features, kernel_size=3, stride=1, padding=1),
            act_layer(),
        )
        self.eca = ECA1d(hidden_features) if use_eca else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dwconv(x)
        return self.eca(x)


class ECA1d(nn.Module):
    def __init__(self, channel: int, k_size: int = 3) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x.transpose(-1, -2))
        y = self.conv(y.transpose(-1, -2))
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class TransformerBlock1(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_expansion_factor: float,
        bias: bool,
        layer_norm_type: str,
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(dim, layer_norm_type)
        self.attn1 = Attention1(dim, num_heads, bias)
        self.attn2 = Attention2(dim, num_heads, bias)
        self.attn3 = Attention3(dim, num_heads, bias)
        self.norm2 = LayerNorm2d(dim, layer_norm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor)

    def forward(self, xx: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = xx
        x_k, x_q, x_v = self.attn1(self.norm1(x))
        y_k, y_q, y_v = self.attn1(self.norm1(y))
        x = x + self.attn2(x, y_k, x_q, y_v)
        x = x + self.ffn(self.norm2(x))
        y = y + self.attn3(y, x_k, y_q, x_v)
        y = y + self.ffn(self.norm2(y))
        return x, y


class EncoderCrossMIMO(nn.Module):
    """Stack of :class:`TransformerBlock1` (same weights across scales when depth>1)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_expansion_factor: float,
        bias: bool,
        layer_norm_type: str,
        depth: int,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock1(dim, num_heads, ffn_expansion_factor, bias, layer_norm_type)
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        t: Tuple[torch.Tensor, torch.Tensor] = (x, y)
        for block in self.blocks:
            t = block(t)
        return t


# ---------------------------------------------------------------------------#
# Patch embed / down / up (PixelUnshuffle via torch.nn.PixelUnshuffle)
# ---------------------------------------------------------------------------#


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_ch: int, embed_dim: int, bias: bool = False) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Downsample(nn.Module):
    """Conv halve channels then ``nn.PixelUnshuffle(2)`` (matches upstream Downsample)."""

    def __init__(self, n_feat: int) -> None:
        super().__init__()
        self.body = nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.unshuffle = nn.PixelUnshuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unshuffle(self.body(x))


class Upsample(nn.Module):
    def __init__(self, n_feat: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


# ---------------------------------------------------------------------------#
# Optional FFT L1 (for loss parity with upstream train.py; not used in forward)
# ---------------------------------------------------------------------------#


def mimosst_fft_l1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean absolute difference in 2D FFT domain (device-neutral)."""
    fx = torch.fft.fft2(x)
    fy = torch.fft.fft2(y)
    return torch.mean(torch.abs(fx - fy))


# ---------------------------------------------------------------------------#
# Main model
# ---------------------------------------------------------------------------#


class MIMOSSTScale4(nn.Module):
    """Multi-scale HSI–MSI fusion following local MIMO-SST ``Net`` topology.

    ``dim`` (``hidden_channels``) controls embedding width; fusion conv widths follow
    ``3*dim``, ``5*dim``, ``4*dim``, ``2*dim`` as in the original 48-wide network.
    """

    def __init__(
        self,
        hsi_channels: int = 103,
        msi_channels: int = 3,
        scale: int = 4,
        hidden_channels: Optional[int] = None,
        base_channels: Optional[int] = None,
        num_heads: int = 1,
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        layer_norm_type: str = "WithBias",
        encoder_depth: int = 1,
        window_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        if hidden_channels is not None and base_channels is not None:
            raise ValueError("pass only one of hidden_channels or base_channels")
        dim = hidden_channels if hidden_channels is not None else base_channels
        if dim is None:
            dim = 48
        if scale < 2:
            raise ValueError("scale must be >= 2")
        if dim % num_heads != 0:
            raise ValueError("hidden/base channel width must be divisible by num_heads")
        self.hsi_channels = hsi_channels
        self.msi_channels = msi_channels
        self.scale = scale
        # Local upstream Net has no spatial windowing (full H×W attention). Kept for API symmetry.
        self.window_size = window_size

        self.lr_to_hr = nn.Upsample(scale_factor=float(scale), mode="bilinear", align_corners=False)

        self.conv_msi = nn.Conv2d(msi_channels, dim, kernel_size=3, padding=1, bias=bias)
        self.conv_hsi = nn.Conv2d(hsi_channels, dim, kernel_size=3, padding=1, bias=bias)

        self.encoder_cross = EncoderCrossMIMO(
            dim, num_heads, ffn_expansion_factor, bias, layer_norm_type, encoder_depth
        )

        inp_cat = 2 * dim + msi_channels + hsi_channels
        self.patch_embed = OverlapPatchEmbed(inp_cat, dim, bias=bias)

        self.down1_2 = Downsample(dim)
        self.fuse_enc12 = nn.Conv2d(3 * dim, 2 * dim, kernel_size=3, padding=1, bias=bias)

        self.down2_3 = Downsample(2 * dim)
        self.fuse_enc23 = nn.Conv2d(5 * dim, 4 * dim, kernel_size=3, padding=1, bias=bias)

        self.to_hsi_quarter = nn.Conv2d(4 * dim, hsi_channels, kernel_size=3, padding=1, bias=bias)
        self.up3_2 = Upsample(4 * dim)
        self.reduce_chan_level2 = nn.Conv2d(4 * dim, 2 * dim, kernel_size=1, bias=bias)

        self.up2_1 = Upsample(2 * dim)
        self.to_hsi_full = nn.Conv2d(2 * dim, hsi_channels, kernel_size=3, padding=1, bias=bias)
        self.out_relu = nn.ReLU(inplace=True)

    def forward(
        self, lr_hsi: torch.Tensor, hr_msi: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        if lr_hsi.dim() != 4 or hr_msi.dim() != 4:
            raise ValueError(f"expected NCHW, got lr_hsi={lr_hsi.shape} hr_msi={hr_msi.shape}")
        b, c_lr, h_lr, w_lr = lr_hsi.shape
        _b2, c_hr, h_hr, w_hr = hr_msi.shape
        if b != _b2:
            raise ValueError(f"batch mismatch lr B={b} hr B={_b2}")
        if c_lr != self.hsi_channels or c_hr != self.msi_channels:
            raise ValueError(
                f"channel mismatch: expected lr C={self.hsi_channels}, hr C={self.msi_channels}; "
                f"got {c_lr}, {c_hr}"
            )
        if h_hr != h_lr * self.scale or w_hr != w_lr * self.scale:
            raise ValueError(
                f"spatial scale mismatch: hr {(h_hr, w_hr)} vs lr {(h_lr, w_lr)} with scale={self.scale}"
            )

        x = hr_msi
        y = self.lr_to_hr(lr_hsi)

        X = self.conv_msi(x)
        Y = self.conv_hsi(y)
        X, Y = self.encoder_cross(X, Y)
        z_full = torch.cat([X, Y, x, y], dim=1)

        x1 = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
        y1 = F.interpolate(y, scale_factor=0.5, mode="bilinear", align_corners=False)
        X1 = self.conv_msi(x1)
        Y1 = self.conv_hsi(y1)
        X1, Y1 = self.encoder_cross(X1, Y1)
        z_half = torch.cat([X1, Y1, x1, y1], dim=1)

        x2 = F.interpolate(x1, scale_factor=0.5, mode="bilinear", align_corners=False)
        y2 = F.interpolate(y1, scale_factor=0.5, mode="bilinear", align_corners=False)
        X2 = self.conv_msi(x2)
        Y2 = self.conv_hsi(y2)
        X2, Y2 = self.encoder_cross(X2, Y2)
        z_quarter = torch.cat([X2, Y2, x2, y2], dim=1)

        inp_enc_level1 = self.patch_embed(z_full)
        inp_enc_level11 = self.patch_embed(z_half)
        inp_enc_level12 = self.patch_embed(z_quarter)

        inp_enc_level2 = self.down1_2(inp_enc_level1)
        zz1 = self.fuse_enc12(torch.cat([inp_enc_level2, inp_enc_level11], dim=1))

        inp_enc_level3 = self.down2_3(zz1)
        zz2 = self.fuse_enc23(torch.cat([inp_enc_level3, inp_enc_level12], dim=1))

        pred_quarter = self.to_hsi_quarter(zz2) + y2

        inp_dec_level2 = self.reduce_chan_level2(
            torch.cat([self.up3_2(zz2), zz1], dim=1)
        )
        pred_half = self.to_hsi_full(inp_dec_level2) + y1

        inp_dec_level1 = torch.cat([self.up2_1(inp_dec_level2), inp_enc_level1], dim=1)
        pred = self.out_relu(self.to_hsi_full(inp_dec_level1) + y)

        return {
            "pred_quarter": pred_quarter,
            "pred_half": pred_half,
            "pred": pred,
        }


MIMOSSTScale4Out = Dict[str, torch.Tensor]


def mimosst_final_pred(out: Union[MIMOSSTScale4Out, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
    """Helper for runners: dict ``['pred']`` or tuple last element."""
    if isinstance(out, dict):
        return out["pred"]
    if isinstance(out, tuple):
        return out[-1]
    raise TypeError(f"expected dict or tuple, got {type(out)}")


def _shape_bracket_str(t: torch.Tensor) -> str:
    return "[" + ",".join(str(int(x)) for x in t.shape) + "]"


def run_smoke_forward(*, verbose: bool = False) -> int:
    """B=2 patch smoke: LR ``[2,103,32,32]``, HR ``[2,3,128,128]``, pred ``[2,103,128,128]``.

    When ``verbose`` is True, prints the three shape lines to stdout. Returns **0** if shapes match, **1** on failure.
    Call this (or ``python .../pytorch_mimosst_scale4.py``) before full baseline training.
    """
    try:
        lr_hsi = torch.randn(2, 103, 32, 32)
        hr_msi = torch.randn(2, 3, 128, 128)
        if verbose:
            print(f"input lr_hsi: {_shape_bracket_str(lr_hsi)}")
            print(f"input hr_msi: {_shape_bracket_str(hr_msi)}")
        model = MIMOSSTScale4(hsi_channels=103, msi_channels=3, scale=4)
        out = model(lr_hsi, hr_msi)
        pred = mimosst_final_pred(out)
        if verbose:
            print(f"output pred: {_shape_bracket_str(pred)}")
        if tuple(pred.shape) != (2, 103, 128, 128):
            print(
                f"run_smoke_forward: expected pred (2, 103, 128, 128), got {tuple(pred.shape)}",
                file=sys.stderr,
            )
            return 1
    except Exception as e:
        print(f"run_smoke_forward failed: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(run_smoke_forward(verbose=True))
