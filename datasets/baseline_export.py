"""Load train patches from paper3 ``data/baseline_export/{dataset}/``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

EvalCropPolicy = Literal["center", "top_left"]
EVAL_CROP_CENTER: EvalCropPolicy = "center"
EVAL_CROP_TOP_LEFT: EvalCropPolicy = "top_left"
_DEFAULT_EVAL_CROP_POLICY: EvalCropPolicy = EVAL_CROP_CENTER


def normalize_eval_crop_policy(
    policy: Optional[str],
    *,
    default: EvalCropPolicy = _DEFAULT_EVAL_CROP_POLICY,
) -> EvalCropPolicy:
    """Normalize ``eval_crop_policy`` from config or meta (default ``center`` for legacy runs)."""
    if policy is None or str(policy).strip() == "":
        return default
    key = str(policy).strip().lower().replace("-", "_")
    aliases = {
        "center": EVAL_CROP_CENTER,
        "center_crop": EVAL_CROP_CENTER,
        "center_crop_div4": EVAL_CROP_CENTER,
        "top_left": EVAL_CROP_TOP_LEFT,
        "topleft": EVAL_CROP_TOP_LEFT,
        "top_left_crop": EVAL_CROP_TOP_LEFT,
        "top_left_crop_div4": EVAL_CROP_TOP_LEFT,
    }
    if key not in aliases:
        raise ValueError(
            f"eval_crop_policy must be 'center' or 'top_left', got {policy!r}"
        )
    return aliases[key]


def _center_crop_div4(x: torch.Tensor, scale: int = 4) -> torch.Tensor:
    """Center-crop CHW tensor so H and W are divisible by ``scale``."""
    if x.ndim != 3:
        raise ValueError(f"expected CHW tensor, got {tuple(x.shape)}")
    _, h, w = x.shape
    h2 = (h // scale) * scale
    w2 = (w // scale) * scale
    if h2 < scale or w2 < scale:
        raise ValueError(f"spatial size too small after crop: {(h, w)} scale={scale}")
    if h2 == h and w2 == w:
        return x
    off_h = (h - h2) // 2
    off_w = (w - w2) // 2
    return x[:, off_h : off_h + h2, off_w : off_w + w2]


def _top_left_crop_div4(x: torch.Tensor, scale: int = 4) -> torch.Tensor:
    """Top-left crop CHW tensor so H and W are divisible by ``scale``."""
    if x.ndim != 3:
        raise ValueError(f"expected CHW tensor, got {tuple(x.shape)}")
    _, h, w = x.shape
    h2 = (h // scale) * scale
    w2 = (w // scale) * scale
    if h2 < scale or w2 < scale:
        raise ValueError(f"spatial size too small after crop: {(h, w)} scale={scale}")
    return x[:, :h2, :w2]


def _align_scene_crops_center(
    z: torch.Tensor,
    h: torch.Tensor,
    m: torch.Tensor,
    scale: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Center-crop HR tensors (z, m) to H,W%scale==0; crop LR ``h`` to matching size."""
    z = _center_crop_div4(z, scale)
    m = _center_crop_div4(m, scale)
    lr_h = z.shape[-2] // scale
    lr_w = z.shape[-1] // scale
    if h.shape[-2] == lr_h and h.shape[-1] == lr_w:
        return z, h, m
    _, hh, hw = h.shape
    off_h = max(0, (hh - lr_h) // 2)
    off_w = max(0, (hw - lr_w) // 2)
    h = h[:, off_h : off_h + lr_h, off_w : off_w + lr_w]
    return z, h, m


def _align_scene_crops_top_left(
    z: torch.Tensor,
    h: torch.Tensor,
    m: torch.Tensor,
    scale: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Top-left crop HR (z, m) to H,W%scale==0; crop LR ``h`` to matching size."""
    z = _top_left_crop_div4(z, scale)
    m = _top_left_crop_div4(m, scale)
    lr_h = z.shape[-2] // scale
    lr_w = z.shape[-1] // scale
    if h.shape[-2] == lr_h and h.shape[-1] == lr_w:
        return z, h, m
    return z, h[:, :lr_h, :lr_w], m


def align_scene_for_eval(
    z: torch.Tensor,
    h: torch.Tensor,
    m: torch.Tensor,
    scale: int,
    policy: EvalCropPolicy | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align val/test scene tensors for evaluation (Table III or legacy center-crop val)."""
    pol = normalize_eval_crop_policy(str(policy))
    if pol == EVAL_CROP_TOP_LEFT:
        return _align_scene_crops_top_left(z, h, m, scale)
    return _align_scene_crops_center(z, h, m, scale)


def eval_crop_policy_description(policy: EvalCropPolicy | str) -> str:
    pol = normalize_eval_crop_policy(str(policy))
    if pol == EVAL_CROP_TOP_LEFT:
        return "top_left_crop_div4 (H,W divisible by scale; Table III common-eval)"
    return "center_crop_div4 (H,W divisible by scale; legacy validation)"


# Backward-compatible alias used internally before policy split.
_align_scene_crops = _align_scene_crops_center


def _load_meta(root: Path) -> Dict[str, Any]:
    meta_path = root / "meta.json"
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


class BaselineExportDataset(Dataset):
    """NPZ train patches exported by ``tools/generate_baseline_export.py``."""

    def __init__(
        self,
        root: str | Path,
        *,
        split: str = "train",
        max_patches: Optional[int] = None,
        eval_crop_policy: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root).resolve()
        self.split = str(split)
        if not self.root.is_dir():
            raise FileNotFoundError(f"baseline_export root not found: {self.root}")
        self.meta = _load_meta(self.root)
        self.scale = int(self.meta.get("scale", 4))
        self.eval_crop_policy = normalize_eval_crop_policy(
            eval_crop_policy if eval_crop_policy is not None else self.meta.get("eval_crop_policy"),
            default=_DEFAULT_EVAL_CROP_POLICY,
        )
        patch_dir = self.root / self.split
        if not patch_dir.is_dir():
            raise FileNotFoundError(f"split directory not found: {patch_dir}")
        self.paths: List[Path] = sorted(patch_dir.glob("*.npz"))
        if max_patches is not None:
            self.paths = self.paths[: int(max_patches)]
        if not self.paths:
            raise FileNotFoundError(f"no npz patches under {patch_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = np.load(self.paths[idx])
        z = torch.from_numpy(data["Z_gt"]).float()
        h = torch.from_numpy(data["H"]).float()
        m = torch.from_numpy(data["M"]).float()
        phi0 = torch.from_numpy(data["Phi0"]).float()
        k0 = torch.from_numpy(data["k0"]).float()
        if k0.ndim == 3 and k0.shape[0] == 1:
            k0 = k0.unsqueeze(0)  # [1, 1, K, K]
        if z.ndim == 3:
            z = z.unsqueeze(0)
        if h.ndim == 3:
            h = h.unsqueeze(0)
        if m.ndim == 3:
            m = m.unsqueeze(0)
        z = z.squeeze(0)
        h = h.squeeze(0)
        m = m.squeeze(0)
        if self.split in ("val", "test"):
            z, h, m = align_scene_for_eval(z, h, m, self.scale, self.eval_crop_policy)
        return {
            "gt": z,
            "h": h,
            "m": m,
            "phi0": phi0,
            "k0": k0,
        }
