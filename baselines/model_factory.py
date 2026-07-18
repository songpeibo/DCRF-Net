"""Lightweight baseline model factories (model code only; no training scripts)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch.nn as nn

BASELINES_ROOT = Path(__file__).resolve().parent


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_cnmf():
  path = BASELINES_ROOT / "cnmf" / "cnmf_fusion.py"
  return _load_module(path, "baseline_cnmf")


def build_mctnet(*, n_bands: int = 103, n_select_bands: int = 3, scale_ratio: int = 4) -> nn.Module:
  root = BASELINES_ROOT / "mctnet"
  p = str(root)
  if p not in sys.path:
    sys.path.insert(0, p)
  from models.MCT import MCT  # noqa: WPS433

  return MCT(
    arch="MCT",
    scale_ratio=scale_ratio,
    n_select_bands=n_select_bands,
    n_bands=n_bands,
    dataset="PaviaU",
  )


def build_mhfnet(*, hsi_channels: int = 103, msi_channels: int = 3, scale: int = 4) -> nn.Module:
  mod = _load_module(BASELINES_ROOT / "mhfnet" / "model.py", "baseline_mhfnet")
  return mod.MHFNetFaithful(hsi_channels=hsi_channels, msi_channels=msi_channels, scale=scale)


def build_mimosst(*, hsi_channels: int = 103, msi_channels: int = 3, scale: int = 4) -> nn.Module:
  mod = _load_module(BASELINES_ROOT / "mimosst" / "model.py", "baseline_mimosst")
  return mod.MIMOSSTScale4(hsi_channels=hsi_channels, msi_channels=msi_channels, scale=scale)


def build_smf2net(*, hsi_channels: int = 103, msi_channels: int = 3, scale: int = 4) -> nn.Module:
  mod = _load_module(BASELINES_ROOT / "smf2net" / "model.py", "baseline_smf2net")
  return mod.SMF2NetScale4(hsi_channels=hsi_channels, msi_channels=msi_channels, scale=scale)


def build_smgunet(*, hsi_channels: int = 103, msi_channels: int = 3, niter: int = 2) -> nn.Module:
  mod = _load_module(BASELINES_ROOT / "smgunet" / "model.py", "baseline_smgunet")
  return mod.Net(niter=niter, hsi_channels=hsi_channels, msi_channels=msi_channels)


def build_ssrnet(*, n_bands: int = 103, n_select_bands: int = 3, scale_ratio: int = 4) -> nn.Module:
  mod = _load_module(BASELINES_ROOT / "ssrnet" / "model.py", "baseline_ssrnet")
  return mod.SSRNET(arch="SSRNET", scale_ratio=scale_ratio, n_select_bands=n_select_bands, n_bands=n_bands)


def build_tfnet(*, n_bands: int = 103, n_select_bands: int = 3, scale_ratio: int = 4) -> nn.Module:
  mod = _load_module(BASELINES_ROOT / "tfnet" / "model.py", "baseline_tfnet")
  return mod.ResTFNet(scale_ratio=scale_ratio, n_select_bands=n_select_bands, n_bands=n_bands)


BUILDERS = {
  "cnmf": build_cnmf,
  "mctnet": build_mctnet,
  "mhfnet": build_mhfnet,
  "mimosst": build_mimosst,
  "smf2net": build_smf2net,
  "smgunet": build_smgunet,
  "ssrnet": build_ssrnet,
  "tfnet": build_tfnet,
}
