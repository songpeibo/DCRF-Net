#!/usr/bin/env python3
"""Run DCRF-Net inference on one baseline_export scene NPZ."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.build_model import build_model_from_config, load_checkpoint_payload  # noqa: E402
from utils.common import resolve_path  # noqa: E402


def _load_npz_scene(path: Path) -> dict[str, torch.Tensor]:
    import numpy as _np

    with _np.load(str(path), allow_pickle=False) as z:
        return {k: torch.from_numpy(z[k].astype(_np.float32)) for k in z.files}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--scene-npz", required=True, help="Path to scene_00.npz")
    ap.add_argument("--out", required=True, help="Output pred.npy path")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg_path = resolve_path(ROOT, args.config)
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device_str = args.device or str(cfg.get("device", "cuda"))
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    scene = _load_npz_scene(resolve_path(ROOT, args.scene_npz))
    batch = {
        "h": scene["h"].unsqueeze(0).to(device),
        "m": scene["m"].unsqueeze(0).to(device),
        "phi0": scene["phi0"].to(device),
        "k0": scene["k0"].to(device),
    }
    if "gt" in scene:
        batch["gt"] = scene["gt"].unsqueeze(0).to(device)

    model = build_model_from_config(cfg, device)
    load_checkpoint_payload(model, resolve_path(ROOT, args.checkpoint), device)
    model.eval()
    with torch.inference_mode():
        pred = model(batch)["z_hat"].detach().cpu().numpy().astype(np.float32)
    out_path = resolve_path(ROOT, args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), pred)
    print(f"saved {out_path} shape={pred.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
