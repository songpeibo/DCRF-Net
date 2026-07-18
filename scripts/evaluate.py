#!/usr/bin/env python3
"""Evaluate a trained DCRF-Net checkpoint on baseline_export val/test split."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.baseline_export import (  # noqa: E402
    BaselineExportDataset,
    eval_crop_policy_description,
    normalize_eval_crop_policy,
)
from models.build_model import build_model_from_config, load_checkpoint_payload  # noqa: E402
from utils.common import resolve_path  # noqa: E402
from utils.metrics import evaluate_reconstruction  # noqa: E402


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in batch[0]:
        vals = [b[k] for b in batch]
        if k in ("phi0", "k0") and all(torch.equal(vals[0], v) for v in vals[1:]):
            out[k] = vals[0]
        elif isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals, dim=0)
        else:
            out[k] = vals
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="YAML config (formal dcrf_net.yaml)")
    ap.add_argument("--checkpoint", required=True, help="Path to checkpoint .pth")
    ap.add_argument("--data-root", required=True, help="baseline_export dataset root")
    ap.add_argument("--split", choices=("val", "test"), default="test")
    ap.add_argument("--device", default=None, help="cuda | cpu")
    ap.add_argument("--out-dir", default=None, help="Optional output directory for pred.npy")
    ap.add_argument("--timing", action="store_true", help="Report average forward time (seconds)")
    args = ap.parse_args()

    cfg_path = resolve_path(ROOT, args.config)
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("data", {})["root"] = args.data_root

    dcfg = cfg.get("data", {})
    eval_crop_policy = normalize_eval_crop_policy(dcfg.get("eval_crop_policy"))
    data_root = resolve_path(ROOT, args.data_root)
    device_str = args.device or str(cfg.get("device", "cuda"))
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)
    scale = int(dcfg.get("scale", 4))

    ds = BaselineExportDataset(data_root, split=args.split, eval_crop_policy=eval_crop_policy)
    sample = ds[0]
    batch = _collate([sample])
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    model = build_model_from_config(cfg, device)
    ckpt_path = resolve_path(ROOT, args.checkpoint)
    epoch, ckpt_metrics, _ = load_checkpoint_payload(model, ckpt_path, device)
    model.eval()

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        pred_out = model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    pred = pred_out["z_hat"].detach().cpu()
    gt = sample["gt"].unsqueeze(0)
    meta_path = data_root / "meta.json"
    data_range = 1.0
    if meta_path.is_file():
        with meta_path.open(encoding="utf-8") as f:
            meta = json.load(f)
        data_range = float(meta.get("metric_data_range", meta.get("metric_range", 1.0)))

    metrics = evaluate_reconstruction(pred, gt, scale=scale, max_val=data_range)
    print(f"split={args.split} epoch={epoch} crop={eval_crop_policy_description(eval_crop_policy)}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")
    if args.timing:
        print(f"  forward_time_sec: {elapsed:.4f}")

    if args.out_dir:
        out_dir = resolve_path(ROOT, args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(out_dir / "pred.npy"), pred.numpy().astype(np.float32))
        meta_doc = {
            "checkpoint_epoch": epoch,
            "checkpoint_metrics": ckpt_metrics,
            "metrics": metrics,
            "forward_time_sec": elapsed,
            "eval_crop_policy": eval_crop_policy,
        }
        (out_dir / "pred_meta.json").write_text(json.dumps(meta_doc, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
