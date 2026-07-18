#!/usr/bin/env python3
"""Train DCSR-Net from a YAML config."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.baseline_export import (
    BaselineExportDataset,
    eval_crop_policy_description,
    normalize_eval_crop_policy,
)
from datasets.synthetic import SyntheticHSIMSIFusionDataset
from models.build_model import (
    build_model_from_config,
    log_effective_config,
    model_variant_label,
    save_effective_config,
)
from models.dcsr_net import DCSRNet
from utils.common import resolve_path
from utils.logger import (
    TRAIN_VALID_METRICS_FIELDS,
    CSVLogger,
    CheckpointManager,
    collect_dcsr_diagnostics,
    count_parameters,
    create_run_dir,
    format_epoch_line,
    format_num_params,
    get_optimizer_lr,
    gpu_memory_gb,
    gpu_memory_reserved_gb,
    sam_rad_deg_pair,
    save_args_snapshot,
    save_config_snapshot,
)
from utils.losses import DCSRLoss
from utils.metrics import ergas, psnr, rmse, sam, ssim
from utils.seed import set_seed
from utils.train_logging import TrainingDualLogger


def _collate_fusion(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        if k in ("phi0", "k0") and all(torch.equal(vals[0], v) for v in vals[1:]):
            out[k] = vals[0]
        elif isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals, dim=0)
        else:
            out[k] = vals
    return out


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _cfg_get(cfg: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in cfg:
            return cfg[k]
    return default


def _na(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float) and not math.isfinite(value):
        return "N/A"
    return str(value)


def _fmt_csv(x: float) -> str:
    if not math.isfinite(x):
        return "NaN"
    return f"{x:.8f}"


def _validation_eval_shape(val_set: Any) -> str:
    if not isinstance(val_set, BaselineExportDataset) or len(val_set) < 1:
        return "N/A"
    sample = val_set[0]
    gt = sample["gt"]
    if not isinstance(gt, torch.Tensor) or gt.ndim != 3:
        return "N/A"
    c, h, w = int(gt.shape[0]), int(gt.shape[1]), int(gt.shape[2])
    return f"{c}x{h}x{w}"


def _build_datasets(cfg: dict[str, Any], root: Path) -> tuple[Any, Any, str, str, str]:
    dcfg = cfg.get("data", {})
    scale = int(_cfg_get(cfg, "scale", default=dcfg.get("scale", 4)))
    hsi_c = int(_cfg_get(cfg, "hsi_channels", default=dcfg.get("hsi_channels", 103)))
    msi_c = int(_cfg_get(cfg, "msi_channels", default=dcfg.get("msi_channels", 4)))
    patch = int(_cfg_get(cfg, "patch_size", default=dcfg.get("patch_size", 32)))
    kernel = int(_cfg_get(cfg, "kernel_size", default=7))
    seed = int(_cfg_get(cfg, "seed", default=0))

    split_mode = str(dcfg.get("split_mode", "synthetic")).lower()
    if split_mode == "smooth_operator_v1":
        protocol_root = resolve_path(
            root, dcfg.get("root", "data/protocol_reboot/paviau/smooth_operator_v1")
        )
        eval_crop_policy = normalize_eval_crop_policy(dcfg.get("eval_crop_policy"))
        train_set = SmoothOperatorTrainDataset(protocol_root)
        val_set = SmoothOperatorValDataset(
            protocol_root, "clean", eval_crop_policy=eval_crop_policy
        )
        return train_set, val_set, split_mode, str(protocol_root), eval_crop_policy

    if split_mode == "mismatch_mixed":
        clean_root = resolve_path(root, dcfg.get("clean_root", dcfg.get("root", "data/baseline_export/paviau")))
        eval_crop_policy = normalize_eval_crop_policy(dcfg.get("eval_crop_policy"))
        train_set = MismatchMixedTrainDataset(clean_root)
        val_set = BaselineExportDataset(clean_root, split="val", eval_crop_policy=eval_crop_policy)
        return train_set, val_set, split_mode, str(clean_root), eval_crop_policy

    if split_mode == "synthetic" or "train_samples" in cfg:
        n_train = int(_cfg_get(cfg, "train_samples", default=64))
        n_val = int(_cfg_get(cfg, "val_samples", default=8))
        train_set = SyntheticHSIMSIFusionDataset(
            num_samples=n_train,
            hsi_channels=hsi_c,
            msi_channels=msi_c,
            patch_size=patch,
            scale=scale,
            kernel_size=kernel,
            seed=seed,
            semi_blind=True,
        )
        val_set = SyntheticHSIMSIFusionDataset(
            num_samples=n_val,
            hsi_channels=hsi_c,
            msi_channels=msi_c,
            patch_size=patch,
            scale=scale,
            kernel_size=kernel,
            seed=seed + 10_000,
            semi_blind=True,
        )
        return train_set, val_set, "synthetic", "synthetic (on-the-fly)", "N/A"

    data_root = resolve_path(root, dcfg.get("root", "data/baseline_export/paviau"))
    split_mode = str(dcfg.get("split_mode", "baseline_export"))
    eval_crop_policy = normalize_eval_crop_policy(dcfg.get("eval_crop_policy"))
    train_set = BaselineExportDataset(data_root, split="train", eval_crop_policy=eval_crop_policy)
    val_set = BaselineExportDataset(data_root, split="val", eval_crop_policy=eval_crop_policy)
    return train_set, val_set, split_mode, str(data_root), eval_crop_policy


def _is_b10_config(cfg: dict[str, Any]) -> bool:
    return str(cfg.get("model", "DCSRNet")) == "DCSROperatorHypothesisNet"


def _build_loss(cfg: dict[str, Any], scale: int) -> nn.Module:
    if _is_b10_config(cfg):
        from utils.b10_losses import B10OperatorHypothesisLoss

        lcfg = cfg.get("loss", cfg)
        return B10OperatorHypothesisLoss(
            scale=scale,
            lambda_rec=float(_cfg_get(lcfg, "lambda_rec", default=1.0)),
            lambda_sam=float(_cfg_get(lcfg, "lambda_sam", default=0.05)),
            lambda_init=float(_cfg_get(lcfg, "lambda_init", default=0.2)),
            lambda_obs=float(_cfg_get(lcfg, "lambda_obs", default=0.1)),
            lambda_diag=float(_cfg_get(lcfg, "lambda_diag", default=0.1)),
        )
    lcfg = cfg.get("loss", cfg)
    return DCSRLoss(
        scale=scale,
        lambda_rec=float(_cfg_get(lcfg, "lambda_rec", default=1.0)),
        lambda_rec_coarse=float(_cfg_get(lcfg, "lambda_rec_coarse", default=0.0)),
        lambda_obs=float(_cfg_get(lcfg, "lambda_obs", default=0.2)),
        lambda_obs_coarse=float(_cfg_get(lcfg, "lambda_obs_coarse", default=0.1)),
        lambda_sam=float(_cfg_get(lcfg, "lambda_sam", default=0.05)),
        lambda_deg=float(_cfg_get(lcfg, "lambda_deg", default=0.001)),
        lambda_sel=float(_cfg_get(lcfg, "lambda_sel", default=0.0001)),
        lambda_feedback_intermediate=float(
            _cfg_get(lcfg, "lambda_feedback_intermediate", default=0.0)
        ),
    )


def _build_config_rows(
    *,
    cfg: dict[str, Any],
    model: nn.Module,
    train_set: Any,
    val_set: Any,
    split_mode: str,
    data_root: str,
    run_dir: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    num_workers: int,
    lr: float,
    weight_decay: float,
    amp: bool,
    eval_crop_policy: str,
    val_eval_shape: str,
) -> list[tuple[str, str]]:
    dcfg = cfg.get("data", {})
    marg = cfg.get("model_args", {})
    pcounts = count_parameters(model)
    dataset_name = _na(dcfg.get("dataset", dcfg.get("name", "synthetic" if split_mode == "synthetic" else None)))

    rows: list[tuple[str, str]] = [
        ("task", _na(cfg.get("task"))),
        ("setting", _na(cfg.get("setting"))),
        ("model", _na(cfg.get("model", "DCSRNet"))),
        ("dataset", dataset_name),
        ("data_root", data_root),
        ("save_dir", str(run_dir)),
        ("split_mode", split_mode),
        ("eval_crop_policy", eval_crop_policy),
        ("validation_eval_shape", val_eval_shape),
        ("epochs", str(epochs)),
        ("batch_size", str(batch_size)),
        ("num_workers", str(num_workers)),
        ("lr", f"{lr:.2e}"),
        ("weight_decay", f"{weight_decay:.2e}"),
        ("amp", str(amp)),
        ("device", str(device)),
        ("hsi_channels", _na(_cfg_get(cfg, "hsi_channels", default=dcfg.get("hsi_channels")))),
        ("msi_channels", _na(_cfg_get(cfg, "msi_channels", default=dcfg.get("msi_channels")))),
        ("scale", _na(_cfg_get(cfg, "scale", default=dcfg.get("scale")))),
        ("patch_size", _na(_cfg_get(cfg, "patch_size", default=dcfg.get("patch_size")))),
        ("num_train_patches", _na(dcfg.get("num_train_patches"))),
        ("train_samples", _na(_cfg_get(cfg, "train_samples", default=len(train_set)))),
        ("val_samples", _na(_cfg_get(cfg, "val_samples", default=len(val_set)))),
        ("kernel_size", _na(marg.get("kernel_size", _cfg_get(cfg, "kernel_size")))),
        ("base_channels", _na(marg.get("base_channels", _cfg_get(cfg, "base_channels")))),
        (
            "delta_phi_scale",
            _na(
                marg.get(
                    "delta_phi_scale",
                    _cfg_get(cfg, "delta_phi_scale", default=getattr(model, "delta_phi_scale", None)),
                )
            ),
        ),
        ("delta_k_scale", _na(marg.get("delta_k_scale", getattr(model, "delta_k_scale", None)))),
        ("guidance_channels", _na(marg.get("guidance_channels"))),
        ("use_signed_residual_guidance", _na(marg.get("use_signed_residual_guidance"))),
        ("use_msi_detail_guidance", _na(marg.get("use_msi_detail_guidance"))),
        ("params_total", f"{pcounts['total']} ({format_num_params(pcounts['total'])})"),
        ("params_trainable", f"{pcounts['trainable']} ({format_num_params(pcounts['trainable'])})"),
        ("params_frozen", f"{pcounts['frozen']} ({format_num_params(pcounts['frozen'])})"),
    ]
    return rows


def _apply_resume(
    resume_path: Path,
    *,
    run_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.StepLR,
    device: torch.device,
    dual: TrainingDualLogger,
    ckpt_mgr: CheckpointManager,
    scaler: Any | None = None,
) -> tuple[int, int, bool, bool, bool]:
    """Load checkpoint; return (start_epoch, loaded_epoch, opt_ok, sched_ok, scaler_ok)."""
    if not resume_path.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")

    ckpt_dir = resume_path.resolve().parent
    if ckpt_dir != run_dir.resolve():
        dual.log_both_plain(
            f"WARNING: resume checkpoint dir {ckpt_dir} != config save_dir {run_dir.resolve()} "
            "(using config save_dir for outputs)"
        )

    payload = torch.load(str(resume_path), map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"resume checkpoint must be a dict with 'model' key: {resume_path}")

    model.load_state_dict(payload["model"], strict=True)

    opt_ok = False
    if isinstance(payload.get("optimizer"), dict):
        optimizer.load_state_dict(payload["optimizer"])
        opt_ok = True

    sched_ok = False
    if isinstance(payload.get("scheduler"), dict):
        scheduler.load_state_dict(payload["scheduler"])
        sched_ok = True

    scaler_ok = False
    if scaler is not None and isinstance(payload.get("scaler"), dict):
        scaler.load_state_dict(payload["scaler"])
        scaler_ok = True

    loaded_epoch = int(payload.get("epoch", -1))
    if loaded_epoch < 0:
        raise ValueError(f"resume checkpoint missing valid 'epoch': {resume_path}")
    start_epoch = loaded_epoch + 1

    meta_ok = ckpt_mgr.restore_from_meta()
    if not meta_ok:
        dual.log_both_plain(
            "WARNING: could not restore best-metric tracker from robust_ckpt_selection_meta.json; "
            "existing best_psnr.pth / best_sam.pth / best_ergas.pth are kept unchanged"
        )

    dual.log_both_plain(f"[Resume] path={resume_path}")
    dual.log_both_plain(f"[Resume] loaded_epoch={loaded_epoch}")
    dual.log_both_plain(f"[Resume] start_epoch={start_epoch}")
    dual.log_both_plain(f"[Resume] optimizer_restored={opt_ok}")
    dual.log_both_plain(f"[Resume] scheduler_restored={sched_ok}")
    dual.log_both_plain(f"[Resume] scaler_restored={scaler_ok}")
    if meta_ok:
        dual.log_both_plain(
            f"[Resume] best_tracker_restored=True "
            f"(psnr={ckpt_mgr.best_psnr:.4f}@{ckpt_mgr.best_psnr_epoch:03d})"
        )
    else:
        dual.log_both_plain("[Resume] best_tracker_restored=False")

    return start_epoch, loaded_epoch, opt_ok, sched_ok, scaler_ok


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: DCSRLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    epoch: int,
    epochs: int,
) -> tuple[float, dict[str, float]]:
    model.train()
    total_loss = 0.0
    n = 0
    comp_sums: dict[str, float] = {}
    lr_now = get_optimizer_lr(optimizer)

    pbar = tqdm(
        loader,
        desc=f"[Train E{epoch:03d}/{epochs:03d}]",
        leave=False,
        dynamic_ncols=True,
        unit="batch",
    )
    for batch in pbar:
        batch = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch)
        loss_dict = criterion(pred, batch)
        loss = loss_dict["loss"]
        if not isinstance(loss, torch.Tensor):
            raise TypeError("DCSRLoss must return tensor 'loss'")
        loss.backward()
        optimizer.step()

        bs = batch["h"].shape[0]
        lv = float(loss.item())
        total_loss += lv * bs
        n += bs
        for k in (
            "loss_rec",
            "loss_rec_coarse",
            "loss_init",
            "loss_obs",
            "loss_obs_coarse",
            "loss_sam",
            "loss_feedback_intermediate",
            "loss_diag",
            "loss_deg",
            "loss_sel",
        ):
            if k in loss_dict:
                comp_sums[k] = comp_sums.get(k, 0.0) + float(loss_dict[k]) * bs

        postfix: dict[str, str] = {
            "loss": f"{lv:.2e}",
            "lr": f"{lr_now:.1e}",
            "mem": f"{gpu_memory_gb():.2f}G",
        }
        if "loss_rec" in loss_dict:
            postfix["rec"] = f"{float(loss_dict['loss_rec']):.2e}"
        if "loss_obs" in loss_dict:
            postfix["obs"] = f"{float(loss_dict['loss_obs']):.2e}"
        pbar.set_postfix(postfix, refresh=True)

    inv = max(1, n)
    return total_loss / inv, {k: v / inv for k, v in comp_sums.items()}


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: DCSRLoss,
    device: torch.device,
    scale: int,
    *,
    epoch: int,
    epochs: int,
) -> tuple[float, dict[str, float], dict[str, torch.Tensor] | None, dict[str, Any] | None]:
    model.eval()
    total_loss = 0.0
    n = 0
    psnr_sum = psnr_coarse_sum = sam_sum = rmse_sum = ergas_sum = ssim_sum = 0.0
    m_count = 0
    last_pred: dict[str, torch.Tensor] | None = None
    last_batch: dict[str, Any] | None = None

    pbar = tqdm(
        loader,
        desc=f"[Val   E{epoch:03d}/{epochs:03d}]",
        leave=False,
        dynamic_ncols=True,
        unit="batch",
    )
    for batch in pbar:
        batch = _to_device(batch, device)
        pred = model(batch)
        loss_dict = criterion(pred, batch)
        loss = loss_dict["loss"]
        bs = batch["h"].shape[0]
        total_loss += float(loss.item()) * bs
        n += bs
        if "gt" in batch:
            z = pred["z_hat"]
            gt = batch["gt"]
            psnr_sum += psnr(z, gt)
            if "z_coarse" in pred:
                psnr_coarse_sum += psnr(pred["z_coarse"], gt)
            elif "z0" in pred:
                psnr_coarse_sum += psnr(pred["z0"], gt)
            sam_sum += sam(z, gt)
            rmse_sum += rmse(z, gt)
            ergas_sum += ergas(z, gt, scale=scale)
            ssim_sum += ssim(z, gt)
            m_count += 1
        last_pred = pred
        last_batch = batch

        nb = max(1, m_count)
        postfix = {
            "PSNR": f"{psnr_sum / nb:.2f}",
            "SAM": f"{sam_sum / nb:.2f}",
            "RMSE": f"{rmse_sum / nb:.4f}",
            "ERGAS": f"{ergas_sum / nb:.2f}",
        }
        if m_count > 0:
            postfix["SSIM"] = f"{ssim_sum / nb:.4f}"
        pbar.set_postfix(postfix, refresh=True)

    vp = psnr_sum / max(1, m_count)
    vpc = psnr_coarse_sum / max(1, m_count)
    metrics = {
        "val_psnr": vp,
        "val_psnr_coarse": vpc,
        "psnr_gap": vp - vpc if m_count > 0 else float("nan"),
        "val_sam": sam_sum / max(1, m_count),
        "val_rmse": rmse_sum / max(1, m_count),
        "val_ergas": ergas_sum / max(1, m_count),
        "val_ssim": ssim_sum / max(1, m_count),
        "has_gt": m_count > 0,
    }
    return total_loss / max(1, n), metrics, last_pred, last_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DCSR-Net")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--data-root", type=str, default=None, help="Override data.root in config")
    parser.add_argument("--device", type=str, default=None, help="cuda | cpu (default: auto)")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing run_dir contents")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint (.pth) to resume; continues from epoch+1 in the same save_dir",
    )
    args = parser.parse_args()

    cfg_path = resolve_path(ROOT, args.config)
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.data_root:
        cfg.setdefault("data", {})["root"] = args.data_root

    seed = int(_cfg_get(cfg, "seed", default=0))
    set_seed(seed)

    device_str = args.device or str(_cfg_get(cfg, "device", default="cuda"))
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    tcfg = cfg.get("train", cfg)
    dcfg = cfg.get("data", {})
    marg = cfg.get("model_args", {})
    scale = int(_cfg_get(cfg, "scale", default=dcfg.get("scale", marg.get("scale", 4))))
    hsi_c = int(_cfg_get(cfg, "hsi_channels", default=dcfg.get("hsi_channels", marg.get("hsi_channels", 103))))
    msi_c = int(_cfg_get(cfg, "msi_channels", default=dcfg.get("msi_channels", marg.get("msi_channels", 4))))
    kernel = int(_cfg_get(cfg, "kernel_size", default=marg.get("kernel_size", 7)))
    base_ch = int(_cfg_get(cfg, "base_channels", default=marg.get("base_channels", 64)))
    delta_phi = float(marg.get("delta_phi_scale", _cfg_get(cfg, "delta_phi_scale", default=0.05)))

    train_set, val_set, split_mode, data_root, eval_crop_policy = _build_datasets(cfg, ROOT)
    val_eval_shape = _validation_eval_shape(val_set)
    batch_size = int(tcfg.get("batch_size", _cfg_get(cfg, "batch_size", default=2)))
    num_workers = int(tcfg.get("num_workers", _cfg_get(cfg, "num_workers", default=0)))
    epochs = int(tcfg.get("epochs", _cfg_get(cfg, "epochs", default=5)))
    lr = float(tcfg.get("lr", _cfg_get(cfg, "lr", default=1e-4)))
    wd = float(tcfg.get("weight_decay", _cfg_get(cfg, "weight_decay", default=0.0)))
    save_dir = str(tcfg.get("save_dir", _cfg_get(cfg, "save_dir", default="outputs/debug_run")))
    amp = bool(tcfg.get("amp", _cfg_get(cfg, "amp", default=False)))
    resume_path: Path | None = None
    if args.resume:
        resume_path = resolve_path(ROOT, args.resume)
    overwrite = bool(args.overwrite) or bool(tcfg.get("overwrite", _cfg_get(cfg, "overwrite", default=False)))
    if resume_path is not None and overwrite:
        print("warning: --overwrite ignored when --resume is set")
        overwrite = False

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate_fusion,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(tcfg.get("val_batch_size", batch_size)),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_fusion,
    )

    if _is_b10_config(cfg) and bool(dcfg.get("protocol_validate", True)):
        from tools.validate_smooth_operator_protocol import run_validation

        proto_root = resolve_path(
            ROOT, dcfg.get("root", "data/protocol_reboot/paviau/smooth_operator_v1")
        )
        if not run_validation(proto_root, skip_export_check=False):
            raise SystemExit("smooth_operator_v1 protocol validation failed; training aborted")

    model = build_model_from_config(cfg, device, enforce_nonblind=not _is_b10_config(cfg))

    criterion = _build_loss(cfg, scale)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(tcfg.get("lr_step", 50)),
        gamma=float(tcfg.get("lr_gamma", 0.5)),
    )

    run_dir = create_run_dir(resolve_path(ROOT, save_dir), overwrite=overwrite)
    if resume_path is None:
        save_config_snapshot(cfg, run_dir)
    save_args_snapshot({"config": str(cfg_path), **vars(args)}, run_dir)

    dual = TrainingDualLogger(run_dir)
    dual.setup(append=resume_path is not None)

    metrics_logger = CSVLogger(run_dir / "train_valid_metrics.csv", fieldnames=TRAIN_VALID_METRICS_FIELDS)
    ckpt_mgr = CheckpointManager(run_dir, primary_metric="psnr", save_latest=True)

    start_epoch = 1
    if resume_path is not None:
        start_epoch, _, _, _, _ = _apply_resume(
            resume_path,
            run_dir=run_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            dual=dual,
            ckpt_mgr=ckpt_mgr,
            scaler=None,
        )
        dual.log_system_file_only("phase=training_main boundary=after_resume")
    else:
        dual.emit_config_box_top()
        for key, value in _build_config_rows(
            cfg=cfg,
            model=model,
            train_set=train_set,
            val_set=val_set,
            split_mode=split_mode,
            data_root=data_root,
            run_dir=run_dir,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            lr=lr,
            weight_decay=wd,
            amp=amp,
            eval_crop_policy=eval_crop_policy,
            val_eval_shape=val_eval_shape,
        ):
            dual.emit_config_line(key, value)
        dual.emit_config_box_bottom()
        if isinstance(model, DCSRNet):
            eff_snap = save_effective_config(run_dir, cfg, model)
            log_effective_config(cfg, model, emit=dual.log_plain_file)
        else:
            eff_snap = {"model_variant": model_variant_label(cfg), "model": str(cfg.get("model"))}
            dual.log_plain_file(f"model_variant: {eff_snap['model_variant']}")
        dual.log_plain_file(f"effective_config.json -> {run_dir / 'effective_config.json'}")
        if not eff_snap.get("strict_nonblind_scales_match", True):
            dual.log_plain_file(
                "WARNING: declared delta_phi_scale/delta_k_scale do not match effective model attributes"
            )
        dual.log_plain_file(f"eval_crop_policy={eval_crop_policy} ({eval_crop_policy_description(eval_crop_policy)})")
        dual.log_plain_file(f"validation_eval_shape={val_eval_shape}")
        if eval_crop_policy == "top_left":
            dual.log_plain_file("formal evaluation crop policy aligned with Table III: top_left")
        else:
            dual.log_plain_file(
                "formal evaluation crop policy: center (legacy; not aligned with Table III top_left common-eval)"
            )
        dual.log_system_file_only("phase=training_main boundary=after_config")

    mismatch_val_roots = dcfg.get("mismatch_val_roots")
    b10_eval_cfg = cfg.get("b10_eval", {}) or {}
    smoke_best_tracker: dict[str, Any] = {}
    b10_best_tracker: dict[str, Any] = {}
    metrics_by_setting_path = run_dir / "metrics_by_setting.csv"
    is_b10 = _is_b10_config(cfg)
    if is_b10:
        from utils.b10_protocol_eval import evaluate_all_b10_settings, step_b10_checkpoints, write_metrics_by_setting_csv
        from utils.smooth_operator_protocol import VAL_SETTINGS

        b10_settings = list(b10_eval_cfg.get("settings", VAL_SETTINGS))
        dual.log_plain_file(f"B10 protocol eval settings: {b10_settings}")
        dual.log_plain_file(
            "B10 checkpoints: best_clean_psnr, best_mean_heldout_psnr, "
            "best_worst_heldout_psnr, best_joint_heldout_psnr"
        )
    if mismatch_val_roots and isinstance(mismatch_val_roots, dict):
        from utils.b9_smoke_eval import (
            evaluate_all_settings,
            step_smoke_best_checkpoints,
            write_metrics_by_setting_csv,
        )

        dual.log_plain_file(f"B9 mismatch_val_roots: {list(mismatch_val_roots.keys())}")
        dual.log_plain_file(
            "B9 smoke checkpoints: best_clean_psnr.pth, best_mean_psnr.pth, "
            "best_worst_case_psnr.pth (best_psnr.pth = clean val_loader, compatibility only)"
        )

    t_wall0 = time.perf_counter()

    for epoch in range(start_epoch, epochs + 1):
        t_epoch0 = time.perf_counter()
        train_loss, train_comps = _train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch=epoch, epochs=epochs
        )
        val_loss, val_metrics, last_pred, last_batch = _evaluate(
            model, val_loader, criterion, device, scale, epoch=epoch, epochs=epochs
        )

        lr_now = get_optimizer_lr(optimizer)
        mem_alloc = gpu_memory_gb()
        mem_reserved = gpu_memory_reserved_gb()
        epoch_wall = time.perf_counter() - t_epoch0
        wall_cum = time.perf_counter() - t_wall0

        vp = float(val_metrics["val_psnr"])
        vs_deg = float(val_metrics["val_sam"])
        vr = float(val_metrics["val_rmse"])
        ve = float(val_metrics["val_ergas"])
        vssim = float(val_metrics["val_ssim"]) if val_metrics["has_gt"] else float("nan")
        sam_rad, sam_deg = sam_rad_deg_pair(vs_deg)

        diag_aux: dict[str, Any] = {}
        if last_pred is not None and last_batch is not None:
            diag_aux = collect_dcsr_diagnostics(last_pred, last_batch)

        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": copy.deepcopy(cfg),
            "metrics": {k: v for k, v in val_metrics.items() if k != "has_gt"},
        }
        updated = ckpt_mgr.step(epoch, payload, val_psnr=vp, val_sam_deg=sam_deg, val_ergas=ve)

        csv_row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": _fmt_csv(train_loss),
            "rec_loss": _fmt_csv(train_comps.get("loss_rec", float("nan"))),
            "rec_coarse_loss": _fmt_csv(train_comps.get("loss_rec_coarse", float("nan"))),
            "obs_loss": _fmt_csv(train_comps.get("loss_obs", float("nan"))),
            "obs_coarse_loss": _fmt_csv(train_comps.get("loss_obs_coarse", float("nan"))),
            "sam_loss": _fmt_csv(train_comps.get("loss_sam", float("nan"))),
            "deg_loss": _fmt_csv(train_comps.get("loss_deg", float("nan"))),
            "sel_loss": _fmt_csv(train_comps.get("loss_sel", float("nan"))),
            "PSNR": _fmt_csv(vp),
            "SSIM": _fmt_csv(vssim) if math.isfinite(vssim) else "NaN",
            "SAM_rad": _fmt_csv(sam_rad),
            "SAM_deg": _fmt_csv(sam_deg),
            "RMSE": _fmt_csv(vr),
            "ERGAS": _fmt_csv(ve),
            "best_psnr": _fmt_csv(ckpt_mgr.best_psnr) if ckpt_mgr.best_psnr_epoch >= 0 else "NaN",
            "best_psnr_epoch": ckpt_mgr.best_psnr_epoch if ckpt_mgr.best_psnr_epoch >= 0 else "",
            "best_sam_deg": _fmt_csv(ckpt_mgr.best_sam_deg) if ckpt_mgr.best_sam_epoch >= 0 else "NaN",
            "best_sam_epoch": ckpt_mgr.best_sam_epoch if ckpt_mgr.best_sam_epoch >= 0 else "",
            "best_ergas": _fmt_csv(ckpt_mgr.best_ergas) if ckpt_mgr.best_ergas_epoch >= 0 else "NaN",
            "best_ergas_epoch": ckpt_mgr.best_ergas_epoch if ckpt_mgr.best_ergas_epoch >= 0 else "",
            "lr": f"{lr_now:.8e}",
            "gpu_mem_allocated_gb": f"{mem_alloc:.4f}",
            "gpu_mem_reserved_gb": f"{mem_reserved:.4f}",
            "epoch_wall_time_sec": f"{epoch_wall:.3f}",
            "psnr_z_coarse": _fmt_csv(float(val_metrics.get("val_psnr_coarse", float("nan")))),
            "psnr_gap": _fmt_csv(float(val_metrics.get("psnr_gap", float("nan")))),
        }
        for k in TRAIN_VALID_METRICS_FIELDS:
            if k in diag_aux:
                csv_row[k] = _fmt_csv(float(diag_aux[k]))
        metrics_logger.append(csv_row)

        updated_names: list[str] = []
        saved_files: list[str] = []
        if updated["psnr"]:
            updated_names.append("PSNR")
            saved_files.append("best_psnr.pth")
        if updated["sam"]:
            updated_names.append("SAM")
            saved_files.append("best_sam.pth")
        if updated["ergas"]:
            updated_names.append("ERGAS")
            saved_files.append("best_ergas.pth")
        dual.log_epoch_summary_line(
            format_epoch_line(
                epoch=epoch,
                epochs=epochs,
                train_loss=train_loss,
                psnr=vp,
                ssim=vssim if val_metrics["has_gt"] else None,
                sam_deg=sam_deg,
                rmse=vr,
                ergas=ve,
                best_psnr=ckpt_mgr.best_psnr,
                best_psnr_epoch=ckpt_mgr.best_psnr_epoch,
                lr=lr_now,
                mem_gb=mem_alloc,
                wall_s=wall_cum,
            )
        )
        if updated_names:
            dual.log_best_update(epoch, updated_names, saved_files)

        if is_b10:
            from utils.b10_protocol_eval import (
                evaluate_all_b10_settings,
                step_b10_checkpoints,
                write_metrics_by_setting_csv,
            )

            model.eval()
            proto_root = resolve_path(ROOT, dcfg.get("root"))
            setting_rows = evaluate_all_b10_settings(
                model,
                proto_root,
                b10_settings,
                eval_crop_policy=eval_crop_policy,
                device=device,
                scale=scale,
            )
            write_metrics_by_setting_csv(metrics_by_setting_path, epoch, setting_rows)
            b10_best_tracker, b10_saved = step_b10_checkpoints(
                b10_best_tracker, epoch, setting_rows, payload, run_dir
            )
            dual.log_plain_file(
                "[b10_val E"
                + f"{epoch:03d}] "
                + " ".join(f"{r['Setting']}={r['PSNR']:.2f}" for r in setting_rows)
            )
            if b10_saved:
                dual.log_plain_file(f"[b10_val E{epoch:03d}] saved: {', '.join(b10_saved)}")
            (run_dir / "b10_best_tracker.json").write_text(
                json.dumps(b10_best_tracker, indent=2) + "\n",
                encoding="utf-8",
            )
        elif mismatch_val_roots and isinstance(mismatch_val_roots, dict):
            from utils.b9_smoke_eval import (
                aggregate_setting_psnrs,
                evaluate_all_settings,
                step_smoke_best_checkpoints,
                write_metrics_by_setting_csv,
            )

            model.eval()
            setting_rows = evaluate_all_settings(
                model,
                mismatch_val_roots,
                eval_crop_policy=eval_crop_policy,
                device=device,
                scale=scale,
                resolve_path_fn=resolve_path,
                project_root=ROOT,
            )
            write_metrics_by_setting_csv(metrics_by_setting_path, epoch, setting_rows)
            smoke_best_tracker, smoke_saved = step_smoke_best_checkpoints(
                smoke_best_tracker, epoch, setting_rows, payload, run_dir
            )
            agg = aggregate_setting_psnrs(setting_rows, "PSNR")
            dual.log_plain_file(
                f"[mismatch_val E{epoch:03d}] mean_psnr={agg['mean_psnr']:.4f} "
                f"worst_case_psnr={agg['worst_case_psnr']:.4f} "
                + " ".join(f"{r['Setting']}={r['PSNR']:.2f}" for r in setting_rows)
            )
            if smoke_saved:
                dual.log_plain_file(f"[mismatch_val E{epoch:03d}] saved smoke ckpt: {', '.join(smoke_saved)}")
            (run_dir / "smoke_best_tracker.json").write_text(
                json.dumps(smoke_best_tracker, indent=2) + "\n",
                encoding="utf-8",
            )

        scheduler.step()

    fin = f"Finished {epochs} epochs -> {run_dir}"
    if ckpt_mgr.best_psnr_epoch >= 0:
        best_line = f"Best PSNR: {ckpt_mgr.best_psnr:.4f} @ epoch {ckpt_mgr.best_psnr_epoch:03d}"
    else:
        best_line = "Best PSNR: N/A"
    dual.log_both_plain(fin)
    dual.log_both_plain(best_line)
    _log_saved_artifacts(dual, run_dir)


def _log_saved_artifacts(dual: TrainingDualLogger, run_dir: Path) -> None:
    """Print final Saved list for artifacts that exist under ``run_dir``."""
    names = (
        "config.yaml",
        "args.json",
        "train_valid.log",
        "train_valid_metrics.csv",
        "latest.pth",
        "best_psnr.pth",
        "best_clean_psnr.pth",
        "best_mean_psnr.pth",
        "best_worst_case_psnr.pth",
        "best_mean_heldout_psnr.pth",
        "best_worst_heldout_psnr.pth",
        "best_joint_heldout_psnr.pth",
        "b10_best_tracker.json",
        "best_sam.pth",
        "best_ergas.pth",
        "smoke_best_tracker.json",
        "metrics_by_setting.csv",
        "robust_ckpt_selection_meta.json",
    )
    dual.log_plain_file("Saved:")
    dual.tty("Saved:")
    for name in names:
        if (run_dir / name).is_file():
            dual.log_plain_file(f"  {name}")
            dual.tty(f"  {name}")


if __name__ == "__main__":
    main()
