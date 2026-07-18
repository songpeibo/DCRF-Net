#!/usr/bin/env python3
"""Train DCRF-Net with the standard baseline-export protocol."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.baseline_export import (  # noqa: E402
    BaselineExportDataset,
    eval_crop_policy_description,
    normalize_eval_crop_policy,
)
from models.build_model import build_model_from_config, save_effective_config  # noqa: E402
from utils.common import resolve_path  # noqa: E402
from utils.losses import DCSRLoss  # noqa: E402
from utils.metrics import ergas, psnr, rmse, sam, ssim  # noqa: E402
from utils.seed import set_seed  # noqa: E402


LOSS_KEYS = (
    "loss_rec",
    "loss_rec_coarse",
    "loss_obs",
    "loss_obs_coarse",
    "loss_sam",
    "loss_deg",
    "loss_sel",
    "loss_feedback_intermediate",
)

CSV_FIELDS = (
    "epoch",
    "train_loss",
    "val_loss",
    *LOSS_KEYS,
    "PSNR",
    "SSIM",
    "SAM_deg",
    "RMSE",
    "ERGAS",
    "PSNR_coarse",
    "PSNR_gain",
    "lr",
    "epoch_time_sec",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DCRF-Net")
    parser.add_argument("--config", required=True, help="Path to the YAML config")
    parser.add_argument("--data-root", default=None, help="Override data.root")
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda:0 or cpu")
    parser.add_argument("--resume", default=None, help="Checkpoint used to resume training")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing non-empty output directory",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"The config must be a YAML mapping: {path}")
    return config


def collate_fusion(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack samples while keeping an identical SRF/PSF shared by the batch."""
    if not batch:
        raise ValueError("Cannot collate an empty batch")

    output: dict[str, Any] = {}
    for key in batch[0]:
        values = [sample[key] for sample in batch]
        first = values[0]

        shared_operator = (
            key in {"phi0", "k0"}
            and isinstance(first, torch.Tensor)
            and all(
                isinstance(value, torch.Tensor) and torch.equal(first, value)
                for value in values[1:]
            )
        )
        if shared_operator:
            output[key] = first
        elif isinstance(first, torch.Tensor):
            output[key] = torch.stack(values, dim=0)
        else:
            output[key] = values
    return output


def move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def prepare_run_dir(path: Path, *, overwrite: bool, resume: bool) -> None:
    if resume:
        path.mkdir(parents=True, exist_ok=True)
        return

    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {path}\n"
                "Use --overwrite or set train.overwrite: true."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def build_datasets(
    config: dict[str, Any],
) -> tuple[BaselineExportDataset, BaselineExportDataset, Path, str]:
    data_config = config.get("data", {}) or {}
    split_mode = str(data_config.get("split_mode", "baseline_export")).lower()
    if split_mode != "baseline_export":
        raise ValueError(
            "This public training script supports only "
            f"data.split_mode: baseline_export, but got {split_mode!r}."
        )
    if "root" not in data_config:
        raise KeyError("Missing data.root in the config")

    data_root = resolve_path(ROOT, data_config["root"])
    crop_policy = normalize_eval_crop_policy(data_config.get("eval_crop_policy"))
    train_set = BaselineExportDataset(
        data_root,
        split="train",
        eval_crop_policy=crop_policy,
    )
    val_set = BaselineExportDataset(
        data_root,
        split="val",
        eval_crop_policy=crop_policy,
    )
    return train_set, val_set, data_root, crop_policy


def build_loss(config: dict[str, Any], scale: int) -> DCSRLoss:
    loss_config = config.get("loss", {}) or {}
    return DCSRLoss(
        scale=scale,
        lambda_rec=float(loss_config.get("lambda_rec", 1.0)),
        lambda_rec_coarse=float(loss_config.get("lambda_rec_coarse", 0.0)),
        lambda_obs=float(loss_config.get("lambda_obs", 0.2)),
        lambda_obs_coarse=float(loss_config.get("lambda_obs_coarse", 0.1)),
        lambda_sam=float(loss_config.get("lambda_sam", 0.05)),
        lambda_deg=float(loss_config.get("lambda_deg", 0.001)),
        lambda_sel=float(loss_config.get("lambda_sel", 0.0001)),
        lambda_feedback_intermediate=float(
            loss_config.get("lambda_feedback_intermediate", 0.0)
        ),
    )


def make_loaders(
    train_set: BaselineExportDataset,
    val_set: BaselineExportDataset,
    *,
    batch_size: int,
    val_batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    if min(batch_size, val_batch_size) < 1:
        raise ValueError("Batch sizes must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    common = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
        "collate_fn": collate_fusion,
    }
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        **common,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=val_batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, val_loader


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: DCSRLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    *,
    epoch: int,
    epochs: int,
    amp_enabled: bool,
) -> tuple[float, dict[str, float]]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    component_sums = {key: 0.0 for key in LOSS_KEYS}

    progress = tqdm(
        loader,
        desc=f"Train {epoch:03d}/{epochs:03d}",
        leave=False,
        dynamic_ncols=True,
        unit="batch",
    )
    for batch in progress:
        batch = move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            prediction = model(batch)
            loss_dict = criterion(prediction, batch)
            loss = loss_dict["loss"]

        if not isinstance(loss, torch.Tensor):
            raise TypeError("DCSRLoss must return a tensor named 'loss'")
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss at epoch {epoch}: {float(loss.detach().item())}"
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        current_batch_size = int(batch["h"].shape[0])
        loss_value = float(loss.detach().item())
        total_loss += loss_value * current_batch_size
        total_samples += current_batch_size

        for key in LOSS_KEYS:
            if key in loss_dict:
                component_sums[key] += float(loss_dict[key]) * current_batch_size

        progress.set_postfix(
            loss=f"{loss_value:.3e}",
            rec=f"{float(loss_dict.get('loss_rec', math.nan)):.3e}",
            obs=f"{float(loss_dict.get('loss_obs', math.nan)):.3e}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            refresh=False,
        )

    denominator = max(1, total_samples)
    components = {key: value / denominator for key, value in component_sums.items()}
    return total_loss / denominator, components


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: DCSRLoss,
    device: torch.device,
    *,
    scale: int,
    epoch: int,
    epochs: int,
    amp_enabled: bool,
) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    metric_batches = 0
    coarse_batches = 0
    sums = {
        "psnr": 0.0,
        "ssim": 0.0,
        "sam": 0.0,
        "rmse": 0.0,
        "ergas": 0.0,
        "psnr_coarse": 0.0,
    }

    progress = tqdm(
        loader,
        desc=f"Val   {epoch:03d}/{epochs:03d}",
        leave=False,
        dynamic_ncols=True,
        unit="batch",
    )
    for batch in progress:
        batch = move_to_device(batch, device)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            prediction = model(batch)
            loss_dict = criterion(prediction, batch)
            loss = loss_dict["loss"]

        current_batch_size = int(batch["h"].shape[0])
        total_loss += float(loss.detach().item()) * current_batch_size
        total_samples += current_batch_size

        reconstruction = prediction["z_hat"].float()
        target = batch["gt"].float()
        sums["psnr"] += psnr(reconstruction, target)
        sums["ssim"] += ssim(reconstruction, target)
        sums["sam"] += sam(reconstruction, target)
        sums["rmse"] += rmse(reconstruction, target)
        sums["ergas"] += ergas(reconstruction, target, scale=scale)
        metric_batches += 1

        coarse = prediction.get("z_coarse", prediction.get("z0"))
        if isinstance(coarse, torch.Tensor):
            sums["psnr_coarse"] += psnr(coarse.float(), target)
            coarse_batches += 1

        progress.set_postfix(
            PSNR=f"{sums['psnr'] / metric_batches:.2f}",
            SSIM=f"{sums['ssim'] / metric_batches:.4f}",
            SAM=f"{sums['sam'] / metric_batches:.2f}",
            refresh=False,
        )

    if metric_batches == 0:
        raise RuntimeError("The validation loader produced no batches")

    metrics = {
        "psnr": sums["psnr"] / metric_batches,
        "ssim": sums["ssim"] / metric_batches,
        "sam": sums["sam"] / metric_batches,
        "rmse": sums["rmse"] / metric_batches,
        "ergas": sums["ergas"] / metric_batches,
        "psnr_coarse": (
            sums["psnr_coarse"] / coarse_batches
            if coarse_batches > 0
            else float("nan")
        ),
    }
    metrics["psnr_gain"] = (
        metrics["psnr"] - metrics["psnr_coarse"]
        if math.isfinite(metrics["psnr_coarse"])
        else float("nan")
    )
    return total_loss / max(1, total_samples), metrics


def save_checkpoint(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    best_path: Path,
) -> tuple[int, float, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")

    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"Invalid checkpoint: {path}")

    model.load_state_dict(payload["model"], strict=True)
    if isinstance(payload.get("optimizer"), dict):
        optimizer.load_state_dict(payload["optimizer"])
    if isinstance(payload.get("scheduler"), dict):
        scheduler.load_state_dict(payload["scheduler"])
    if scaler.is_enabled() and isinstance(payload.get("scaler"), dict):
        scaler.load_state_dict(payload["scaler"])

    epoch = int(payload.get("epoch", -1))
    if epoch < 0:
        raise ValueError(f"Checkpoint has no valid epoch: {path}")

    best_psnr = float(payload.get("best_psnr", -math.inf))
    best_epoch = int(payload.get("best_psnr_epoch", -1))
    if not math.isfinite(best_psnr) and best_path.is_file():
        best_payload = torch.load(best_path, map_location=device, weights_only=False)
        if isinstance(best_payload, dict):
            best_metrics = best_payload.get("metrics", {})
            if isinstance(best_metrics, dict):
                restored = best_metrics.get("psnr", best_metrics.get("val_psnr"))
                if restored is not None:
                    best_psnr = float(restored)
                    best_epoch = int(best_payload.get("epoch", -1))

    if not math.isfinite(best_psnr):
        metrics = payload.get("metrics", {})
        if isinstance(metrics, dict):
            current = metrics.get("psnr", metrics.get("val_psnr"))
            if current is not None:
                best_psnr = float(current)
                best_epoch = epoch

    return epoch + 1, best_psnr, best_epoch


def make_checkpoint(
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    config: dict[str, Any],
    metrics: dict[str, float],
    best_psnr: float,
    best_psnr_epoch: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": copy.deepcopy(config),
        "metrics": metrics,
        "best_psnr": best_psnr,
        "best_psnr_epoch": best_psnr_epoch,
    }
    if scaler.is_enabled():
        payload["scaler"] = scaler.state_dict()
    return payload


def append_metrics(path: Path, row: dict[str, Any]) -> None:
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def finite_text(value: Any, digits: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{number:.{digits}f}" if math.isfinite(number) else ""


def log(message: str, file: Any) -> None:
    print(message, flush=True)
    file.write(message + "\n")
    file.flush()


def main() -> None:
    args = parse_args()
    config_path = resolve_path(ROOT, args.config)
    config = load_config(config_path)

    if args.data_root is not None:
        config.setdefault("data", {})["root"] = args.data_root

    train_config = config.get("train", {}) or {}
    data_config = config.get("data", {}) or {}
    model_config = config.get("model_args", {}) or {}

    seed = int(config.get("seed", 42))
    set_seed(seed)

    device_name = str(args.device or config.get("device", "cuda"))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; using CPU instead.", flush=True)
        device_name = "cpu"
    device = torch.device(device_name)

    scale = int(
        config.get(
            "scale",
            data_config.get("scale", model_config.get("scale", 4)),
        )
    )
    epochs = int(train_config.get("epochs", 400))
    batch_size = int(train_config.get("batch_size", 8))
    val_batch_size = int(train_config.get("val_batch_size", batch_size))
    num_workers = int(train_config.get("num_workers", 4))
    learning_rate = float(train_config.get("lr", 2e-4))
    weight_decay = float(train_config.get("weight_decay", 1e-6))
    eval_every = int(train_config.get("eval_every", 1))
    amp_requested = bool(train_config.get("amp", False))
    amp_enabled = amp_requested and device.type == "cuda"

    if epochs < 1 or eval_every < 1:
        raise ValueError("train.epochs and train.eval_every must be positive")

    run_dir = resolve_path(
        ROOT,
        train_config.get("save_dir", "outputs/runs/dcrf_net"),
    )
    resume_path = resolve_path(ROOT, args.resume) if args.resume else None
    overwrite = bool(args.overwrite or train_config.get("overwrite", False))
    if resume_path is not None:
        overwrite = False
    prepare_run_dir(run_dir, overwrite=overwrite, resume=resume_path is not None)

    train_set, val_set, data_root, crop_policy = build_datasets(config)
    train_loader, val_loader = make_loaders(
        train_set,
        val_set,
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        num_workers=num_workers,
        device=device,
    )

    model = build_model_from_config(config, device=device, enforce_nonblind=True)
    criterion = build_loss(config, scale).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(train_config.get("lr_step", 50)),
        gamma=float(train_config.get("lr_gamma", 0.5)),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    latest_path = run_dir / "latest.pth"
    best_path = run_dir / "best_psnr.pth"
    metrics_path = run_dir / "train_valid_metrics.csv"
    start_epoch = 1
    best_psnr = -math.inf
    best_psnr_epoch = -1

    if resume_path is not None:
        start_epoch, best_psnr, best_psnr_epoch = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            best_path=best_path,
        )
        if resume_path.resolve().parent != run_dir.resolve():
            print(
                f"Warning: checkpoint is in {resume_path.parent}, "
                f"but outputs will be saved to {run_dir}.",
                flush=True,
            )

    with (run_dir / "config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)
    with (run_dir / "args.json").open("w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2, ensure_ascii=False)
        file.write("\n")
    save_effective_config(run_dir, config, model)

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    log_mode = "a" if resume_path is not None else "w"
    with (run_dir / "train.log").open(log_mode, encoding="utf-8") as log_file:
        log("=" * 72, log_file)
        log("DCRF-Net training", log_file)
        log(f"Config: {config_path}", log_file)
        log(f"Data root: {data_root}", log_file)
        log(
            f"Evaluation crop: {crop_policy} "
            f"({eval_crop_policy_description(crop_policy)})",
            log_file,
        )
        log(f"Output: {run_dir}", log_file)
        log(f"Device: {device}", log_file)
        log(f"Seed: {seed}", log_file)
        log(f"Training samples: {len(train_set)}", log_file)
        log(f"Validation samples: {len(val_set)}", log_file)
        log(
            f"Parameters: {total_parameters:,} total, "
            f"{trainable_parameters:,} trainable",
            log_file,
        )
        log(
            f"Epochs: {epochs}, batch size: {batch_size}, "
            f"learning rate: {learning_rate:.3e}, AMP: {amp_enabled}",
            log_file,
        )
        if amp_requested and not amp_enabled:
            log("AMP was requested but is disabled on the selected device.", log_file)
        if resume_path is not None:
            log(f"Resume: {resume_path}", log_file)
            log(f"Start epoch: {start_epoch}", log_file)
        log("=" * 72, log_file)

        if start_epoch > epochs:
            log("The checkpoint has already reached the configured epoch limit.", log_file)
            return

        training_start = time.perf_counter()
        for epoch in range(start_epoch, epochs + 1):
            epoch_start = time.perf_counter()
            current_lr = float(optimizer.param_groups[0]["lr"])

            train_loss, train_components = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                device,
                epoch=epoch,
                epochs=epochs,
                amp_enabled=amp_enabled,
            )

            should_validate = epoch % eval_every == 0 or epoch == epochs
            val_loss = float("nan")
            val_metrics: dict[str, float] = {}
            improved = False

            if should_validate:
                val_loss, val_metrics = validate(
                    model,
                    val_loader,
                    criterion,
                    device,
                    scale=scale,
                    epoch=epoch,
                    epochs=epochs,
                    amp_enabled=amp_enabled,
                )
                if val_metrics["psnr"] > best_psnr:
                    best_psnr = val_metrics["psnr"]
                    best_psnr_epoch = epoch
                    improved = True

            scheduler.step()

            checkpoint_metrics = dict(val_metrics)
            checkpoint_metrics["train_loss"] = train_loss
            checkpoint_metrics["val_loss"] = val_loss
            payload = make_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=config,
                metrics=checkpoint_metrics,
                best_psnr=best_psnr,
                best_psnr_epoch=best_psnr_epoch,
            )
            save_checkpoint(payload, latest_path)
            if improved:
                save_checkpoint(payload, best_path)

            epoch_time = time.perf_counter() - epoch_start
            row = {
                "epoch": epoch,
                "train_loss": finite_text(train_loss),
                "val_loss": finite_text(val_loss),
                **{
                    key: finite_text(train_components.get(key))
                    for key in LOSS_KEYS
                },
                "PSNR": finite_text(val_metrics.get("psnr")),
                "SSIM": finite_text(val_metrics.get("ssim")),
                "SAM_deg": finite_text(val_metrics.get("sam")),
                "RMSE": finite_text(val_metrics.get("rmse")),
                "ERGAS": finite_text(val_metrics.get("ergas")),
                "PSNR_coarse": finite_text(val_metrics.get("psnr_coarse")),
                "PSNR_gain": finite_text(val_metrics.get("psnr_gain")),
                "lr": f"{current_lr:.8e}",
                "epoch_time_sec": f"{epoch_time:.3f}",
            }
            append_metrics(metrics_path, row)

            if should_validate:
                log(
                    f"Epoch {epoch:03d}/{epochs:03d} | "
                    f"train {train_loss:.6f} | val {val_loss:.6f} | "
                    f"PSNR {val_metrics['psnr']:.4f} | "
                    f"SSIM {val_metrics['ssim']:.6f} | "
                    f"SAM {val_metrics['sam']:.4f} | "
                    f"RMSE {val_metrics['rmse']:.6f} | "
                    f"ERGAS {val_metrics['ergas']:.4f} | "
                    f"lr {current_lr:.3e} | {epoch_time:.1f}s",
                    log_file,
                )
            else:
                log(
                    f"Epoch {epoch:03d}/{epochs:03d} | "
                    f"train {train_loss:.6f} | validation skipped | "
                    f"lr {current_lr:.3e} | {epoch_time:.1f}s",
                    log_file,
                )

            if improved:
                log(
                    f"Saved best_psnr.pth: PSNR={best_psnr:.6f} "
                    f"at epoch {best_psnr_epoch}.",
                    log_file,
                )

        elapsed_hours = (time.perf_counter() - training_start) / 3600.0
        log("=" * 72, log_file)
        log(f"Training finished in {elapsed_hours:.2f} hours.", log_file)
        log(f"Latest checkpoint: {latest_path}", log_file)
        if best_psnr_epoch >= 0:
            log(
                f"Best checkpoint: {best_path} "
                f"(PSNR={best_psnr:.6f}, epoch={best_psnr_epoch})",
                log_file,
            )
        log(f"Metrics: {metrics_path}", log_file)
        log("=" * 72, log_file)


if __name__ == "__main__":
    main()
