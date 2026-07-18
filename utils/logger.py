"""Training, metric, and checkpoint logging utilities for DCRF-Net."""

from __future__ import annotations

import csv
import json
import logging
import math
import re
import shutil
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

_ANSI_RE = re.compile(r"\x1B\[[0-9;]*m")

PREFIX_CONFIG = "[Config] "
PREFIX_SYSTEM = "[System] "

RESET = "\033[0m"
CYAN = "\033[36m"
BLUE = "\033[34m"
GREEN = "\033[32m"
DIM = "\033[2m"


def strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)


def _c(msg: str, color: str) -> str:
    return f"{color}{msg}{RESET}"


class _AnsiFreeFileFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__(fmt="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        return strip_ansi(super().format(record))


class TrainingDualLogger:
    """Routes plain timestamped lines to ``train_valid.log`` and TTY via ``tqdm.write``."""

    _LOGGER_NAME = "dcrf_net.train.dual"

    def __init__(self, run_dir: Path) -> None:
        self._run_dir = Path(run_dir)
        self._log_path = self._run_dir / "train_valid.log"
        self._logger: Optional[logging.Logger] = None

    def setup(self, *, append: bool = False) -> None:
        self._run_dir.mkdir(parents=True, exist_ok=True)
        lg = logging.getLogger(self._LOGGER_NAME)
        lg.handlers.clear()
        lg.setLevel(logging.INFO)
        lg.propagate = False
        fh = logging.FileHandler(self._log_path, mode="a" if append else "w", encoding="utf-8")
        fh.setFormatter(_AnsiFreeFileFormatter())
        lg.addHandler(fh)
        self._logger = lg

    def _file(self, plain: str) -> None:
        if self._logger is not None:
            self._logger.info(plain)

    def tty(self, msg: str) -> None:
        tqdm.write(msg)

    def log_plain_file(self, plain: str) -> None:
        self._file(plain)

    def log_system_file_only(self, plain_detail: str) -> None:
        self._file(f"{PREFIX_SYSTEM}{plain_detail}")

    def log_both_plain(self, plain: str) -> None:
        self._file(plain)
        tqdm.write(plain)

    def log_epoch_summary_line(self, line: str) -> None:
        plain = strip_ansi(line)
        self._file(plain)
        tqdm.write(plain)

    def emit_config_box_top(self, *, width: int = 60) -> None:
        sep = "=" * width
        self.tty("")
        self.tty(_c(sep, DIM))
        self.tty(_c(sep, DIM))

    def emit_config_line(self, key: str, value: str) -> None:
        plain = f"{key}={value}"
        self._file(f"{PREFIX_CONFIG}{plain}")
        self.tty(_c(f"{PREFIX_CONFIG}{plain}", CYAN))

    def emit_config_box_bottom(self, *, width: int = 60) -> None:
        self.log_system_file_only("phase=training_main  boundary=after_config")
        sep = "=" * width
        self.tty(_c(sep, DIM))
        self.tty(_c(sep, DIM))
        self.tty("")

    def log_best_update(self, epoch: int, updated_names: list[str], saved_files: list[str]) -> None:
        names = ", ".join(updated_names)
        files = " ".join(saved_files)
        best_plain = f"[Best] epoch={epoch:03d} updated: {names}"
        save_plain = f"  -> saved: {files}"
        self._file(best_plain)
        self._file(save_plain)
        self.tty(_c(best_plain, GREEN))
        self.tty(save_plain)
        self.log_system_file_only(
            f"epoch={epoch:03d}  event=best_checkpoint  updated={names.replace(' ', '')}  files={files}"
        )


ArgsLike = Union[Namespace, Mapping[str, Any], dict]

TRAIN_VALID_METRICS_FIELDS: List[str] = [
    "epoch",
    "train_loss",
    "rec_loss",
    "rec_coarse_loss",
    "obs_loss",
    "sam_loss",
    "deg_loss",
    "sel_loss",
    "PSNR",
    "SSIM",
    "SAM_rad",
    "SAM_deg",
    "RMSE",
    "ERGAS",
    "best_psnr",
    "best_psnr_epoch",
    "best_sam_deg",
    "best_sam_epoch",
    "best_ergas",
    "best_ergas_epoch",
    "lr",
    "gpu_mem_allocated_gb",
    "gpu_mem_reserved_gb",
    "epoch_wall_time_sec",
    "r_spe_mean",
    "r_spe_max",
    "r_spa_mean",
    "r_spa_max",
    "gate_reliable_mean",
    "gate_spectral_mean",
    "gate_spatial_mean",
    "gate_mixed_mean",
    "gate_entropy",
    "delta_spe_norm",
    "delta_spa_norm",
    "delta_mix_norm",
    "phi_delta_norm",
    "k_delta_norm",
    "selection_ratio",
    "psnr_z_coarse",
    "psnr_gap",
    "obs_coarse_loss",
    "signed_m_residual_abs_mean",
    "signed_h_residual_abs_mean",
    "detail_m_residual_abs_mean",
    "m_res_guidance_abs_mean",
    "h_res_guidance_abs_mean",
    "m_detail_guidance_abs_mean",
    "weighted_residual_abs_mean",
    "residual_tanh_saturation_ratio",
    "gate_reliable_std",
    "gate_spectral_std",
    "gate_spatial_std",
    "gate_mixed_std",
    "gate_entropy_mean",
]


def format_num_params(n: int) -> str:
    n = int(n)
    if n < 0:
        return str(n)
    if n < 1_000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1_000:.2f}K"
    if n < 1_000_000_000:
        return f"{n / 1_000_000:.2f}M"
    return f"{n / 1_000_000_000:.2f}B"


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def create_run_dir(save_dir: str | Path, overwrite: bool = False) -> Path:
    run_dir = Path(save_dir).resolve()
    if run_dir.exists() and overwrite:
        shutil.rmtree(run_dir)
    elif run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        print(
            f"warning: run_dir already exists and is non-empty: {run_dir} "
            "(continuing; set overwrite=True to replace)"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config_snapshot(cfg: dict, run_dir: Path) -> Path:
    path = run_dir / "config.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    return path


def _args_to_dict(args: ArgsLike) -> dict:
    if isinstance(args, Namespace):
        return vars(args)
    return dict(args)


def save_args_snapshot(args: ArgsLike, run_dir: Path) -> Path:
    path = run_dir / "args.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(_args_to_dict(args), f, indent=2, default=str)
    return path


class CSVSchemaMismatchError(ValueError):
    """Raised when an existing metrics CSV header does not match the expected schema."""


class CSVLogger:
    """Append rows to ``train_valid_metrics.csv`` with a fixed header schema."""

    def __init__(self, csv_path: Path, fieldnames: Optional[List[str]] = None) -> None:
        self.path = Path(csv_path)
        self.fieldnames: List[str] = list(fieldnames or TRAIN_VALID_METRICS_FIELDS)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._validate_existing_header()

    def _validate_existing_header(self) -> None:
        if not self.path.is_file() or self.path.stat().st_size == 0:
            return
        with self.path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            existing = next(reader, None)
        if existing is None:
            return
        if list(existing) == self.fieldnames:
            return
        raise CSVSchemaMismatchError(
            "train_valid_metrics.csv schema mismatch:\n"
            f"  path: {self.path}\n"
            f"  existing header ({len(existing)} cols): {existing}\n"
            f"  expected header ({len(self.fieldnames)} cols): {self.fieldnames}\n"
            "Use a new run_name, delete the old CSV, or recreate the run with overwrite=True."
        )

    def append(self, row: Mapping[str, Any]) -> None:
        out = {k: "" for k in self.fieldnames}
        for k, v in row.items():
            if k in self.fieldnames:
                out[k] = v
        write_header = not self.path.is_file() or self.path.stat().st_size == 0
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(out)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def tensor_stats(x: torch.Tensor, prefix: str = "") -> Dict[str, float]:
    t = x.detach().float().reshape(-1)
    t = t[torch.isfinite(t)]
    if t.numel() == 0:
        base = prefix.rstrip("_")
        return {
            f"{base}_mean": float("nan"),
            f"{base}_std": float("nan"),
            f"{base}_min": float("nan"),
            f"{base}_max": float("nan"),
            f"{base}_norm": float("nan"),
        }
    p = f"{prefix}_" if prefix and not prefix.endswith("_") else prefix
    if prefix and prefix.endswith("_"):
        p = prefix
    return {
        f"{p}mean": float(t.mean().item()),
        f"{p}std": float(t.std(unbiased=False).item()),
        f"{p}min": float(t.min().item()),
        f"{p}max": float(t.max().item()),
        f"{p}norm": float(t.norm().item()),
    }


def gpu_memory_gb() -> float:
    if torch.cuda.is_available():
        return float(torch.cuda.memory_allocated() / (1024**3))
    return 0.0


def gpu_memory_reserved_gb() -> float:
    if torch.cuda.is_available():
        return float(torch.cuda.memory_reserved() / (1024**3))
    return 0.0


def get_optimizer_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def sam_rad_deg_pair(sam_deg: float) -> tuple[float, float]:
    """Return consistent (SAM_rad, SAM_deg) from SAM in degrees."""
    if not math.isfinite(sam_deg):
        return float("nan"), float("nan")
    rad = sam_deg * math.pi / 180.0
    return rad, sam_deg


def format_epoch_line(
    *,
    epoch: int,
    epochs: int,
    train_loss: float,
    psnr: float,
    ssim: Optional[float],
    sam_deg: float,
    rmse: float,
    ergas: float,
    best_psnr: float,
    best_psnr_epoch: int,
    lr: float,
    mem_gb: float,
    wall_s: float,
) -> str:
    best_tok = f"{best_psnr:.4f}@{best_psnr_epoch:03d}" if best_psnr_epoch >= 0 else "---@---"
    ssim_s = f"{ssim:.4f}" if ssim is not None and math.isfinite(ssim) else "N/A"
    return (
        f"[E{epoch:03d}/{epochs:03d}] train={train_loss:.2e} | "
        f"PSNR={psnr:.4f} SSIM={ssim_s} SAM_deg={sam_deg:.3f} RMSE={rmse:.4f} ERGAS={ergas:.4f} | "
        f"best={best_tok} | lr={lr:.2e} | mem={mem_gb:.2f}G | wall={wall_s:.1f}s"
    )


def collect_dcsr_diagnostics(
    pred: Mapping[str, torch.Tensor],
    batch: Optional[Mapping[str, torch.Tensor]] = None,
) -> Dict[str, float]:
    """DCSR-Net diagnostic scalars for ``train_valid_metrics.csv`` aux columns."""

    def _mean_scalar(t: torch.Tensor) -> float:
        return float(t.detach().float().mean().item())

    def _max_scalar(t: torch.Tensor) -> float:
        return float(t.detach().float().max().item())

    out: Dict[str, float] = {}
    if "r_spe" in pred:
        out["r_spe_mean"] = _mean_scalar(pred["r_spe"])
        out["r_spe_max"] = _max_scalar(pred["r_spe"])
    if "r_spa" in pred:
        out["r_spa_mean"] = _mean_scalar(pred["r_spa"])
        out["r_spa_max"] = _max_scalar(pred["r_spa"])
    if "gates" in pred:
        g = pred["gates"].detach().float()
        out["gate_reliable_mean"] = _mean_scalar(g[:, 0:1])
        out["gate_spectral_mean"] = _mean_scalar(g[:, 1:2])
        out["gate_spatial_mean"] = _mean_scalar(g[:, 2:3])
        out["gate_mixed_mean"] = _mean_scalar(g[:, 3:4])
        out["gate_reliable_std"] = float(g[:, 0:1].std(unbiased=False).item())
        out["gate_spectral_std"] = float(g[:, 1:2].std(unbiased=False).item())
        out["gate_spatial_std"] = float(g[:, 2:3].std(unbiased=False).item())
        out["gate_mixed_std"] = float(g[:, 3:4].std(unbiased=False).item())
        g_clamped = g.clamp(min=1e-8)
        entropy = -(g_clamped * g_clamped.log()).sum(dim=1).mean()
        ent_val = float(entropy.item())
        out["gate_entropy"] = ent_val
        out["gate_entropy_mean"] = ent_val
        out["selection_ratio"] = float(g[:, 1:].sum(dim=1).mean().item())
    if "signed_m_residual" in pred:
        out["signed_m_residual_abs_mean"] = _mean_scalar(pred["signed_m_residual"].abs())
    if "signed_h_residual_up" in pred:
        out["signed_h_residual_abs_mean"] = _mean_scalar(pred["signed_h_residual_up"].abs())
    if "detail_m_residual" in pred:
        out["detail_m_residual_abs_mean"] = _mean_scalar(pred["detail_m_residual"].abs())
    if "m_res_guidance" in pred:
        out["m_res_guidance_abs_mean"] = _mean_scalar(pred["m_res_guidance"].abs())
    if "h_res_guidance" in pred:
        out["h_res_guidance_abs_mean"] = _mean_scalar(pred["h_res_guidance"].abs())
    if "m_detail_guidance" in pred:
        out["m_detail_guidance_abs_mean"] = _mean_scalar(pred["m_detail_guidance"].abs())
    if "weighted_residual" in pred:
        wr = pred["weighted_residual"].detach().float()
        out["weighted_residual_abs_mean"] = _mean_scalar(wr.abs())
        out["residual_tanh_saturation_ratio"] = float((wr.abs() > 3.0).float().mean().item())
    for key, out_key in (
        ("delta_spe", "delta_spe_norm"),
        ("delta_spa", "delta_spa_norm"),
        ("delta_mix", "delta_mix_norm"),
    ):
        if key in pred:
            out[out_key] = float(pred[key].detach().float().norm().item())
    if batch is not None and "phi0" in batch and "phi_tilde" in pred:
        phi0 = batch["phi0"]
        if phi0.ndim == 3 and phi0.shape[0] == 1:
            phi0 = phi0.expand(pred["phi_tilde"].shape[0], -1, -1)
        out["phi_delta_norm"] = float((pred["phi_tilde"] - phi0).detach().float().norm().item())
    if batch is not None and "k0" in batch and "k_tilde" in pred:
        k0 = batch["k0"]
        b = pred["k_tilde"].shape[0]
        if k0.ndim == 5 and k0.shape[1] == 1 and k0.shape[2] == 1:
            k0 = k0.squeeze(2)
        if k0.ndim == 4 and k0.shape[0] == 1 and b > 1:
            k0 = k0.expand(b, -1, -1, -1)
        out["k_delta_norm"] = float((pred["k_tilde"] - k0).detach().float().norm().item())
    return out


class CheckpointManager:
    """Save latest/best checkpoints and ``robust_ckpt_selection_meta.json``."""

    def __init__(
        self,
        run_dir: Path,
        *,
        primary_metric: str = "psnr",
        save_latest: bool = True,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.primary_metric = primary_metric
        self.save_latest = save_latest
        self.best_psnr = float("-inf")
        self.best_psnr_epoch = -1
        self.best_sam_deg = float("inf")
        self.best_sam_epoch = -1
        self.best_ergas = float("inf")
        self.best_ergas_epoch = -1
        self.latest_epoch = 0

    def restore_from_meta(self, meta_path: Path | None = None) -> bool:
        """Restore best-metric tracker from ``robust_ckpt_selection_meta.json`` (resume)."""
        path = Path(meta_path) if meta_path is not None else self.run_dir / "robust_ckpt_selection_meta.json"
        if not path.is_file():
            return False
        with path.open(encoding="utf-8") as f:
            meta = json.load(f)
        bp = meta.get("best_psnr")
        if bp is not None and meta.get("best_psnr_epoch", -1) >= 0:
            self.best_psnr = float(bp)
            self.best_psnr_epoch = int(meta["best_psnr_epoch"])
        bs = meta.get("best_sam_deg")
        if bs is not None and meta.get("best_sam_epoch", -1) >= 0:
            self.best_sam_deg = float(bs)
            self.best_sam_epoch = int(meta["best_sam_epoch"])
        be = meta.get("best_ergas")
        if be is not None and meta.get("best_ergas_epoch", -1) >= 0:
            self.best_ergas = float(be)
            self.best_ergas_epoch = int(meta["best_ergas_epoch"])
        le = meta.get("latest_epoch")
        if le is not None:
            self.latest_epoch = int(le)
        return True

    def _is_better(self, metric: str, value: float, current_best: float) -> bool:
        if not math.isfinite(value):
            return False
        if metric in ("sam", "ergas"):
            return value < current_best
        return value > current_best

    def step(
        self,
        epoch: int,
        payload: dict,
        *,
        val_psnr: Optional[float] = None,
        val_sam_deg: Optional[float] = None,
        val_ergas: Optional[float] = None,
    ) -> dict[str, bool]:
        """Save checkpoints; return which best metrics improved this epoch."""
        updated = {"psnr": False, "sam": False, "ergas": False}
        self.latest_epoch = int(epoch)
        if self.save_latest:
            torch.save(payload, self.run_dir / "latest.pth")

        if val_psnr is not None and self._is_better("psnr", val_psnr, self.best_psnr):
            self.best_psnr = float(val_psnr)
            self.best_psnr_epoch = int(epoch)
            torch.save(payload, self.run_dir / "best_psnr.pth")
            updated["psnr"] = True

        if val_sam_deg is not None and self._is_better("sam", val_sam_deg, self.best_sam_deg):
            self.best_sam_deg = float(val_sam_deg)
            self.best_sam_epoch = int(epoch)
            torch.save(payload, self.run_dir / "best_sam.pth")
            updated["sam"] = True

        if val_ergas is not None and self._is_better("ergas", val_ergas, self.best_ergas):
            self.best_ergas = float(val_ergas)
            self.best_ergas_epoch = int(epoch)
            torch.save(payload, self.run_dir / "best_ergas.pth")
            updated["ergas"] = True

        self.write_meta()
        return updated

    def write_meta(self) -> None:
        meta = {
            "best_psnr": self.best_psnr if self.best_psnr_epoch >= 0 else None,
            "best_psnr_epoch": self.best_psnr_epoch,
            "best_sam_deg": self.best_sam_deg if self.best_sam_epoch >= 0 else None,
            "best_sam_epoch": self.best_sam_epoch,
            "best_ergas": self.best_ergas if self.best_ergas_epoch >= 0 else None,
            "best_ergas_epoch": self.best_ergas_epoch,
            "latest_epoch": self.latest_epoch,
            "primary_metric": self.primary_metric,
            "ckpt_type": self.primary_metric,
        }
        path = self.run_dir / "robust_ckpt_selection_meta.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=_json_default)
