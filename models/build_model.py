"""Construct DCSRNet from a training/export YAML config (single source of truth)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import torch
import torch.nn as nn

from models.dcsr_net import DCSRNet

NONBLIND_ASSERT_MSG = (
    "nonblind_standard requires fixed SRF/PSF: delta_phi_scale and delta_k_scale must both be 0.0."
)

# Constructor fallbacks only when YAML omits a key (logged explicitly).
_GENERAL_DEFAULT_DELTA_PHI = 0.05
_GENERAL_DEFAULT_DELTA_K = 0.05
_NONBLIND_DEFAULT_DELTA_PHI = 0.0
_NONBLIND_DEFAULT_DELTA_K = 0.0


def _cfg_get(cfg: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in cfg:
            return cfg[k]
    return default


def _marg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("model_args", {}) or {}


def _is_nonblind_standard(cfg: dict[str, Any]) -> bool:
    return str(cfg.get("setting", "")).strip() == "nonblind_standard"


def _resolve_float_param(
    cfg: dict[str, Any],
    key: str,
    *,
    nonblind_default: float,
    general_default: float,
) -> tuple[float, str]:
    """Return (value, source) where source is ``explicit``, ``top_level``, or ``default``."""
    marg = _marg(cfg)
    if key in marg:
        return float(marg[key]), "explicit:model_args"
    top = _cfg_get(cfg, key)
    if top is not None:
        return float(top), "explicit:top_level"
    default = nonblind_default if _is_nonblind_standard(cfg) else general_default
    return default, f"default:{default}"


def declared_model_config_from_yaml(cfg: dict[str, Any]) -> dict[str, Any]:
    """Resolved training declaration (explicit YAML + documented defaults)."""
    marg = _marg(cfg)
    dcfg = cfg.get("data", {}) or {}
    dphi, _ = _resolve_float_param(
        cfg,
        "delta_phi_scale",
        nonblind_default=_NONBLIND_DEFAULT_DELTA_PHI,
        general_default=_GENERAL_DEFAULT_DELTA_PHI,
    )
    dk, _ = _resolve_float_param(
        cfg,
        "delta_k_scale",
        nonblind_default=_NONBLIND_DEFAULT_DELTA_K,
        general_default=_GENERAL_DEFAULT_DELTA_K,
    )
    return {
        "delta_phi_scale": dphi,
        "delta_k_scale": dk,
        "base_channels": int(_cfg_get(cfg, "base_channels", default=marg.get("base_channels", 64))),
        "guidance_channels": int(marg.get("guidance_channels", 8)),
        "use_signed_residual_guidance": bool(marg.get("use_signed_residual_guidance", False)),
        "use_msi_detail_guidance": bool(marg.get("use_msi_detail_guidance", False)),
        "use_reliability_gated_coarse_fusion": bool(marg.get("use_reliability_gated_coarse_fusion", False)),
        "use_dual_observation_feedback": bool(marg.get("use_dual_observation_feedback", False)),
        "num_feedback_steps": int(marg.get("num_feedback_steps", 2)),
        "share_feedback_weights": bool(marg.get("share_feedback_weights", True)),
        "feedback_residual_scale": float(marg.get("feedback_residual_scale", 0.1)),
        "feedback_observation_mode": str(marg.get("feedback_observation_mode", "full")),
        "hsi_channels": int(_cfg_get(cfg, "hsi_channels", default=dcfg.get("hsi_channels", 103))),
        "msi_channels": int(_cfg_get(cfg, "msi_channels", default=dcfg.get("msi_channels", 3))),
        "scale": int(_cfg_get(cfg, "scale", default=dcfg.get("scale", 4))),
        "kernel_size": int(marg.get("kernel_size", 7)),
        "norm": str(marg.get("norm", "gn")),
    }


def model_config_resolution_report(cfg: dict[str, Any]) -> dict[str, Any]:
    """How delta scales were resolved from YAML (for logs / effective_config.json)."""
    dphi, dphi_src = _resolve_float_param(
        cfg,
        "delta_phi_scale",
        nonblind_default=_NONBLIND_DEFAULT_DELTA_PHI,
        general_default=_GENERAL_DEFAULT_DELTA_PHI,
    )
    dk, dk_src = _resolve_float_param(
        cfg,
        "delta_k_scale",
        nonblind_default=_NONBLIND_DEFAULT_DELTA_K,
        general_default=_GENERAL_DEFAULT_DELTA_K,
    )
    return {
        "delta_phi_scale": {"value": dphi, "source": dphi_src},
        "delta_k_scale": {"value": dk, "source": dk_src},
        "setting": str(cfg.get("setting", "")),
    }


def effective_model_attributes(model: DCSRNet) -> dict[str, Any]:
    """Runtime attributes on an instantiated module (authoritative for forward)."""
    return {
        "delta_phi_scale": float(model.delta_phi_scale),
        "delta_k_scale": float(model.delta_k_scale),
        "base_channels": int(model.base_channels),
        "guidance_channels": int(model.guidance_channels),
        "use_signed_residual_guidance": bool(model.use_signed_residual_guidance),
        "use_msi_detail_guidance": bool(model.use_msi_detail_guidance),
        "use_reliability_gated_coarse_fusion": bool(model.use_reliability_gated_coarse_fusion),
        "use_dual_observation_feedback": bool(model.use_dual_observation_feedback),
        "num_feedback_steps": int(model.num_feedback_steps),
        "share_feedback_weights": bool(model.share_feedback_weights),
        "feedback_residual_scale": float(model.feedback_residual_scale),
        "feedback_observation_mode": str(model.feedback_observation_mode),
        "use_guidance": bool(model.use_guidance),
        "res_scale": float(model.res_scale),
    }


def assert_nonblind_standard_config(cfg: dict[str, Any]) -> None:
    """Raise if ``setting: nonblind_standard`` but SRF/PSF deltas are non-zero."""
    setting = str(cfg.get("setting", "")).strip()
    if setting != "nonblind_standard":
        return
    declared = declared_model_config_from_yaml(cfg)
    dphi = declared["delta_phi_scale"]
    dk = declared["delta_k_scale"]
    if dphi != 0.0 or dk != 0.0:
        raise ValueError(
            f"{NONBLIND_ASSERT_MSG} "
            f"Got delta_phi_scale={dphi}, delta_k_scale={dk} in config."
        )


def model_kwargs_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Build ``DCSRNet`` constructor kwargs from ``config.yaml`` (data + model_args)."""
    dcfg = cfg.get("data", {}) or {}
    marg = _marg(cfg)
    declared = declared_model_config_from_yaml(cfg)
    return {
        "hsi_channels": int(_cfg_get(cfg, "hsi_channels", default=dcfg.get("hsi_channels", marg.get("hsi_channels", 103)))),
        "msi_channels": int(_cfg_get(cfg, "msi_channels", default=dcfg.get("msi_channels", marg.get("msi_channels", 4)))),
        "scale": int(_cfg_get(cfg, "scale", default=dcfg.get("scale", marg.get("scale", 4)))),
        "kernel_size": int(_cfg_get(cfg, "kernel_size", default=marg.get("kernel_size", 7))),
        "base_channels": int(declared["base_channels"]),
        "delta_phi_scale": float(declared["delta_phi_scale"]),
        "delta_k_scale": float(declared["delta_k_scale"]),
        "norm": str(declared["norm"]),
        "guidance_channels": int(declared["guidance_channels"]),
        "use_signed_residual_guidance": bool(declared["use_signed_residual_guidance"]),
        "use_msi_detail_guidance": bool(declared["use_msi_detail_guidance"]),
        "use_reliability_gated_coarse_fusion": bool(declared["use_reliability_gated_coarse_fusion"]),
        "use_dual_observation_feedback": bool(declared["use_dual_observation_feedback"]),
        "num_feedback_steps": int(declared["num_feedback_steps"]),
        "share_feedback_weights": bool(declared["share_feedback_weights"]),
        "feedback_residual_scale": float(declared["feedback_residual_scale"]),
        "feedback_observation_mode": str(declared["feedback_observation_mode"]),
    }


def model_variant_label(cfg: dict[str, Any]) -> str:
    model_name = str(cfg.get("model", "DCSRNet"))
    if model_name == "DCSROperatorHypothesisNet":
        return str(cfg.get("model_variant", "b10_operator_hypothesis_guided"))
    declared = declared_model_config_from_yaml(cfg)
    if declared.get("use_dual_observation_feedback"):
        mode = str(declared.get("feedback_observation_mode", "full")).strip().lower()
        if mode == "zero_backprojection":
            return str(cfg.get("model_variant", "b10_postrefine_control_zero_backprojection"))
        return str(cfg.get("model_variant", "b10_dcrf2_dual_observation_feedback"))
    if declared.get("use_reliability_gated_coarse_fusion"):
        return "b9_reliable_coarse"
    if declared["use_signed_residual_guidance"] or declared["use_msi_detail_guidance"]:
        return "b8_signed_detail_guided"
    return "b7_no_guidance"


def b10_kwargs_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    dcfg = cfg.get("data", {}) or {}
    marg = _marg(cfg)
    return {
        "hsi_channels": int(_cfg_get(cfg, "hsi_channels", default=dcfg.get("hsi_channels", 103))),
        "msi_channels": int(_cfg_get(cfg, "msi_channels", default=dcfg.get("msi_channels", 3))),
        "scale": int(_cfg_get(cfg, "scale", default=dcfg.get("scale", 4))),
        "base_channels": int(marg.get("base_channels", 64)),
        "init_channels": int(marg.get("init_channels", 32)),
        "norm": str(marg.get("norm", "gn")),
        "refine_scale": float(marg.get("refine_scale", 0.1)),
    }


def effective_training_config_snapshot(cfg: dict[str, Any], model: DCSRNet) -> dict[str, Any]:
    """Full effective training record: YAML + resolution + runtime module attributes."""
    dcfg = cfg.get("data", {}) or {}
    tcfg = cfg.get("train", {}) or {}
    declared = declared_model_config_from_yaml(cfg)
    effective = effective_model_attributes(model)
    resolution = model_config_resolution_report(cfg)

    def _match(key: str) -> bool:
        return float(declared[key]) == float(effective[key])

    return {
        "model_variant": model_variant_label(cfg),
        "setting": str(cfg.get("setting", "")),
        "dataset": dcfg.get("dataset"),
        "data_root": dcfg.get("root"),
        "split_mode": dcfg.get("split_mode"),
        "eval_crop_policy": dcfg.get("eval_crop_policy"),
        "patch_size": dcfg.get("patch_size"),
        "num_train_patches": dcfg.get("num_train_patches"),
        "seed": cfg.get("seed"),
        "epochs": tcfg.get("epochs"),
        "batch_size": tcfg.get("batch_size"),
        "lr": tcfg.get("lr"),
        "weight_decay": tcfg.get("weight_decay"),
        "save_dir": tcfg.get("save_dir"),
        "base_channels": declared["base_channels"],
        "use_signed_residual_guidance": declared["use_signed_residual_guidance"],
        "use_msi_detail_guidance": declared["use_msi_detail_guidance"],
        "use_reliability_gated_coarse_fusion": declared["use_reliability_gated_coarse_fusion"],
        "declared_model": declared,
        "effective_model": effective,
        "resolution_report": resolution,
        "strict_nonblind_scales_match": _match("delta_phi_scale") and _match("delta_k_scale"),
    }


def save_effective_config(
    run_dir: Union[str, Path],
    cfg: dict[str, Any],
    model: DCSRNet,
) -> dict[str, Any]:
    """Write ``effective_config.json`` under the run directory."""
    snap = effective_training_config_snapshot(cfg, model)
    path = Path(run_dir) / "effective_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(snap, f, indent=2)
    return snap


def log_effective_config(
    cfg: dict[str, Any],
    model: DCSRNet,
    *,
    emit: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    """Print/log key effective parameters at training (or audit) startup."""
    snap = effective_training_config_snapshot(cfg, model)
    res = snap["resolution_report"]
    lines = [
        "=== Effective model / degradation config ===",
        f"model_variant: {snap['model_variant']}",
        f"dataset: {snap.get('dataset')}  eval_crop_policy: {snap.get('eval_crop_policy')}",
        f"seed: {snap.get('seed')}  epochs: {snap.get('epochs')}",
        f"base_channels: {snap['base_channels']}",
        f"use_signed_residual_guidance: {snap['use_signed_residual_guidance']}",
        f"use_msi_detail_guidance: {snap['use_msi_detail_guidance']}",
        (
            f"delta_phi_scale: declared={snap['declared_model']['delta_phi_scale']} "
            f"effective={snap['effective_model']['delta_phi_scale']} "
            f"({res['delta_phi_scale']['source']})"
        ),
        (
            f"delta_k_scale: declared={snap['declared_model']['delta_k_scale']} "
            f"effective={snap['effective_model']['delta_k_scale']} "
            f"({res['delta_k_scale']['source']})"
        ),
        f"res_scale (module): {snap['effective_model']['res_scale']}",
        f"strict_nonblind_scales_match: {snap['strict_nonblind_scales_match']}",
    ]
    for line in lines:
        if emit is not None:
            emit(line)
        else:
            print(line)
    return snap


def model_args_used_snapshot(model: DCSRNet, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Backward-compatible snapshot: constructor kwargs + selected runtime fields."""
    router_last = model.router[-1]
    assert isinstance(router_last, nn.Conv2d)
    snap = dict(kwargs)
    snap["delta_k_scale"] = float(model.delta_k_scale)
    snap["delta_phi_scale"] = float(model.delta_phi_scale)
    snap["res_scale"] = float(model.res_scale)
    snap["router_bias"] = [float(x) for x in router_last.bias.detach().cpu().tolist()]
    snap["use_guidance"] = bool(model.use_guidance)
    return snap


def build_model_from_config(
    cfg: dict[str, Any],
    device: Optional[Union[torch.device, str]] = None,
    *,
    enforce_nonblind: bool = True,
    model_kwargs_overrides: Optional[dict[str, Any]] = None,
) -> nn.Module:
    """Instantiate model from YAML (DCSRNet or DCSROperatorHypothesisNet)."""
    model_name = str(cfg.get("model", "DCSRNet"))
    if enforce_nonblind:
        assert_nonblind_standard_config(cfg)
    kwargs = model_kwargs_from_config(cfg)
    if model_kwargs_overrides:
        kwargs.update(model_kwargs_overrides)
    model = DCSRNet(**kwargs)
    if device is not None:
        model = model.to(device)
    return model


def strict_load_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Strict checkpoint load; raise before mutating weights if keys mismatch."""
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    missing = sorted(model_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_keys)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint strict load failed (key mismatch). "
            f"missing_keys={missing} unexpected_keys={unexpected}"
        )
    model.load_state_dict(state_dict, strict=True)
    return {
        "strict_load_success": True,
        "checkpoint_missing_keys": [],
        "checkpoint_unexpected_keys": [],
    }


def load_checkpoint_payload(
    model: DCSRNet,
    ckpt_path: Union[str, Any],
    device: Optional[Union[torch.device, str]] = None,
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    """Load ``payload['model']`` with strict=True; return epoch, metrics, load report."""
    map_loc = device if device is not None else "cpu"
    payload = torch.load(str(ckpt_path), map_location=map_loc, weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"checkpoint must be a dict with 'model' key: {ckpt_path}")
    state = payload["model"]
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint['model'] must be a state dict: {ckpt_path}")
    load_report = strict_load_state_dict(model, state)
    epoch = int(payload.get("epoch", -1))
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    return epoch, metrics, load_report
