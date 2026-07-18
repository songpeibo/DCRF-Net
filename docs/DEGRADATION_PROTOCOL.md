# Degradation protocol

## Non-blind setting

Formal configs set `setting: nonblind_standard` with `delta_phi_scale: 0.0` and
`delta_k_scale: 0.0`. Nominal SRF (`phi0`) and PSF (`k0`) tensors from the dataset export
are used without learnable perturbation during training and evaluation.

## Operators

- **Spectral:** `utils.degradation_ops.spectral_degrade` (HR-HSI → synthetic MSI)
- **Spatial:** `utils.degradation_ops.spatial_degrade` (HR-HSI → LR-HSI)

PSF and SRF are loaded from each `baseline_export` sample (`k0`, `phi0`).

## Matched-degradation evaluation

Validation and test use the same nominal operators stored in the export. Eval crops follow
`eval_crop_policy: top_left` in the formal configs.

## Mismatch experiments

Mismatch / robustness experiments from the paper are **not** enabled in the formal public configs.
If implemented in a private branch, they would use alternate dataset roots and are out of scope for this package.

## Real-sensor claims

This code targets the synthetic `baseline_export` protocol. It does not claim support for
arbitrary real-sensor blind fusion without additional data preparation.
