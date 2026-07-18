# Datasets

This release does **not** distribute benchmark imagery. Obtain the datasets from their
original providers and prepare a `baseline_export` layout compatible with
`datasets/baseline_export.py`.

## Supported benchmarks

| Dataset | Config directory | Typical channels (HSI / MSI) |
|---------|------------------|------------------------------|
| PaviaU | `configs/paviau/` | 103 / 3 |
| Chikusei | `configs/chikusei/` | 128 / 3 |
| Houston 2018 | `configs/houston2018/` | 48 / 3 |

## Expected directory layout

```
/path/to/baseline_export/<dataset>/
  meta.json
  train/
    patch_0000.npz
    ...
  val/scene_00.npz
  test/scene_00.npz
```

Each NPZ should provide at least `h` (LR-HSI), `m` (HR-MSI), `phi0` (SRF), `k0` (PSF),
and `gt` for supervised evaluation splits.

## Preprocessing

1. Download the original hyperspectral/multispectral scenes.
2. Build or export matched LR-HSI / HR-MSI pairs with fixed PSF and SRF (non-blind protocol).
3. Point `--data-root` or `data.root` in the YAML config to the prepared directory.

The authors should confirm the exact export recipe used for the paper tables before publishing data preparation scripts.
