# Reproduction notes

## Formal DCRF-Net configuration

| Dataset | Config | Feedback steps T | Epochs | Seed |
|---------|--------|------------------|--------|------|
| PaviaU | `configs/paviau/dcrf_net.yaml` | 2 | 400 | 42 |
| Chikusei | `configs/chikusei/dcrf_net.yaml` | 2 | 400 | 42 |
| Houston 2018 | `configs/houston2018/dcrf_net.yaml` | 2 | 400 | 42 |

Model variant: `b10_dcrf2_dual_observation_feedback` (`DCSRNet` with dual-observation feedback).

## Training entry

```bash
python scripts/train.py --config configs/<dataset>/dcrf_net.yaml --data-root /path/to/baseline_export/<dataset>
```

## Evaluation entry

```bash
python scripts/evaluate.py --config configs/<dataset>/dcrf_net.yaml --data-root /path/to/baseline_export/<dataset> --checkpoint /path/to/best_psnr.pth --split test
```

## Metrics

`utils.metrics.evaluate_reconstruction` reports PSNR, SSIM, SAM (degrees), RMSE, and ERGAS
with `data_range` from `meta.json` (default 1.0).

## Checkpoints

Training saves `best_psnr.pth`, `best_sam.pth`, `best_ergas.pth`, and `latest.pth` under `save_dir`.
This public package does **not** include pretrained weights.

## T-step ablation

Edit `model_args.num_feedback_steps` in the YAML config (e.g. 1, 2, 3, 4) and train or load a matching checkpoint.

## Inference timing

```bash
python scripts/evaluate.py ... --timing
```

Reports single forward-pass wall time for one full scene (after optional CUDA synchronization).
