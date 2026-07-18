# DCRF-Net (Dual-Channel Residual Feedback Network)

Official public release of the DCRF-Net implementation for hyperspectral and
multispectral image fusion under a strict non-blind degradation protocol.

DCRF-Net couples a guided initialization backbone with a shared dual-observation
residual feedback cell iterated for `T` steps (`num_feedback_steps` in config).

## Repository structure

| Path | Description |
|------|-------------|
| `models/` | DCRF-Net (`DCSRNet`) and feedback blocks |
| `datasets/` | `baseline_export` and synthetic loaders |
| `utils/` | Degradation operators, losses, metrics, training helpers |
| `configs/` | Formal non-blind configs for PaviaU, Chikusei, Houston 2018 |
| `scripts/` | `train.py`, `evaluate.py`, `infer.py` |
| `baselines/` | Baseline **model code only** (no training scripts) |
| `docs/` | Dataset, reproduction, degradation, and baseline notes |
| `examples/` | Example shell commands |

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate dcrf-net
```

## Dataset preparation

See `docs/DATASETS.md`. This package does **not** include raw benchmark data.

## Training

```bash
python scripts/train.py \
  --config configs/paviau/dcrf_net.yaml \
  --data-root /path/to/baseline_export/paviau
```

Checkpoints are written to the `save_dir` in the config (default under `outputs/runs/`).

## Evaluation

```bash
python scripts/evaluate.py \
  --config configs/paviau/dcrf_net.yaml \
  --data-root /path/to/baseline_export/paviau \
  --checkpoint /path/to/best_psnr.pth \
  --split test \
  --timing
```

## Inference

```bash
python scripts/infer.py \
  --config configs/paviau/dcrf_net.yaml \
  --checkpoint /path/to/best_psnr.pth \
  --scene-npz /path/to/baseline_export/paviau/test/scene_00.npz \
  --out /path/to/pred.npy
```

## Configuration

Formal paper settings use `configs/<dataset>/dcrf_net.yaml` with `num_feedback_steps: 2`.
Set `model_args.num_feedback_steps` to 1–4 for T-step ablations.

## Degradation protocol

See `docs/DEGRADATION_PROTOCOL.md`.

## Baseline model code

Comparison methods are provided as instantiable model definitions only.
See `baselines/README.md` and per-method `SOURCE.md` files.

## Citation

See `CITATION.cff`.

## License

See `LICENSE` or `LICENSE_NOTICE.md`. Third-party baseline notices: `THIRD_PARTY_NOTICES.md`.

## Acknowledgments

Baseline implementations are derived from the original authors' repositories listed in
`baselines/*/SOURCE.md`.
