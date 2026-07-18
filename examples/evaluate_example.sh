#!/usr/bin/env bash
set -euo pipefail
python scripts/evaluate.py --config configs/paviau/dcrf_net.yaml --data-root /path/to/baseline_export/paviau --checkpoint /path/to/best_psnr.pth --split test --timing
