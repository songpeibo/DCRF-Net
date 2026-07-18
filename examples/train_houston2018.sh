#!/usr/bin/env bash
set -euo pipefail
python scripts/train.py --config configs/houston2018/dcrf_net.yaml --data-root /path/to/baseline_export/houston18
