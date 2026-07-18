#!/usr/bin/env bash
set -euo pipefail
python scripts/train.py --config configs/chikusei/dcrf_net.yaml --data-root /path/to/baseline_export/chikusei
