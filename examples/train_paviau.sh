#!/usr/bin/env bash
set -euo pipefail
python scripts/train.py --config configs/paviau/dcrf_net.yaml --data-root /path/to/baseline_export/paviau
