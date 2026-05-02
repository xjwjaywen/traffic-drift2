#!/bin/bash
# Run H0 NCDE feasibility experiment on one visible GPU.
#
# Usage:
#   cd /data/xjw/traffic-drift2/Experiment/core_code
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_h0_ncde.sh

set -e

python train_h0_ncde.py --config configs/h0_quic22_ncde.yaml
