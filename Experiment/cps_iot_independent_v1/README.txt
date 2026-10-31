Independent IoT flow feasibility audit

Source: https://github.com/shay-fi/CPSIoTSec25-labelShiftAdaptation-dataset
The authors provide 12 homes, chronological train/test CSVs, 22 numeric flow
features, device labels, and flow-end timestamps. This dataset was assembled
for label-shift adaptation, so it tests cross-dataset robustness but does not
by itself prove a method repairs TMA-style conditional drift or class collapse.

The dataset license limits use to research and restricts redistribution.
Keep data under /data, outside this code repository. Do not commit CSV files.

Server commands:
  git clone --depth 1 \
    https://github.com/shay-fi/CPSIoTSec25-labelShiftAdaptation-dataset.git \
    /data/xjw/tma-care-data/cps_iotsec25
  cd /data/xjw/traffic-drift2
  python -u Experiment/cps_iot_independent_v1/audit.py \
    /data/xjw/tma-care-data/cps_iotsec25
  cat /data/xjw/tma-care-data/cps_iot_independent_audit_v1/REPORT.txt

The audit reads training labels only. It reads test timestamps and hashes,
but never held-out test labels. It checks feature schema and time order across
all 12 homes and writes train class counts and coverage. After this audit,
freeze a source/query split and metric plan before any test predictions.

Development-only baselines (does not open test_full.csv):
  python -u Experiment/cps_iot_independent_v1/development.py \
    /data/xjw/tma-care-data/cps_iotsec25
  cat /data/xjw/tma-care-data/cps_iot_development_v1/REPORT.txt

This run trains one source MLP per home, then uses paired 1000-label queries
for target-only, original CARE and frozen-feature RF heads over repair seeds
0-4. The 12 development sets may be used to design the next method; the
published test labels remain reserved for final evaluation. See
PROTOCOL.md for the frozen split, methods and metrics.
