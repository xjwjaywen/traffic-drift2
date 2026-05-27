#!/usr/bin/env bash
set -euo pipefail

# QUIC22 targeted channel augmentation averaging.
# This differs from MVFC: no parameter update, and each run perturbs only one
# selected channel/region so we can see which augmentation is helpful.

CONFIG="${CONFIG:-configs/eval_quic22.yaml}"
CHECKPOINT="${CHECKPOINT:-outputs/quic22_cnn/best_model.pt}"
PERIODS="${PERIODS:-W-2022-46 W-2022-47}"
SETTINGS="${SETTINGS:-raw size_noise_0.02 size_noise_0.05 ipt_noise_0.05 ipt_noise_0.10 direction_dropout_0.02 direction_front_dropout_0.02 direction_front_dropout_0.05 packet_mask_0.02}"
NUM_VIEWS="${NUM_VIEWS:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/quic22_channel_augmentation}"

CMD=(
  python scripts/quic22_channel_aug_eval.py
  --config "${CONFIG}"
  --checkpoint "${CHECKPOINT}"
  --num-views "${NUM_VIEWS}"
  --output-dir "${OUTPUT_DIR}"
  --periods
)
for period in ${PERIODS}; do
  CMD+=("${period}")
done
CMD+=(--settings)
for setting in ${SETTINGS}; do
  CMD+=("${setting}")
done

"${CMD[@]}"

