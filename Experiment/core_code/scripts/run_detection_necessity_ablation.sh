#!/bin/bash
# Detection necessity ablation: does the collapse detector actually help?
#
# Compares 4 replay class selection strategies:
# 1. no_detection_all_replay: replay from ALL 178 classes (no detector needed)
# 2. detected_replay: replay from detector-discovered collapse+absorber classes
# 3. oracle_replay: replay from oracle collapse+absorber classes
# 4. random_detected: replay from random subset of same size as detected set
#
# All use margin selection (no absorber info), budget=1000, 5 seeds, strict eval.
#
# Usage: bash scripts/run_detection_necessity_ablation.sh

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="configs/eval_tls22.yaml"
CHECKPOINT="outputs/tls22_cnn/best_model.pt"
REF_PERIOD="M-2022-4"
TGT_PERIOD="M-2022-12"
BASE_OUT="outputs/detection_necessity_ablation"
NUM_SEEDS=5
EVAL_COLLAPSE="56,163,174,48,38,69,104,47,66,10,109,26"

ORACLE_COLLAPSE="56,163,174,48,38,69,104,47,66,10,109,26"
ORACLE_ABSORBER="96,46,2,14,45,105,2,5,71,156,71,13"
ORACLE_STABLE="8,15,44,57,59,62,64,76,94,98,99,107,113,119,128,130,131,132,144,145"

# Get detected classes from existing detection results
DET_DIR="outputs/autonomous_5seeds_strict/detection"
if [[ -f "${DET_DIR}/detection_summary.json" ]]; then
    DET_COLLAPSE=$(python3 -c "import json; d=json.load(open('${DET_DIR}/detection_summary.json')); print(','.join(str(c) for c in d.get('detected_collapse',[])))")
    DET_ABSORBER=$(python3 -c "import json; d=json.load(open('${DET_DIR}/detection_summary.json')); print(','.join(str(c) for c in d.get('detected_absorbers',[])))")
else
    echo "ERROR: Detection results not found at ${DET_DIR}. Run diagnostic monitor pipeline first."
    exit 1
fi

echo "Oracle collapse: ${ORACLE_COLLAPSE}"
echo "Detected collapse: ${DET_COLLAPSE}"
echo "Detected absorber: ${DET_ABSORBER}"

run_config() {
    local NAME="$1"
    local REPLAY_MODE="$2"
    local COLLAPSE_ARG="$3"
    local ABSORBER_ARG="$4"
    local OUT="${BASE_OUT}/${NAME}"

    echo ""
    echo "=== ${NAME} (${NUM_SEEDS} seeds) ==="
    for SEED in $(seq 0 $((NUM_SEEDS - 1))); do
        echo "--- Seed ${SEED} ---"
        python scripts/collapse_active_maintenance_tls22.py \
            --config "${CONFIG}" \
            --checkpoint "${CHECKPOINT}" \
            --reference-period "${REF_PERIOD}" \
            --target-period "${TGT_PERIOD}" \
            --strategies "margin" \
            --budgets "1000" \
            --eval-collapse-classes "${EVAL_COLLAPSE}" \
            ${COLLAPSE_ARG} \
            ${ABSORBER_ARG} \
            --replay-mode "${REPLAY_MODE}" \
            --replay-per-class 5 \
            --replay-distill-weight 0.5 \
            --target-repeat 2 \
            --seed "${SEED}" \
            --output-dir "${OUT}/seed_${SEED}"
    done
    python scripts/aggregate_seeds.py --base-dir "${OUT}" --num-seeds "${NUM_SEEDS}"
}

# 1. No detection: replay from all classes
run_config "no_detection_all_replay" "all" \
    "--collapse-classes ${ORACLE_COLLAPSE}" \
    "--absorber-classes ${ORACLE_ABSORBER}"

# 2. Detected replay: use detector output for replay class selection
run_config "detected_replay" "stable_absorber" \
    "--collapse-classes ${DET_COLLAPSE}" \
    "--absorber-classes ${DET_ABSORBER}"

# 3. Oracle replay: use ground truth classes
run_config "oracle_replay" "stable_absorber" \
    "--collapse-classes ${ORACLE_COLLAPSE}" \
    "--absorber-classes ${ORACLE_ABSORBER}"

# 4. Random detected set: random classes of same size as detected
N_DET=$(echo "${DET_COLLAPSE}" | tr ',' '\n' | wc -l)
RANDOM_CLASSES=$(python3 -c "
import random; random.seed(42)
all_cls = list(range(178))
exclude = [${ORACLE_COLLAPSE}]
candidates = [c for c in all_cls if c not in exclude]
selected = random.sample(candidates, min(${N_DET}, len(candidates)))
print(','.join(str(c) for c in selected))
")
echo ""
echo "Random 'detected' classes (n=${N_DET}): ${RANDOM_CLASSES}"
run_config "random_detected_replay" "stable_absorber" \
    "--collapse-classes ${RANDOM_CLASSES}" \
    "--absorber-classes ${RANDOM_CLASSES}"

echo ""
echo "=== Detection Necessity Ablation Complete ==="
echo "Results in ${BASE_OUT}/"
echo ""
echo "Compare with:"
for d in no_detection_all_replay detected_replay oracle_replay random_detected_replay; do
    echo "  ${d}:"
    grep "margin.*1000" "${BASE_OUT}/${d}/aggregated_mean_std.csv" 2>/dev/null || echo "    (not found)"
done
