#!/bin/bash
# Multi-period CARE evaluation: run CARE at M7, M9, M11, M12
# to show how repair effectiveness varies with drift severity.
# Uses auto-discovered absorbers from the preceding period.
#
# Usage: bash scripts/run_multiperiod_care.sh <config> <checkpoint> <output_base>

set -euo pipefail

CONFIG="${1:?Usage: $0 <config> <checkpoint> <output_base>}"
CHECKPOINT="${2:?}"
OUTPUT_BASE="${3:?}"

REFERENCE="M-2022-4"
STRATEGIES="margin"
BUDGETS="200,500,1000"

declare -A PROBE_MAP
PROBE_MAP[M-2022-7]="M-2022-6"
PROBE_MAP[M-2022-9]="M-2022-8"
PROBE_MAP[M-2022-11]="M-2022-10"
PROBE_MAP[M-2022-12]="M-2022-11"

for TARGET in M-2022-7 M-2022-9 M-2022-11 M-2022-12; do
    PROBE="${PROBE_MAP[$TARGET]}"
    TARGET_SLUG=$(echo "${TARGET}" | tr '-' '_')
    PROBE_SLUG=$(echo "${PROBE}" | tr '-' '_')
    PERIOD_DIR="${OUTPUT_BASE}/${TARGET_SLUG}"

    echo "========================================"
    echo "=== Target: ${TARGET} (probe: ${PROBE}) ==="
    echo "========================================"

    # Step 1: Discover absorbers from probe period
    echo "--- Discovering absorbers from ${PROBE} ---"
    python scripts/discover_absorbers.py \
        --config "${CONFIG}" \
        --checkpoint "${CHECKPOINT}" \
        --probe-period "${PROBE}" \
        --recall-threshold 0.1 \
        --top-k-absorbers 10 \
        --output-dir "${PERIOD_DIR}/discovery"

    DISCOVERY_FILE="${PERIOD_DIR}/discovery/discovered_${PROBE_SLUG}.json"
    COLLAPSE_CLASSES=$(python -c "import json; d=json.load(open('${DISCOVERY_FILE}')); print(','.join(str(c) for c in d['collapse_classes']))")
    ABSORBER_CLASSES=$(python -c "import json; d=json.load(open('${DISCOVERY_FILE}')); print(','.join(str(c) for c in d['absorber_classes']))")

    echo "Collapse: ${COLLAPSE_CLASSES:-none}"
    echo "Absorbers: ${ABSORBER_CLASSES:-none}"

    # Step 2: Run CARE with 3 seeds
    for SEED in 0 1 2; do
        echo "--- ${TARGET} seed ${SEED} ---"
        EXTRA_ARGS=""
        if [[ -n "${COLLAPSE_CLASSES}" ]]; then
            EXTRA_ARGS="--collapse-classes ${COLLAPSE_CLASSES}"
        fi
        if [[ -n "${ABSORBER_CLASSES}" ]]; then
            EXTRA_ARGS="${EXTRA_ARGS} --absorber-classes ${ABSORBER_CLASSES}"
        fi

        python scripts/collapse_active_maintenance_tls22.py \
            --config "${CONFIG}" \
            --checkpoint "${CHECKPOINT}" \
            --reference-period "${REFERENCE}" \
            --target-period "${TARGET}" \
            --strategies "${STRATEGIES}" \
            --budgets "${BUDGETS}" \
            ${EXTRA_ARGS} \
            --replay-mode stable_absorber \
            --replay-per-class 5 \
            --replay-distill-weight 0.5 \
            --target-repeat 2 \
            --seed "${SEED}" \
            --output-dir "${PERIOD_DIR}/seed_${SEED}"
    done

    echo "--- Aggregate ${TARGET} ---"
    python scripts/aggregate_seeds.py --base-dir "${PERIOD_DIR}" --num-seeds 3
    echo ""
done

echo "=== All periods complete ==="
