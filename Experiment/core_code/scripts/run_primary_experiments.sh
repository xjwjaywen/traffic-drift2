#!/bin/bash
# Primary experiments for paper submission (P1-P4)
# Run from Experiment/core_code/
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="configs/eval_tls22.yaml"
CHECKPOINT="outputs/tls22_cnn/best_model.pt"
REF_PERIOD="M-2022-4"
EVAL_COLLAPSE="56,163,174,48,38,69,104,47,66,10,109,26"
STABLE="8,15,44,57,59,62,64,76,94,98,99,107,113,119,128,130,131,132,144,145"
NUM_SEEDS=5

# ============================================================
# P1: Autonomous + all-class replay (canonical main result)
# Detection as trigger → margin selection → all-class replay
# ============================================================
run_p1() {
    echo "================================================================"
    echo "P1: Autonomous pipeline with all-class replay (${NUM_SEEDS} seeds)"
    echo "================================================================"
    local OUT="outputs/autonomous_allreplay_5seeds"

    # Step 1: Detection (reuse existing if available)
    local DET_DIR="outputs/autonomous_5seeds_strict/detection"
    if [[ ! -f "${DET_DIR}/detection_summary.json" ]]; then
        echo "Running detection..."
        python scripts/unsupervised_collapse_detection.py \
            --config "${CONFIG}" --checkpoint "${CHECKPOINT}" \
            --reference-period "${REF_PERIOD}" --target-period M-2022-12 \
            --output-dir "${OUT}/detection"
        DET_DIR="${OUT}/detection"
    else
        echo "Reusing existing detection from ${DET_DIR}"
        mkdir -p "${OUT}"
        cp -r "${DET_DIR}" "${OUT}/detection" 2>/dev/null || true
    fi

    # Step 2: CARE with all-class replay
    for SEED in $(seq 0 $((NUM_SEEDS - 1))); do
        echo "--- Seed ${SEED} ---"
        python scripts/collapse_active_maintenance_tls22.py \
            --config "${CONFIG}" --checkpoint "${CHECKPOINT}" \
            --reference-period "${REF_PERIOD}" --target-period M-2022-12 \
            --strategies "margin" --budgets "1000" \
            --eval-collapse-classes "${EVAL_COLLAPSE}" \
            --replay-mode all --replay-per-class 5 --target-repeat 2 \
            --replay-distill-weight 0.5 --distill-temperature 2.0 \
            --seed "${SEED}" \
            --output-dir "${OUT}/care/seed_${SEED}"
    done
    python scripts/aggregate_seeds.py --base-dir "${OUT}/care" --num-seeds "${NUM_SEEDS}"
    echo ""
    echo "P1 results:"
    grep "margin.*1000" "${OUT}/care/aggregated_mean_std.csv"
}

# ============================================================
# P2: Detection trigger evaluation across periods
# For each period, check if detection would have triggered correctly
# ============================================================
run_p2() {
    echo "================================================================"
    echo "P2: Detection trigger evaluation (M4→M7, M4→M9, M4→M11, M4→M12)"
    echo "================================================================"
    local OUT="outputs/detection_trigger_eval"
    mkdir -p "${OUT}"

    for TGT in M-2022-7 M-2022-9 M-2022-11 M-2022-12; do
        local SLUG=$(echo "${TGT}" | sed 's/M-2022-/M/')
        echo "--- ${SLUG} ---"
        python scripts/unsupervised_collapse_detection.py \
            --config "${CONFIG}" --checkpoint "${CHECKPOINT}" \
            --reference-period "${REF_PERIOD}" --target-period "${TGT}" \
            --output-dir "${OUT}/${SLUG}"
    done

    echo ""
    echo "P2 summary:"
    python3 -c "
import json, os
print(f'{'Period':>8} {'Detected':>10} {'Precision':>10} {'Recall':>8} {'F1':>6} {'Trigger?':>10}')
print('-' * 60)
for slug in ['M7','M9','M11','M12']:
    path = 'outputs/detection_trigger_eval/${slug}/detection_summary.json'  # will be substituted
    path = f'${OUT}/{slug}/detection_summary.json'
    if not os.path.exists(path):
        continue
    d = json.load(open(path))
    n = len(d.get('detected_collapse', []))
    p = d.get('precision', 0)
    r = d.get('recall', 0)
    f1 = d.get('f1', 0)
    trigger = 'YES' if n > 0 else 'NO'
    print(f'{slug:>8} {n:>10} {p:>10.3f} {r:>8.3f} {f1:>6.3f} {trigger:>10}')
"
}

# ============================================================
# P3: Fair replay budget ablation
# all-class k=5 → 178×5=890 samples
# detected-class k=5 → ~30×5=150 samples
# Fair comparison: match total replay count
# ============================================================
run_p3() {
    echo "================================================================"
    echo "P3: Fair replay budget ablation (${NUM_SEEDS} seeds)"
    echo "================================================================"
    local OUT="outputs/fair_replay_budget"

    # Config A: all-class, k=5 → 890 replay samples (current best)
    echo "--- Config A: all-class k=5 (890 replay) ---"
    for SEED in $(seq 0 $((NUM_SEEDS - 1))); do
        python scripts/collapse_active_maintenance_tls22.py \
            --config "${CONFIG}" --checkpoint "${CHECKPOINT}" \
            --reference-period "${REF_PERIOD}" --target-period M-2022-12 \
            --strategies "margin" --budgets "1000" \
            --eval-collapse-classes "${EVAL_COLLAPSE}" \
            --replay-mode all --replay-per-class 5 --target-repeat 2 \
            --replay-distill-weight 0.5 \
            --seed "${SEED}" \
            --output-dir "${OUT}/all_k5/seed_${SEED}"
    done
    python scripts/aggregate_seeds.py --base-dir "${OUT}/all_k5" --num-seeds "${NUM_SEEDS}"

    # Config B: detected-class, k=30 → ~30×30=900 replay samples (matched budget)
    echo "--- Config B: detected-class k=30 (matched ~900 replay) ---"
    for SEED in $(seq 0 $((NUM_SEEDS - 1))); do
        python scripts/collapse_active_maintenance_tls22.py \
            --config "${CONFIG}" --checkpoint "${CHECKPOINT}" \
            --reference-period "${REF_PERIOD}" --target-period M-2022-12 \
            --strategies "margin" --budgets "1000" \
            --eval-collapse-classes "${EVAL_COLLAPSE}" \
            --replay-mode stable_absorber --replay-per-class 30 --target-repeat 2 \
            --replay-distill-weight 0.5 \
            --seed "${SEED}" \
            --output-dir "${OUT}/detected_k30/seed_${SEED}"
    done
    python scripts/aggregate_seeds.py --base-dir "${OUT}/detected_k30" --num-seeds "${NUM_SEEDS}"

    # Config C: all-class, k=1 → 178 replay samples (matched to detected k=5 ~150)
    echo "--- Config C: all-class k=1 (178 replay, matched to detected-class budget) ---"
    for SEED in $(seq 0 $((NUM_SEEDS - 1))); do
        python scripts/collapse_active_maintenance_tls22.py \
            --config "${CONFIG}" --checkpoint "${CHECKPOINT}" \
            --reference-period "${REF_PERIOD}" --target-period M-2022-12 \
            --strategies "margin" --budgets "1000" \
            --eval-collapse-classes "${EVAL_COLLAPSE}" \
            --replay-mode all --replay-per-class 1 --target-repeat 2 \
            --replay-distill-weight 0.5 \
            --seed "${SEED}" \
            --output-dir "${OUT}/all_k1/seed_${SEED}"
    done
    python scripts/aggregate_seeds.py --base-dir "${OUT}/all_k1" --num-seeds "${NUM_SEEDS}"

    echo ""
    echo "P3 results (margin@1000):"
    for d in all_k5 detected_k30 all_k1; do
        echo "  ${d}:"
        grep "margin.*1000" "${OUT}/${d}/aggregated_mean_std.csv" 2>/dev/null || echo "    (not found)"
    done
}

# ============================================================
# P4: BADGE in monitor-assisted pipeline with all-class replay
# ============================================================
run_p4() {
    echo "================================================================"
    echo "P4: BADGE + all-class replay (${NUM_SEEDS} seeds)"
    echo "================================================================"
    local OUT="outputs/badge_allreplay_5seeds"

    for SEED in $(seq 0 $((NUM_SEEDS - 1))); do
        echo "--- Seed ${SEED} ---"
        python scripts/collapse_active_maintenance_tls22.py \
            --config "${CONFIG}" --checkpoint "${CHECKPOINT}" \
            --reference-period "${REF_PERIOD}" --target-period M-2022-12 \
            --strategies "badge" --budgets "1000" \
            --eval-collapse-classes "${EVAL_COLLAPSE}" \
            --replay-mode all --replay-per-class 5 --target-repeat 2 \
            --replay-distill-weight 0.5 --distill-temperature 2.0 \
            --seed "${SEED}" \
            --output-dir "${OUT}/seed_${SEED}"
    done
    python scripts/aggregate_seeds.py --base-dir "${OUT}" --num-seeds "${NUM_SEEDS}"
    echo ""
    echo "P4 results:"
    grep "badge.*1000" "${OUT}/aggregated_mean_std.csv"
}

# ============================================================
# Main: run selected experiments
# ============================================================
EXPS="${1:-all}"

case "${EXPS}" in
    p1) run_p1 ;;
    p2) run_p2 ;;
    p3) run_p3 ;;
    p4) run_p4 ;;
    all)
        run_p1
        run_p2
        run_p3
        run_p4
        echo ""
        echo "========================================"
        echo "ALL PRIMARY EXPERIMENTS COMPLETE"
        echo "========================================"
        ;;
    *)
        echo "Usage: $0 [p1|p2|p3|p4|all]"
        exit 1
        ;;
esac
