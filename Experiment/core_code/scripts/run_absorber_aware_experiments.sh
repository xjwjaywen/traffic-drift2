#!/bin/bash
# Absorber-Aware Sampling experiments (P5)
# Compares absorber_aware against margin, badge, random, entropy, coreset
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
# P5a: Absorber-Aware + all-class replay (fair comparison with P1/P4)
# ============================================================
run_p5a() {
    echo "================================================================"
    echo "P5a: Absorber-Aware + all-class replay (${NUM_SEEDS} seeds)"
    echo "================================================================"
    local OUT="outputs/absorber_aware_allreplay_5seeds"

    for SEED in $(seq 0 $((NUM_SEEDS - 1))); do
        echo "--- Seed ${SEED} ---"
        python scripts/collapse_active_maintenance_tls22.py \
            --config "${CONFIG}" --checkpoint "${CHECKPOINT}" \
            --reference-period "${REF_PERIOD}" --target-period M-2022-12 \
            --strategies "absorber_aware" --budgets "1000" \
            --eval-collapse-classes "${EVAL_COLLAPSE}" \
            --replay-mode all --replay-per-class 5 --target-repeat 2 \
            --replay-distill-weight 0.5 --distill-temperature 2.0 \
            --seed "${SEED}" \
            --output-dir "${OUT}/seed_${SEED}"
    done
    python scripts/aggregate_seeds.py --base-dir "${OUT}" --num-seeds "${NUM_SEEDS}"
    echo ""
    echo "P5a results:"
    grep "absorber_aware.*1000" "${OUT}/aggregated_mean_std.csv"
}

# ============================================================
# P5b: All strategies head-to-head (same all-class replay conditions)
# ============================================================
run_p5b() {
    echo "================================================================"
    echo "P5b: All strategies comparison under unified conditions (${NUM_SEEDS} seeds)"
    echo "================================================================"
    local OUT="outputs/strategy_comparison_unified"
    local STRATEGIES="random,entropy,margin,coreset,badge,absorber_aware"

    for SEED in $(seq 0 $((NUM_SEEDS - 1))); do
        echo "--- Seed ${SEED} ---"
        python scripts/collapse_active_maintenance_tls22.py \
            --config "${CONFIG}" --checkpoint "${CHECKPOINT}" \
            --reference-period "${REF_PERIOD}" --target-period M-2022-12 \
            --strategies "${STRATEGIES}" --budgets "1000" \
            --eval-collapse-classes "${EVAL_COLLAPSE}" \
            --replay-mode all --replay-per-class 5 --target-repeat 2 \
            --replay-distill-weight 0.5 --distill-temperature 2.0 \
            --seed "${SEED}" \
            --output-dir "${OUT}/seed_${SEED}"
    done
    python scripts/aggregate_seeds.py --base-dir "${OUT}" --num-seeds "${NUM_SEEDS}"
    echo ""
    echo "P5b results (all strategies @ B=1000, all-class replay):"
    cat "${OUT}/aggregated_mean_std.csv"
}

# ============================================================
# P5c: Budget sweep for absorber-aware vs margin vs badge
# ============================================================
run_p5c() {
    echo "================================================================"
    echo "P5c: Budget sweep (${NUM_SEEDS} seeds)"
    echo "================================================================"
    local OUT="outputs/budget_sweep_absorber_aware"
    local STRATEGIES="margin,badge,absorber_aware"
    local BUDGETS="100,200,500,1000,2000"

    for SEED in $(seq 0 $((NUM_SEEDS - 1))); do
        echo "--- Seed ${SEED} ---"
        python scripts/collapse_active_maintenance_tls22.py \
            --config "${CONFIG}" --checkpoint "${CHECKPOINT}" \
            --reference-period "${REF_PERIOD}" --target-period M-2022-12 \
            --strategies "${STRATEGIES}" --budgets "${BUDGETS}" \
            --eval-collapse-classes "${EVAL_COLLAPSE}" \
            --replay-mode all --replay-per-class 5 --target-repeat 2 \
            --replay-distill-weight 0.5 --distill-temperature 2.0 \
            --seed "${SEED}" \
            --output-dir "${OUT}/seed_${SEED}"
    done
    python scripts/aggregate_seeds.py --base-dir "${OUT}" --num-seeds "${NUM_SEEDS}"
    echo ""
    echo "P5c results (budget sweep):"
    cat "${OUT}/aggregated_mean_std.csv"
}

# ============================================================
# P5d: Multi-period evaluation for absorber-aware
# ============================================================
run_p5d() {
    echo "================================================================"
    echo "P5d: Multi-period absorber-aware (M7, M9, M11, M12)"
    echo "================================================================"
    local OUT="outputs/absorber_aware_multiperiod"

    for TGT in M-2022-7 M-2022-9 M-2022-11 M-2022-12; do
        local SLUG=$(echo "${TGT}" | sed 's/M-2022-/M/')
        echo "--- ${SLUG} ---"
        for SEED in 0 1 2; do
            python scripts/collapse_active_maintenance_tls22.py \
                --config "${CONFIG}" --checkpoint "${CHECKPOINT}" \
                --reference-period "${REF_PERIOD}" --target-period "${TGT}" \
                --strategies "margin,absorber_aware" --budgets "1000" \
                --eval-collapse-classes "${EVAL_COLLAPSE}" \
                --replay-mode all --replay-per-class 5 --target-repeat 2 \
                --replay-distill-weight 0.5 --distill-temperature 2.0 \
                --seed "${SEED}" \
                --output-dir "${OUT}/${SLUG}/seed_${SEED}"
        done
        python scripts/aggregate_seeds.py --base-dir "${OUT}/${SLUG}" --num-seeds 3
    done
}

# ============================================================
# Main
# ============================================================
EXPS="${1:-all}"

case "${EXPS}" in
    p5a) run_p5a ;;
    p5b) run_p5b ;;
    p5c) run_p5c ;;
    p5d) run_p5d ;;
    all)
        run_p5a
        run_p5b
        run_p5c
        run_p5d
        echo ""
        echo "========================================"
        echo "ALL ABSORBER-AWARE EXPERIMENTS COMPLETE"
        echo "========================================"
        ;;
    *)
        echo "Usage: $0 [p5a|p5b|p5c|p5d|all]"
        exit 1
        ;;
esac
