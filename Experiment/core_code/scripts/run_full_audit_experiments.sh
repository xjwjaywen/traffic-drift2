#!/bin/bash
# ==============================================================================
# Full Audit Experiment Suite
# Addresses all critical/major findings from code audit.
# Run from: /data/xjw/traffic-drift2/Experiment/core_code/
#
# Usage:
#   bash scripts/run_full_audit_experiments.sh [phase]
#   Phases: fix | train | eval | aggregate | all
#
# Prerequisites:
#   conda activate traffic-ncde
#   4x NVIDIA L20 available
# ==============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

PHASE="${1:-all}"

# ──────────────────────────────────────────────────────────────────────────────
# Phase 0: Apply code fixes (audit findings)
# ──────────────────────────────────────────────────────────────────────────────
apply_fixes() {
    echo "=========================================="
    echo "Phase 0: Applying code fixes"
    echo "=========================================="
    # LEGACY: script removed — apply_audit_fixes.py no longer exists
    # python scripts/apply_audit_fixes.py
    echo "Skipped. (apply_audit_fixes.py removed)"
    echo ""
}

# ──────────────────────────────────────────────────────────────────────────────
# Phase 1: Train 3 base models with different training seeds
#   Uses 3 GPUs in parallel (GPU 0,1,2)
# ──────────────────────────────────────────────────────────────────────────────
train_multi_seed() {
    echo "=========================================="
    echo "Phase 1: Training 3 base models (seeds 0,1,2)"
    echo "=========================================="

    for SEED in 0 1 2; do
        OUTPUT="outputs/tls22_cnn_trainseed${SEED}"
        if [ -f "${OUTPUT}/best_model.pt" ]; then
            echo "--- Train seed ${SEED}: already exists, skipping ---"
        else
            echo "--- Training seed ${SEED} on GPU ${SEED} ---"
            CUDA_VISIBLE_DEVICES=${SEED} python train.py \
                --config configs/train_tls22_cnn.yaml \
                --output-dir "${OUTPUT}" \
                --seed "${SEED}" &
        fi
    done
    wait
    echo "All 3 models trained."
    echo ""
}

# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: Run CARE + baselines for each training seed
#   For each base model: 3 fine-tuning seeds × 3 strategies × 3 budgets
#   Plus: static baseline evaluation
# ──────────────────────────────────────────────────────────────────────────────
eval_multi_seed() {
    echo "=========================================="
    echo "Phase 2: Evaluating CARE across training seeds"
    echo "=========================================="

    STRATEGIES="random,margin,absorber_margin"
    BUDGETS="200,500,1000"
    CONFIG="configs/eval_tls22.yaml"

    for TSEED in 0 1 2; do
        CHECKPOINT="outputs/tls22_cnn_trainseed${TSEED}/best_model.pt"
        if [ ! -f "${CHECKPOINT}" ]; then
            echo "ERROR: ${CHECKPOINT} not found. Run phase 'train' first."
            exit 1
        fi

        # 2a. Static baseline evaluation
        echo "--- Train seed ${TSEED}: static baseline ---"
        CUDA_VISIBLE_DEVICES=${TSEED} python scripts/eval_baselines_with_groups.py \
            --config "${CONFIG}" \
            --checkpoint "${CHECKPOINT}" \
            --test-period M-2022-12 \
            --output-dir "outputs/multiseed_audit/trainseed${TSEED}/static" &

        # Wait a bit so data loading doesn't clash
        sleep 5
    done
    wait

    # 2b. CARE experiments: 3 train seeds × 3 fine-tune seeds
    # Run sequentially per GPU to avoid OOM; use 3 GPUs in parallel
    for TSEED in 0 1 2; do
        CHECKPOINT="outputs/tls22_cnn_trainseed${TSEED}/best_model.pt"
        OUTPUT_BASE="outputs/multiseed_audit/trainseed${TSEED}/care"

        echo "--- Train seed ${TSEED}: CARE with 3 FT seeds (GPU ${TSEED}) ---"
        (
            for FTSEED in 0 1 2; do
                echo "  Train seed ${TSEED}, FT seed ${FTSEED}"
                CUDA_VISIBLE_DEVICES=${TSEED} python scripts/collapse_active_maintenance_tls22.py \
                    --config "${CONFIG}" \
                    --checkpoint "${CHECKPOINT}" \
                    --strategies "${STRATEGIES}" \
                    --budgets "${BUDGETS}" \
                    --replay-mode stable_absorber \
                    --replay-per-class 5 \
                    --replay-distill-weight 0.5 \
                    --target-repeat 2 \
                    --seed "${FTSEED}" \
                    --output-dir "${OUTPUT_BASE}/seed_${FTSEED}"
            done
        ) &
    done
    wait
    echo "All CARE evaluations complete."
    echo ""

    # 2c. Re-run TTA baselines with period reset (fair comparison)
    echo "--- Re-running TTA baselines with period reset on original model ---"
    CUDA_VISIBLE_DEVICES=3 python scripts/eval_baselines_with_groups.py \
        --config "${CONFIG}" \
        --checkpoint outputs/tls22_cnn/best_model.pt \
        --test-period M-2022-12 \
        --output-dir "outputs/multiseed_audit/baselines_reset_M12"
    echo ""

    # 2d. Complete Badge baseline (missing seeds 2,3,4)
    echo "--- Completing Badge baseline (seeds 2,3,4) ---"
    if [ -f "scripts/run_al_baselines.sh" ]; then
        for BSEED in 2 3 4; do
            BADGE_OUT="outputs/al_baselines_strict/seed_${BSEED}"
            if [ -d "${BADGE_OUT}" ] && ls "${BADGE_OUT}"/results_*.csv >/dev/null 2>&1; then
                echo "  Badge seed ${BSEED}: already exists, skipping"
            else
                echo "  Badge seed ${BSEED} on GPU 3"
                CUDA_VISIBLE_DEVICES=3 python scripts/collapse_active_maintenance_tls22.py \
                    --config "${CONFIG}" \
                    --checkpoint outputs/tls22_cnn/best_model.pt \
                    --strategies "badge" \
                    --budgets "200,500,1000" \
                    --replay-mode stable_absorber \
                    --replay-per-class 5 \
                    --replay-distill-weight 0.5 \
                    --target-repeat 2 \
                    --seed "${BSEED}" \
                    --output-dir "${BADGE_OUT}" || echo "  Badge seed ${BSEED} failed (badge may require extra deps)"
            fi
        done
    fi
    echo ""
}

# ──────────────────────────────────────────────────────────────────────────────
# Phase 3: Aggregate all results
# ──────────────────────────────────────────────────────────────────────────────
aggregate_results() {
    echo "=========================================="
    echo "Phase 3: Aggregating results"
    echo "=========================================="

    # 3a. Per-training-seed aggregation
    for TSEED in 0 1 2; do
        CARE_DIR="outputs/multiseed_audit/trainseed${TSEED}/care"
        if [ -d "${CARE_DIR}/seed_0" ]; then
            echo "--- Aggregating train seed ${TSEED} (3 FT seeds) ---"
            python scripts/aggregate_seeds.py \
                --base-dir "${CARE_DIR}" \
                --num-seeds 3
        fi
    done

    # 3b. Cross-training-seed summary
    echo ""
    echo "--- Cross-training-seed summary ---"
    python scripts/summarize_multiseed_audit.py \
        --base-dir "outputs/multiseed_audit" \
        --num-train-seeds 3

    # 3c. Re-aggregate existing experiments with fixed ddof=1
    echo ""
    echo "--- Re-aggregating existing results with corrected std (ddof=1) ---"
    for DIR in outputs/care_5seeds_strict_cnn outputs/ablation_strict/* outputs/al_baselines_strict outputs/care_quic22_strict; do
        if [ -d "${DIR}/seed_0" ]; then
            echo "  Re-aggregating ${DIR}"
            python scripts/aggregate_seeds.py --base-dir "${DIR}" --num-seeds 5 2>/dev/null || true
        fi
    done
    echo ""
}

# ──────────────────────────────────────────────────────────────────────────────
# Main dispatch
# ──────────────────────────────────────────────────────────────────────────────
case "${PHASE}" in
    fix)       apply_fixes ;;
    train)     train_multi_seed ;;
    eval)      eval_multi_seed ;;
    aggregate) aggregate_results ;;
    all)
        apply_fixes
        train_multi_seed
        eval_multi_seed
        aggregate_results
        echo "=========================================="
        echo "ALL PHASES COMPLETE"
        echo "Results in: outputs/multiseed_audit/"
        echo "=========================================="
        ;;
    *)
        echo "Usage: $0 [fix|train|eval|aggregate|all]"
        exit 1
        ;;
esac
