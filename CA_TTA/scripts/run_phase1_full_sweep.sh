#!/usr/bin/env bash
# CA-TTA Phase 1 — full sweep with all the methodological fixes:
#   - Constraint-aware smoothing (no noise on direction channel)
#   - Clean / smoothed / certified accuracy all reported
#   - Multi-seed for statistical significance
#   - Multi-sigma to test robustness across noise levels
#   - Lambda sweep for CA-TTA
#   - Loss-type ablation (MACER vs stability)
#
# Run from Experiment/core_code/ :
#   bash ../../CA_TTA/scripts/run_phase1_full_sweep.sh

set -e
cd "$(dirname "$0")/../.."
REPO_ROOT="$PWD"
cd "$REPO_ROOT/Experiment/core_code"

DATASETS=("quic22" "tls22")
SEEDS=(0 1 2)
SIGMAS=(0.10 0.25 0.50)
CA_LAMBDAS=(0.5 1.0 2.0 5.0)
CA_LOSS_TYPES=("macer" "stability")

OUT_DIR="$REPO_ROOT/CA_TTA/outputs/phase1_adapt_then_certify"
mkdir -p "$OUT_DIR"

run_one () {
    local dataset=$1
    local method=$2
    local sigma=$3
    local seed=$4
    local extra_tag=$5
    local extra_args=$6
    local ckpt="outputs/${dataset}_cnn/best_model.pt"
    local cfg="configs/eval_${dataset}.yaml"
    local tag="seed${seed}_sig${sigma}${extra_tag}"
    local log="${OUT_DIR}/${dataset}_${method}_${tag}.log"

    echo "==> [${dataset}/${method}] ${tag}"
    python ../../CA_TTA/scripts/phase1_adapt_then_certify.py \
        --config "$cfg" \
        --checkpoint "$ckpt" \
        --method "$method" \
        --sigma "$sigma" \
        --seed "$seed" \
        --output-suffix "$tag" \
        --max-samples-per-period 500 \
        $extra_args \
        2>&1 | tee "$log"
}

# === Block A: baselines × seeds × sigmas ===
echo "================================================================"
echo "Block A: baselines × seeds × sigmas (3 seeds × 3 sigmas × 4 methods × 2 datasets = 72 runs)"
echo "================================================================"
for dataset in "${DATASETS[@]}"; do
    for sigma in "${SIGMAS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            for method in static tent supervised_norm ft_head; do
                run_one "$dataset" "$method" "$sigma" "$seed" "" ""
            done
        done
    done
done

# === Block B: CA-TTA × seeds × sigmas × lambdas (default loss = macer) ===
echo "================================================================"
echo "Block B: CA-TTA lambda sweep (3 seeds × 3 sigmas × 4 lambdas × 2 datasets = 72 runs)"
echo "================================================================"
for dataset in "${DATASETS[@]}"; do
    for sigma in "${SIGMAS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            for lam in "${CA_LAMBDAS[@]}"; do
                run_one "$dataset" "ca_tta" "$sigma" "$seed" \
                    "_lam${lam}_macer_oc1" \
                    "--ca-lambda $lam --ca-loss-type macer"
            done
        done
    done
done

# === Block C: CA-TTA loss-type ablation (lam=1.0, sigma=0.25) ===
echo "================================================================"
echo "Block C: CA-TTA loss-type ablation (3 seeds × 2 loss types × 2 datasets = 12 runs)"
echo "================================================================"
for dataset in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        # stability loss
        run_one "$dataset" "ca_tta" "0.25" "$seed" \
            "_lam1.0_stability_oc1" \
            "--ca-lambda 1.0 --ca-loss-type stability"
        # only_correct off
        run_one "$dataset" "ca_tta" "0.25" "$seed" \
            "_lam1.0_macer_oc0" \
            "--ca-lambda 1.0 --ca-loss-type macer --ca-no-only-correct"
    done
done

echo
echo "==> Sweep complete."
echo "==> Run: python $REPO_ROOT/CA_TTA/scripts/phase1_aggregate.py"
