#!/bin/bash
# Run all 5 paper experiments for reviewer response.
#
# Experiments:
#   1. Threshold sensitivity (τ = 0.05/0.10/0.15)
#   2. Support threshold (≥50/100/200)        — combined with #1
#   3. Detection weight comparison (manual/equal/adaptive)
#   4. Drift mechanism scatter plot
#   5. Label cost-benefit curve (B = 50/100/200/500/1000)
#
# Usage from Experiment/core_code/:
#   bash scripts/paper_experiments/run_all.sh
#
# Expected total time: ~30-40 minutes on a single GPU.

set -euo pipefail
cd "$(dirname "$0")/../.."

CONFIG="configs/eval_tls22.yaml"
CHECKPOINT="outputs/tls22_cnn/best_model.pt"
REF="M-2022-4"
TARGET="M-2022-12"
EVAL_COLLAPSE="56,163,174,48,38,69,104,47,66,10,109,26"
BASE="outputs/paper_experiments"
SEEDS=5

CARE_ARGS="--replay-mode all --replay-per-class 5 --target-repeat 2 --replay-distill-weight 0.5"

mkdir -p "${BASE}"

echo "=========================================="
echo "  Paper Experiments Runner"
echo "=========================================="

# -----------------------------------------------
# Experiment 1+2: Threshold & Support Sensitivity
# -----------------------------------------------
echo ""
echo "=== Experiment 1+2: Threshold & Support Sensitivity ==="
for SEED in $(seq 0 $((SEEDS - 1))); do
    OUTDIR="${BASE}/threshold_sensitivity/seed_${SEED}"
    if [ -f "${OUTDIR}/sensitivity_results.csv" ]; then
        echo "  Seed ${SEED} exists, skipping"
        continue
    fi
    echo "  Seed ${SEED}..."
    python scripts/paper_experiments/threshold_support_sensitivity.py \
        --config "${CONFIG}" \
        --checkpoint "${CHECKPOINT}" \
        --reference-period "${REF}" \
        --target-period "${TARGET}" \
        --budget 1000 \
        --seed "${SEED}" \
        --output-dir "${OUTDIR}" \
        --recall-thresholds "0.05,0.10,0.15" \
        --support-thresholds "50,100,200"
done

# Aggregate threshold sensitivity across seeds
if [ -d "${BASE}/threshold_sensitivity" ]; then
    python -c "
import csv, os, json
from collections import defaultdict
import numpy as np

base = '${BASE}/threshold_sensitivity'
all_rows = defaultdict(list)
for seed in range(${SEEDS}):
    path = os.path.join(base, f'seed_{seed}', 'sensitivity_results.csv')
    if not os.path.exists(path):
        print(f'ERROR: missing seed {seed} at {path}')
        exit(1)
    with open(path) as f:
        for row in csv.DictReader(f):
            all_rows[row['config']].append(row)

if not all_rows:
    print('No sensitivity results found')
    exit()

metrics = ['static_collapse_f1', 'care_collapse_f1', 'delta_collapse_f1',
           'static_overall_macro_f1', 'care_overall_macro_f1',
           'static_stable_f1', 'care_stable_f1', 'care_collapsed_count']

agg = []
for config, rows in sorted(all_rows.items()):
    entry = {
        'config': config,
        'recall_threshold': rows[0]['recall_threshold'],
        'support_threshold': rows[0]['support_threshold'],
        'n_collapse_classes': rows[0]['n_collapse_classes'],
        'n_seeds': len(rows),
    }
    for m in metrics:
        vals = [float(r[m]) for r in rows if r.get(m, '') != '']
        if vals:
            entry[f'{m}_mean'] = f'{np.mean(vals):.4f}'
            entry[f'{m}_std'] = f'{np.std(vals, ddof=1):.4f}' if len(vals) > 1 else '0.0000'
    agg.append(entry)

out_path = os.path.join(base, 'aggregated_sensitivity.csv')
with open(out_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
    w.writeheader()
    w.writerows(agg)
print(f'Aggregated {len(all_rows)} configs x {${SEEDS}} seeds -> {out_path}')
for e in agg:
    print(f\"  {e['config']:<20} #col={e['n_collapse_classes']:>3} \"
          f\"static_col={e.get('static_collapse_f1_mean',''):>7} \"
          f\"care_col={e.get('care_collapse_f1_mean',''):>7}±{e.get('care_collapse_f1_std',''):>6} \"
          f\"Δ={e.get('delta_collapse_f1_mean',''):>7}\")
"
fi

# -----------------------------------------------
# Experiment 3: Detection Weight Comparison
# -----------------------------------------------
echo ""
echo "=== Experiment 3: Detection Weight Comparison ==="
OUTDIR="${BASE}/detection_weights"
if [ -f "${OUTDIR}/weight_comparison.csv" ]; then
    echo "  Already exists, skipping"
else
    python scripts/paper_experiments/detection_weight_comparison.py \
        --config "${CONFIG}" \
        --checkpoint "${CHECKPOINT}" \
        --reference-period "${REF}" \
        --output-dir "${OUTDIR}"
fi

# -----------------------------------------------
# Experiment 4: Drift Mechanism Scatter Plot
# -----------------------------------------------
echo ""
echo "=== Experiment 4: Drift Mechanism Scatter Plot ==="
OUTDIR="${BASE}/drift_scatter"
if [ -f "${OUTDIR}/drift_scatter_data.csv" ]; then
    echo "  Already exists, skipping"
else
    python scripts/paper_experiments/drift_scatter.py \
        --config "${CONFIG}" \
        --checkpoint "${CHECKPOINT}" \
        --reference-period "${REF}" \
        --target-period "${TARGET}" \
        --output-dir "${OUTDIR}"
fi

# -----------------------------------------------
# Experiment 5: Budget Curve (B=50,100 补充)
# -----------------------------------------------
echo ""
echo "=== Experiment 5: Label Cost-Benefit Curve ==="
for SEED in $(seq 0 $((SEEDS - 1))); do
    OUTDIR="${BASE}/budget_curve/seed_${SEED}"
    if [ -f "${OUTDIR}/results_by_budget.csv" ]; then
        echo "  Seed ${SEED} exists, skipping"
        continue
    fi
    echo "  Seed ${SEED}..."
    python scripts/collapse_active_maintenance_tls22.py \
        --config "${CONFIG}" \
        --checkpoint "${CHECKPOINT}" \
        --reference-period "${REF}" \
        --target-period "${TARGET}" \
        --strategies "margin" \
        --budgets "50,100,200,500,1000,2000,4000,8000" \
        --eval-collapse-classes "${EVAL_COLLAPSE}" \
        --seed "${SEED}" \
        --output-dir "${OUTDIR}" \
        ${CARE_ARGS}
done

# Aggregate budget curve
if [ -d "${BASE}/budget_curve" ]; then
    python scripts/aggregate_seeds.py \
        --base-dir "${BASE}/budget_curve" \
        --num-seeds "${SEEDS}" \
        --require-complete
fi

echo ""
echo "=========================================="
echo "  All paper experiments complete"
echo "=========================================="
echo "Results in ${BASE}/"
ls -la "${BASE}/" 2>/dev/null
