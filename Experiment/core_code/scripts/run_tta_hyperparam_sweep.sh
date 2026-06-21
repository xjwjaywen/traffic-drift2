#!/bin/bash
# TTA Hyperparameter Sweep
# Purpose: Show that TTA collapse F1 stays near 0 across reasonable hyperparameters.
# Sweeps: learning rate for all 5 TTA methods (batch_size column in output retained from earlier script but not claimed in paper).
#
# Usage: bash scripts/run_tta_hyperparam_sweep.sh
# Output: outputs/tta_hyperparam_sweep/<lr>_<batch>/baselines_group_metrics.csv

set -euo pipefail
cd "$(dirname "$0")/.."

BASE_CONFIG="configs/eval_tls22.yaml"
CHECKPOINT="outputs/tls22_cnn/best_model.pt"
OUTPUT_BASE="outputs/tta_hyperparam_sweep"

LRS=(1e-5 3e-5 1e-4 3e-4 1e-3)
BATCHES=(64 256)

mkdir -p "$OUTPUT_BASE"

for lr in "${LRS[@]}"; do
  for bs in "${BATCHES[@]}"; do
    tag="lr${lr}_bs${bs}"
    outdir="${OUTPUT_BASE}/${tag}"

    if [ -f "${outdir}/baselines_group_metrics.csv" ]; then
      echo "=== SKIP ${tag} (already exists) ==="
      continue
    fi

    echo "=== Running ${tag} ==="

    # Create temporary config with this lr and batch size
    tmpconfig=$(mktemp /tmp/tta_sweep_XXXXXX.yaml)
    python3 -c "
import yaml
with open('${BASE_CONFIG}') as f:
    cfg = yaml.safe_load(f)
cfg.setdefault('tta', {})['adapt_lr'] = float('${lr}')
cfg.setdefault('tta', {})['adapt_batch_size'] = int('${bs}')
with open('${tmpconfig}', 'w') as f:
    yaml.dump(cfg, f)
"

    python3 scripts/eval_baselines_with_groups.py \
      --config "$tmpconfig" \
      --checkpoint "$CHECKPOINT" \
      --test-period "M-2022-12" \
      --output-dir "$outdir" \
      --methods "static,tent,eata,sar,cotta,note" \
      --seed 42

    rm -f "$tmpconfig"
    echo "=== Done ${tag} ==="
  done
done

echo ""
echo "=== Aggregating results ==="
python3 -c "
import csv, os, glob

base = '${OUTPUT_BASE}'
rows = []
for d in sorted(glob.glob(os.path.join(base, 'lr*_bs*'))):
    tag = os.path.basename(d)
    parts = tag.split('_')
    lr = parts[0].replace('lr','')
    bs = parts[1].replace('bs','')
    csv_path = os.path.join(d, 'baselines_group_metrics.csv')
    if not os.path.exists(csv_path):
        continue
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.append({
                'lr': lr, 'batch_size': bs,
                'method': row['method'],
                'macro_f1': float(row['overall_macro_f1']),
                'collapse_f1': float(row['collapse_macro_f1']),
                'stable_f1': float(row['stable_macro_f1']),
                'collapsed_count': int(float(row['collapsed_count'])),
            })

out_path = os.path.join(base, 'sweep_summary.csv')
with open(out_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['method','lr','batch_size','macro_f1','collapse_f1','stable_f1','collapsed_count'])
    w.writeheader()
    for r in sorted(rows, key=lambda x: (x['method'], x['lr'], x['batch_size'])):
        w.writerow(r)

print(f'Summary saved to {out_path}')
print()
print(f\"{'Method':<10} {'LR':>8} {'BS':>4} {'Macro':>7} {'Col F1':>8} {'Stb F1':>8} {'#Col':>5}\")
print('-'*55)
for r in sorted(rows, key=lambda x: (x['method'], -x['collapse_f1'])):
    print(f\"{r['method']:<10} {r['lr']:>8} {r['batch_size']:>4} {r['macro_f1']:>7.4f} {r['collapse_f1']:>8.4f} {r['stable_f1']:>8.4f} {r['collapsed_count']:>5}\")
"

echo ""
echo "=== Best collapse F1 per method ==="
python3 -c "
import csv
rows = []
with open('${OUTPUT_BASE}/sweep_summary.csv') as f:
    for r in csv.DictReader(f):
        if r['method'] == 'Static': continue
        rows.append(r)

from collections import defaultdict
best = {}
for r in rows:
    m = r['method']
    cf1 = float(r['collapse_f1'])
    if m not in best or cf1 > float(best[m]['collapse_f1']):
        best[m] = r

print(f\"{'Method':<10} {'Best LR':>8} {'BS':>4} {'Col F1':>8} {'Macro':>7}\")
print('-'*42)
for m in sorted(best):
    r = best[m]
    print(f\"{m:<10} {r['lr']:>8} {r['batch_size']:>4} {float(r['collapse_f1']):>8.4f} {float(r['macro_f1']):>7.4f}\")
"
