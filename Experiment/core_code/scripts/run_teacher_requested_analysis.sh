#!/usr/bin/env bash
set -euo pipefail

# Run the advisor-requested post-processing:
#   1. optional AdaBN evaluation on a BN checkpoint;
#   2. optional TTA drift-type group evaluation;
#   3. per-class norm/AdaBN summaries;
#   4. compact collapsed-class statistics;
#   5. TTA / normalization / collapse visualizations.
#
# Usage from Experiment/core_code/:
#   bash scripts/run_teacher_requested_analysis.sh
#
# To skip expensive reruns and only regenerate figures:
#   RUN_ADABN=0 RUN_TTA_DRIFT=0 bash scripts/run_teacher_requested_analysis.sh

RUN_ADABN="${RUN_ADABN:-1}"
RUN_TTA_DRIFT="${RUN_TTA_DRIFT:-1}"

if [[ "$RUN_ADABN" == "1" ]]; then
  bash scripts/run_tls22_adabn_drift_ablation.sh
else
  echo "Skipping AdaBN evaluation because RUN_ADABN=0"
fi

if [[ "$RUN_TTA_DRIFT" == "1" ]]; then
  bash scripts/run_tls22_tta_drift_type_ablation.sh
else
  echo "Skipping TTA drift-type evaluation because RUN_TTA_DRIFT=0"
fi

python scripts/summarize_norm_adabn_class_effects.py \
  --output-dir outputs/teacher_result_visuals

python scripts/summarize_teacher_collapse_stats.py \
  --output-dir outputs/teacher_result_visuals

python scripts/visualize_method_inventory_summary.py \
  --output-dir outputs/teacher_result_visuals

python scripts/visualize_teacher_results.py \
  --output-dir outputs/teacher_result_visuals

echo
echo "Advisor-facing summary:"
echo "  outputs/teacher_result_visuals/teacher_result_visuals_summary.md"
echo "  outputs/teacher_result_visuals/collapse_stat_summary.md"
echo "  outputs/teacher_result_visuals/method_inventory_summary.png"
