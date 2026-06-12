# Reject-Option Ablation

- Reference period: `M-2022-4`
- Target period: `M-2022-12`

## Static Baseline

- macro-F1: `0.6286`
- collapsed-class macro-F1: `0.0255`
- stable-class macro-F1: `0.9028`

## Best Rules

| rule | threshold | coverage | collapsed reject | stable false reject | absorber error reduction | accepted macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| absorber_distance | ref_p70 | 0.956 | 0.089 | 0.003 | 0.240 | 0.6339 |
| absorber_proto_disagree | ref_p70 | 0.963 | 0.082 | 0.003 | 0.222 | 0.6347 |
| confidence | ref_p30 | 0.514 | 0.895 | 0.180 | 0.975 | 0.8201 |
| hybrid | conf_p30_margin_p30_dist_p70 | 0.485 | 0.904 | 0.187 | 0.996 | 0.8125 |
| margin | ref_p30 | 0.517 | 0.858 | 0.183 | 0.973 | 0.8167 |
| prototype_distance | ref_p70 | 0.646 | 0.259 | 0.106 | 0.240 | 0.6771 |

## Figures

- Trade-off: `outputs/reject_option_ablation_tls22_m12/reject_tradeoff_collapsed_vs_stable.png`
- Best-rule bar chart: `outputs/reject_option_ablation_tls22_m12/reject_best_rules_summary.png`

## Interpretation Guide

- A useful reject rule should reject many collapsed samples while keeping stable false rejections low.
- If confidence or margin performs poorly but prototype/absorber-risk performs better, softmax confidence alone is insufficient.
- If all rules either miss collapsed samples or reject stable samples heavily, post-hoc reject alone is not enough.
