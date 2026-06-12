# Reject-Option Ablation

- Reference period: `M-2022-4`
- Target period: `M-2022-7`

## Static Baseline

- macro-F1: `0.7402`
- collapsed-class macro-F1: `0.4647`
- stable-class macro-F1: `0.8322`

## Best Rules

| rule | threshold | coverage | collapsed reject | stable false reject | absorber error reduction | accepted macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| absorber_distance | ref_p70 | 0.957 | 0.027 | 0.002 | 0.107 | 0.7423 |
| absorber_proto_disagree | ref_p70 | 0.965 | 0.023 | 0.002 | 0.089 | 0.7431 |
| confidence | ref_p30 | 0.609 | 0.756 | 0.168 | 0.968 | 0.9003 |
| hybrid | conf_p30_margin_p30_dist_p70 | 0.569 | 0.769 | 0.177 | 0.999 | 0.8921 |
| margin | ref_p30 | 0.610 | 0.735 | 0.175 | 0.968 | 0.8988 |
| prototype_distance | ref_p70 | 0.668 | 0.184 | 0.103 | 0.107 | 0.7807 |

## Figures

- Trade-off: `outputs/reject_option_ablation_tls22_M-2022-7/reject_tradeoff_collapsed_vs_stable.png`
- Best-rule bar chart: `outputs/reject_option_ablation_tls22_M-2022-7/reject_best_rules_summary.png`

## Interpretation Guide

- A useful reject rule should reject many collapsed samples while keeping stable false rejections low.
- If confidence or margin performs poorly but prototype/absorber-risk performs better, softmax confidence alone is insufficient.
- If all rules either miss collapsed samples or reject stable samples heavily, post-hoc reject alone is not enough.
