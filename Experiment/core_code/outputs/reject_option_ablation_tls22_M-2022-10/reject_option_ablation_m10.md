# Reject-Option Ablation

- Reference period: `M-2022-4`
- Target period: `M-2022-10`

## Static Baseline

- macro-F1: `0.6836`
- collapsed-class macro-F1: `0.2393`
- stable-class macro-F1: `0.9090`

## Best Rules

| rule | threshold | coverage | collapsed reject | stable false reject | absorber error reduction | accepted macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| absorber_distance | ref_p70 | 0.959 | 0.078 | 0.002 | 0.275 | 0.6874 |
| absorber_proto_disagree | ref_p70 | 0.967 | 0.070 | 0.002 | 0.245 | 0.6883 |
| confidence | ref_p30 | 0.547 | 0.827 | 0.169 | 0.976 | 0.8728 |
| hybrid | conf_p30_margin_p30_dist_p70 | 0.517 | 0.836 | 0.176 | 0.998 | 0.8633 |
| margin | ref_p30 | 0.549 | 0.808 | 0.173 | 0.974 | 0.8696 |
| prototype_distance | ref_p70 | 0.657 | 0.253 | 0.103 | 0.275 | 0.7272 |

## Figures

- Trade-off: `outputs/reject_option_ablation_tls22_M-2022-10/reject_tradeoff_collapsed_vs_stable.png`
- Best-rule bar chart: `outputs/reject_option_ablation_tls22_M-2022-10/reject_best_rules_summary.png`

## Interpretation Guide

- A useful reject rule should reject many collapsed samples while keeping stable false rejections low.
- If confidence or margin performs poorly but prototype/absorber-risk performs better, softmax confidence alone is insufficient.
- If all rules either miss collapsed samples or reject stable samples heavily, post-hoc reject alone is not enough.
