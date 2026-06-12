# Collapse-Aware Active Maintenance Summary

## Replay Setting

- replay mode: `all`
- replay samples: `890`
- target repeat: `2`

## Static Baseline

- macro-F1: `0.6286`
- collapsed-class macro-F1: `0.0276`
- stable-class macro-F1: `0.9028`

## Best Strategy Per Budget

| budget | strategy | macro-F1 | collapsed F1 | stable F1 | selected collapsed | selected absorber preds |
|---:|---|---:|---:|---:|---:|---:|
| 200 | absorber_margin_balanced | 0.6440 | 0.2179 | 0.8943 | 22 | 200 |
| 500 | absorber_margin_balanced | 0.6690 | 0.3345 | 0.8916 | 61 | 500 |
| 1000 | absorber_margin | 0.6834 | 0.4262 | 0.8913 | 76 | 1000 |

## Best Run Per Strategy

| strategy | budget | macro-F1 | collapsed F1 | stable F1 | selected collapsed |
|---|---:|---:|---:|---:|---:|
| absorber_margin | 1000 | 0.6834 | 0.4262 | 0.8913 | 76 |
| absorber_margin_balanced | 1000 | 0.6804 | 0.4073 | 0.8930 | 108 |
| absorber_random | 1000 | 0.6715 | 0.3662 | 0.8947 | 76 |
| margin | 1000 | 0.6862 | 0.2620 | 0.8953 | 39 |
| random | 1000 | 0.6713 | 0.2106 | 0.9029 | 19 |

## Figures

- Budget curve: `outputs/collapse_active_replay_tls22_M-2022-12_all_r5_tr2_distill0.5/active_maintenance_budget_curve.png`
- Selected collapsed samples: `outputs/collapse_active_replay_tls22_M-2022-12_all_r5_tr2_distill0.5/active_maintenance_selected_collapse.png`

## Reading

- A useful active-maintenance strategy should select more collapsed samples than random at the same budget and improve collapsed-class F1 without collapsing stable-class F1.
- Oracle-collapse sampling is an upper bound because it uses target labels for selection.
