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
| 200 | absorber_margin_balanced | 0.6456 | 0.2142 | 0.8955 | 22 | 200 |
| 500 | absorber_margin_balanced | 0.6688 | 0.3283 | 0.8912 | 61 | 500 |
| 1000 | absorber_margin | 0.6826 | 0.4237 | 0.8941 | 76 | 1000 |

## Best Run Per Strategy

| strategy | budget | macro-F1 | collapsed F1 | stable F1 | selected collapsed |
|---|---:|---:|---:|---:|---:|
| absorber_margin | 1000 | 0.6826 | 0.4237 | 0.8941 | 76 |
| absorber_margin_balanced | 1000 | 0.6798 | 0.4030 | 0.8921 | 108 |
| margin | 1000 | 0.6869 | 0.2652 | 0.8953 | 39 |

## Figures

- Budget curve: `outputs/collapse_active_replay_tls22_M-2022-12_all_r5_tr2_distill05/active_maintenance_budget_curve.png`
- Selected collapsed samples: `outputs/collapse_active_replay_tls22_M-2022-12_all_r5_tr2_distill05/active_maintenance_selected_collapse.png`

## Reading

- A useful active-maintenance strategy should select more collapsed samples than random at the same budget and improve collapsed-class F1 without collapsing stable-class F1.
- Oracle-collapse sampling is an upper bound because it uses target labels for selection.
