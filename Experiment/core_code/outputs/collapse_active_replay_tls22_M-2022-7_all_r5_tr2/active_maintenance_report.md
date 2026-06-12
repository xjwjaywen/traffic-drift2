# Collapse-Aware Active Maintenance Summary

## Replay Setting

- replay mode: `all`
- replay samples: `890`
- target repeat: `2`

## Static Baseline

- macro-F1: `0.7402`
- collapsed-class macro-F1: `0.5034`
- stable-class macro-F1: `0.8322`

## Best Strategy Per Budget

| budget | strategy | macro-F1 | collapsed F1 | stable F1 | selected collapsed | selected absorber preds |
|---:|---|---:|---:|---:|---:|---:|
| 200 | absorber_margin | 0.6977 | 0.5480 | 0.8062 | 8 | 200 |
| 500 | random | 0.7226 | 0.5747 | 0.8101 | 8 | 59 |
| 1000 | random | 0.7376 | 0.5990 | 0.8188 | 18 | 116 |

## Best Run Per Strategy

| strategy | budget | macro-F1 | collapsed F1 | stable F1 | selected collapsed |
|---|---:|---:|---:|---:|---:|
| absorber_margin | 500 | 0.7152 | 0.5585 | 0.8086 | 25 |
| absorber_random | 1000 | 0.6980 | 0.5598 | 0.8089 | 36 |
| margin | 1000 | 0.7350 | 0.5451 | 0.8191 | 38 |
| random | 1000 | 0.7376 | 0.5990 | 0.8188 | 18 |

## Figures

- Budget curve: `outputs/collapse_active_replay_tls22_M-2022-7_all_r5_tr2/active_maintenance_budget_curve.png`
- Selected collapsed samples: `outputs/collapse_active_replay_tls22_M-2022-7_all_r5_tr2/active_maintenance_selected_collapse.png`

## Reading

- A useful active-maintenance strategy should select more collapsed samples than random at the same budget and improve collapsed-class F1 without collapsing stable-class F1.
- Oracle-collapse sampling is an upper bound because it uses target labels for selection.
