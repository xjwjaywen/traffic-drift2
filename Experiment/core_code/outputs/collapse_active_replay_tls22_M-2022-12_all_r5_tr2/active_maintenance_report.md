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
| 200 | absorber_random | 0.5970 | 0.1845 | 0.8381 | 18 | 200 |
| 500 | absorber_random | 0.6078 | 0.2814 | 0.8454 | 37 | 500 |
| 1000 | absorber_margin | 0.6399 | 0.3500 | 0.8559 | 76 | 1000 |

## Best Run Per Strategy

| strategy | budget | macro-F1 | collapsed F1 | stable F1 | selected collapsed |
|---|---:|---:|---:|---:|---:|
| absorber_margin | 1000 | 0.6399 | 0.3500 | 0.8559 | 76 |
| absorber_random | 1000 | 0.6175 | 0.2978 | 0.8513 | 76 |
| margin | 1000 | 0.6604 | 0.2426 | 0.8745 | 39 |
| random | 1000 | 0.6435 | 0.2140 | 0.8710 | 19 |

## Figures

- Budget curve: `outputs/collapse_active_replay_tls22_M-2022-12_all_r5_tr2/active_maintenance_budget_curve.png`
- Selected collapsed samples: `outputs/collapse_active_replay_tls22_M-2022-12_all_r5_tr2/active_maintenance_selected_collapse.png`

## Reading

- A useful active-maintenance strategy should select more collapsed samples than random at the same budget and improve collapsed-class F1 without collapsing stable-class F1.
- Oracle-collapse sampling is an upper bound because it uses target labels for selection.
