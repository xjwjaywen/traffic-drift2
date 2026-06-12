# Collapse-Aware Active Maintenance Summary

## Replay Setting

- replay mode: `all`
- replay samples: `890`
- target repeat: `2`

## Static Baseline

- macro-F1: `0.6836`
- collapsed-class macro-F1: `0.2581`
- stable-class macro-F1: `0.9090`

## Best Strategy Per Budget

| budget | strategy | macro-F1 | collapsed F1 | stable F1 | selected collapsed | selected absorber preds |
|---:|---|---:|---:|---:|---:|---:|
| 200 | absorber_margin | 0.6527 | 0.3199 | 0.8475 | 12 | 200 |
| 500 | absorber_margin | 0.6509 | 0.3411 | 0.8558 | 22 | 500 |
| 1000 | random | 0.6886 | 0.3917 | 0.8766 | 19 | 113 |

## Best Run Per Strategy

| strategy | budget | macro-F1 | collapsed F1 | stable F1 | selected collapsed |
|---|---:|---:|---:|---:|---:|
| absorber_margin | 1000 | 0.6634 | 0.3681 | 0.8495 | 35 |
| absorber_random | 1000 | 0.6586 | 0.3461 | 0.8541 | 44 |
| margin | 1000 | 0.6901 | 0.3437 | 0.8692 | 27 |
| random | 1000 | 0.6886 | 0.3917 | 0.8766 | 19 |

## Figures

- Budget curve: `outputs/collapse_active_replay_tls22_M-2022-10_all_r5_tr2/active_maintenance_budget_curve.png`
- Selected collapsed samples: `outputs/collapse_active_replay_tls22_M-2022-10_all_r5_tr2/active_maintenance_selected_collapse.png`

## Reading

- A useful active-maintenance strategy should select more collapsed samples than random at the same budget and improve collapsed-class F1 without collapsing stable-class F1.
- Oracle-collapse sampling is an upper bound because it uses target labels for selection.
