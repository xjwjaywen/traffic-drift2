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
| 200 | absorber_margin | 0.7449 | 0.5993 | 0.8360 | 8 | 200 |
| 500 | absorber_margin_balanced | 0.7591 | 0.6480 | 0.8415 | 31 | 500 |
| 1000 | absorber_margin_balanced | 0.7679 | 0.6825 | 0.8397 | 57 | 1000 |

## Best Run Per Strategy

| strategy | budget | macro-F1 | collapsed F1 | stable F1 | selected collapsed |
|---|---:|---:|---:|---:|---:|
| absorber_margin | 1000 | 0.7619 | 0.6362 | 0.8391 | 43 |
| absorber_margin_balanced | 1000 | 0.7679 | 0.6825 | 0.8397 | 57 |
| absorber_random | 1000 | 0.7542 | 0.6066 | 0.8422 | 36 |
| margin | 1000 | 0.7769 | 0.6635 | 0.8488 | 38 |
| random | 1000 | 0.7756 | 0.6423 | 0.8529 | 18 |

## Figures

- Budget curve: `outputs/collapse_active_replay_tls22_M-2022-7_all_r5_tr2_distill0.5/active_maintenance_budget_curve.png`
- Selected collapsed samples: `outputs/collapse_active_replay_tls22_M-2022-7_all_r5_tr2_distill0.5/active_maintenance_selected_collapse.png`

## Reading

- A useful active-maintenance strategy should select more collapsed samples than random at the same budget and improve collapsed-class F1 without collapsing stable-class F1.
- Oracle-collapse sampling is an upper bound because it uses target labels for selection.
