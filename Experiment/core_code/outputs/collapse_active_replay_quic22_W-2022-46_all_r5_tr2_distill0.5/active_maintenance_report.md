# Collapse-Aware Active Maintenance Summary

## Replay Setting

- replay mode: `all`
- replay samples: `510`
- target repeat: `2`

## Static Baseline

- macro-F1: `0.7135`
- collapsed-class macro-F1: `0.1627`
- stable-class macro-F1: `0.8925`

## Best Strategy Per Budget

| budget | strategy | macro-F1 | collapsed F1 | stable F1 | selected collapsed | selected absorber preds |
|---:|---|---:|---:|---:|---:|---:|
| 200 | absorber_random | 0.7063 | 0.1616 | 0.8907 | 1 | 200 |
| 500 | margin | 0.7419 | 0.1333 | 0.8900 | 0 | 2 |
| 1000 | margin | 0.7489 | 0.1394 | 0.8887 | 0 | 4 |

## Best Run Per Strategy

| strategy | budget | macro-F1 | collapsed F1 | stable F1 | selected collapsed |
|---|---:|---:|---:|---:|---:|
| absorber_margin | 200 | 0.7069 | 0.1068 | 0.8906 | 0 |
| absorber_margin_balanced | 200 | 0.7076 | 0.1058 | 0.8907 | 0 |
| absorber_random | 200 | 0.7063 | 0.1616 | 0.8907 | 1 |
| margin | 1000 | 0.7489 | 0.1394 | 0.8887 | 0 |
| random | 1000 | 0.7477 | 0.1375 | 0.8886 | 0 |

## Figures

- Budget curve: `outputs/collapse_active_replay_quic22_W-2022-46_all_r5_tr2_distill0.5/active_maintenance_budget_curve.png`
- Selected collapsed samples: `outputs/collapse_active_replay_quic22_W-2022-46_all_r5_tr2_distill0.5/active_maintenance_selected_collapse.png`

## Reading

- A useful active-maintenance strategy should select more collapsed samples than random at the same budget and improve collapsed-class F1 without collapsing stable-class F1.
- Oracle-collapse sampling is an upper bound because it uses target labels for selection.
