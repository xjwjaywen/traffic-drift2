# Collapse-Aware Active Maintenance Summary

## Replay Setting

- replay mode: `stable`
- replay samples: `500`
- target repeat: `2`

## Static Baseline

- macro-F1: `0.6286`
- collapsed-class macro-F1: `0.0276`
- stable-class macro-F1: `0.9028`

## Best Strategy Per Budget

| budget | strategy | macro-F1 | collapsed F1 | stable F1 | selected collapsed | selected absorber preds |
|---:|---|---:|---:|---:|---:|---:|
| 50 | oracle_collapse_random | 0.4997 | 0.1157 | 0.5751 | 50 | 21 |
| 100 | oracle_collapse_random | 0.4847 | 0.1169 | 0.5865 | 100 | 46 |
| 200 | absorber_random | 0.5109 | 0.1592 | 0.6041 | 18 | 200 |
| 500 | absorber_random | 0.4967 | 0.2473 | 0.6511 | 37 | 500 |
| 1000 | absorber_margin | 0.5443 | 0.3384 | 0.7491 | 76 | 1000 |

## Best Run Per Strategy

| strategy | budget | macro-F1 | collapsed F1 | stable F1 | selected collapsed |
|---|---:|---:|---:|---:|---:|
| absorber_margin | 1000 | 0.5443 | 0.3384 | 0.7491 | 76 |
| absorber_random | 1000 | 0.4879 | 0.2702 | 0.6967 | 76 |
| margin | 1000 | 0.6087 | 0.2332 | 0.8057 | 39 |
| oracle_collapse_random | 100 | 0.4847 | 0.1169 | 0.5865 | 100 |
| random | 1000 | 0.5839 | 0.1934 | 0.7667 | 19 |

## Figures

- Budget curve: `outputs/collapse_active_replay_tls22_M-2022-12_stable_r25_tr2/active_maintenance_budget_curve.png`
- Selected collapsed samples: `outputs/collapse_active_replay_tls22_M-2022-12_stable_r25_tr2/active_maintenance_selected_collapse.png`

## Reading

- A useful active-maintenance strategy should select more collapsed samples than random at the same budget and improve collapsed-class F1 without collapsing stable-class F1.
- Oracle-collapse sampling is an upper bound because it uses target labels for selection.
