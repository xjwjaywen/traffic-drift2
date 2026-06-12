# Collapse-Aware Active Maintenance Summary

## Static Baseline

- macro-F1: `0.6286`
- collapsed-class macro-F1: `0.0276`
- stable-class macro-F1: `0.9028`

## Best Strategy Per Budget

| budget | strategy | macro-F1 | collapsed F1 | stable F1 | selected collapsed | selected absorber preds |
|---:|---|---:|---:|---:|---:|---:|
| 50 | oracle_collapse_random | 0.6263 | 0.0533 | 0.8997 | 50 | 21 |
| 100 | oracle_collapse_random | 0.6121 | 0.1030 | 0.8813 | 100 | 46 |
| 200 | absorber_random | 0.5886 | 0.1063 | 0.8592 | 18 | 200 |
| 500 | absorber_margin | 0.5841 | 0.1570 | 0.8481 | 37 | 500 |
| 1000 | absorber_margin | 0.5888 | 0.2244 | 0.8201 | 76 | 1000 |

## Best Run Per Strategy

| strategy | budget | macro-F1 | collapsed F1 | stable F1 | selected collapsed |
|---|---:|---:|---:|---:|---:|
| absorber_distance | 1000 | 0.5082 | 0.0309 | 0.8311 | 23 |
| absorber_margin | 1000 | 0.5888 | 0.2244 | 0.8201 | 76 |
| absorber_proto_disagree | 1000 | 0.5085 | 0.0316 | 0.8294 | 23 |
| absorber_random | 1000 | 0.5448 | 0.2067 | 0.8058 | 76 |
| entropy | 1000 | 0.5111 | 0.0578 | 0.8101 | 17 |
| hybrid_risk | 1000 | 0.5038 | 0.0476 | 0.7965 | 12 |
| margin | 1000 | 0.6388 | 0.1321 | 0.8400 | 39 |
| oracle_collapse_random | 100 | 0.6121 | 0.1030 | 0.8813 | 100 |
| random | 1000 | 0.6137 | 0.1289 | 0.8649 | 19 |

## Figures

- Budget curve: `outputs/collapse_active_maintenance_tls22_M-2022-12/active_maintenance_budget_curve.png`
- Selected collapsed samples: `outputs/collapse_active_maintenance_tls22_M-2022-12/active_maintenance_selected_collapse.png`

## Reading

- A useful active-maintenance strategy should select more collapsed samples than random at the same budget and improve collapsed-class F1 without collapsing stable-class F1.
- Oracle-collapse sampling is an upper bound because it uses target labels for selection.
