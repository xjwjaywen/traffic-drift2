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
| 200 | absorber_margin | 0.6921 | 0.3456 | 0.8877 | 12 | 200 |
| 500 | absorber_margin_balanced | 0.7061 | 0.4039 | 0.8955 | 25 | 500 |
| 1000 | absorber_margin_balanced | 0.7204 | 0.4923 | 0.8944 | 49 | 1000 |

## Best Run Per Strategy

| strategy | budget | macro-F1 | collapsed F1 | stable F1 | selected collapsed |
|---|---:|---:|---:|---:|---:|
| absorber_margin | 1000 | 0.7100 | 0.4193 | 0.9027 | 35 |
| absorber_margin_balanced | 1000 | 0.7204 | 0.4923 | 0.8944 | 49 |
| absorber_random | 1000 | 0.7133 | 0.4499 | 0.9044 | 44 |
| margin | 1000 | 0.7253 | 0.3924 | 0.9008 | 27 |
| random | 1000 | 0.7220 | 0.4344 | 0.9058 | 19 |

## Figures

- Budget curve: `outputs/collapse_active_replay_tls22_M-2022-10_all_r5_tr2_distill0.5/active_maintenance_budget_curve.png`
- Selected collapsed samples: `outputs/collapse_active_replay_tls22_M-2022-10_all_r5_tr2_distill0.5/active_maintenance_selected_collapse.png`

## Reading

- A useful active-maintenance strategy should select more collapsed samples than random at the same budget and improve collapsed-class F1 without collapsing stable-class F1.
- Oracle-collapse sampling is an upper bound because it uses target labels for selection.
