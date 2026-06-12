# Collapse-Aware Active Maintenance Summary

## Replay Setting

- replay mode: `all`
- replay samples: `510`
- target repeat: `2`

## Static Baseline

- macro-F1: `0.7194`
- collapsed-class macro-F1: `0.1182`
- stable-class macro-F1: `0.8959`

## Best Strategy Per Budget

| budget | strategy | macro-F1 | collapsed F1 | stable F1 | selected collapsed | selected absorber preds |
|---:|---|---:|---:|---:|---:|---:|
| 200 | absorber_margin_balanced | 0.7154 | 0.3559 | 0.8942 | 16 | 200 |
| 500 | absorber_margin | 0.7172 | 0.3655 | 0.8934 | 28 | 500 |
| 1000 | absorber_margin | 0.7192 | 0.3875 | 0.8947 | 60 | 1000 |

## Best Run Per Strategy

| strategy | budget | macro-F1 | collapsed F1 | stable F1 | selected collapsed |
|---|---:|---:|---:|---:|---:|
| absorber_margin | 1000 | 0.7192 | 0.3875 | 0.8947 | 60 |
| absorber_margin_balanced | 1000 | 0.7187 | 0.3859 | 0.8946 | 60 |
| absorber_random | 500 | 0.7169 | 0.3403 | 0.8935 | 11 |
| margin | 200 | 0.7323 | 0.2486 | 0.8915 | 0 |
| random | 200 | 0.7289 | 0.2454 | 0.8926 | 0 |

## Figures

- Budget curve: `outputs/collapse_active_replay_quic22_W-2022-47_all_r5_tr2_distill0.5/active_maintenance_budget_curve.png`
- Selected collapsed samples: `outputs/collapse_active_replay_quic22_W-2022-47_all_r5_tr2_distill0.5/active_maintenance_selected_collapse.png`

## Reading

- A useful active-maintenance strategy should select more collapsed samples than random at the same budget and improve collapsed-class F1 without collapsing stable-class F1.
- Oracle-collapse sampling is an upper bound because it uses target labels for selection.
