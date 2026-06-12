# Active Replay Multi-Period Summary

## Best Run Per Period

| period | strategy | budget | macro-F1 | collapse F1 | stable F1 | Δmacro | Δcollapse | Δstable | selected collapsed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M-2022-7 | random | 1000 | 0.7376 | 0.5990 | 0.8188 | -0.0026 | 0.0956 | -0.0134 | 18 |
| M-2022-10 | random | 1000 | 0.6886 | 0.3917 | 0.8766 | 0.0050 | 0.1336 | -0.0324 | 19 |
| M-2022-12 | absorber_margin | 1000 | 0.6399 | 0.3500 | 0.8559 | 0.0112 | 0.3224 | -0.0468 | 76 |

## Static Baselines

| period | macro-F1 | collapse F1 | stable F1 | collapsed count |
|---|---:|---:|---:|---:|
| M-2022-7 | 0.7402 | 0.5034 | 0.8322 | 4 |
| M-2022-10 | 0.6836 | 0.2581 | 0.9090 | 7 |
| M-2022-12 | 0.6286 | 0.0276 | 0.9028 | 12 |

## Figures

- Collapsed F1 by period: `outputs/collapse_active_replay_tls22_summary_all_r5_tr2/active_replay_collapse_f1_by_period.png`
- Macro-F1 by period: `outputs/collapse_active_replay_tls22_summary_all_r5_tr2/active_replay_macro_f1_by_period.png`
- Trade-off M-2022-7: `outputs/collapse_active_replay_tls22_summary_all_r5_tr2/active_replay_tradeoff_M-2022-7.png`
- Trade-off M-2022-10: `outputs/collapse_active_replay_tls22_summary_all_r5_tr2/active_replay_tradeoff_M-2022-10.png`
- Trade-off M-2022-12: `outputs/collapse_active_replay_tls22_summary_all_r5_tr2/active_replay_tradeoff_M-2022-12.png`

## Reading

- The key question is whether the M-2022-12 signal repeats on M-2022-7 and M-2022-10.
- A method-oriented result should improve collapsed-class F1 while keeping stable-class F1 close to the static baseline.
