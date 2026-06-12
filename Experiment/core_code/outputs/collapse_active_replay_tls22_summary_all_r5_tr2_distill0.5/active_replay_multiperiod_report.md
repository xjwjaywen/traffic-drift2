# Active Replay Multi-Period Summary

## Best Run Per Period

| period | strategy | budget | macro-F1 | collapse F1 | stable F1 | Δmacro | Δcollapse | Δstable | selected collapsed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M-2022-7 | absorber_margin_balanced | 1000 | 0.7679 | 0.6825 | 0.8397 | 0.0277 | 0.1790 | 0.0075 | 57 |
| M-2022-10 | absorber_margin_balanced | 1000 | 0.7204 | 0.4923 | 0.8944 | 0.0368 | 0.2341 | -0.0146 | 49 |
| M-2022-12 | absorber_margin | 1000 | 0.6834 | 0.4262 | 0.8913 | 0.0547 | 0.3986 | -0.0114 | 76 |

## Static Baselines

| period | macro-F1 | collapse F1 | stable F1 | collapsed count |
|---|---:|---:|---:|---:|
| M-2022-7 | 0.7402 | 0.5034 | 0.8322 | 4 |
| M-2022-10 | 0.6836 | 0.2581 | 0.9090 | 7 |
| M-2022-12 | 0.6286 | 0.0276 | 0.9028 | 12 |

## Figures

- Collapsed F1 by period: `outputs/collapse_active_replay_tls22_summary_all_r5_tr2_distill0.5/active_replay_collapse_f1_by_period.png`
- Macro-F1 by period: `outputs/collapse_active_replay_tls22_summary_all_r5_tr2_distill0.5/active_replay_macro_f1_by_period.png`
- Trade-off M-2022-7: `outputs/collapse_active_replay_tls22_summary_all_r5_tr2_distill0.5/active_replay_tradeoff_M-2022-7.png`
- Trade-off M-2022-10: `outputs/collapse_active_replay_tls22_summary_all_r5_tr2_distill0.5/active_replay_tradeoff_M-2022-10.png`
- Trade-off M-2022-12: `outputs/collapse_active_replay_tls22_summary_all_r5_tr2_distill0.5/active_replay_tradeoff_M-2022-12.png`

## Reading

- The key question is whether the M-2022-12 signal repeats on M-2022-7 and M-2022-10.
- A method-oriented result should improve collapsed-class F1 while keeping stable-class F1 close to the static baseline.
