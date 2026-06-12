# Active Replay Multi-Period Summary

## Best Run Per Period

| period | strategy | budget | macro-F1 | collapse F1 | stable F1 | Δmacro | Δcollapse | Δstable | selected collapsed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| W-2022-46 | absorber_random | 200 | 0.7063 | 0.1616 | 0.8907 | -0.0072 | -0.0011 | -0.0018 | 1 |
| W-2022-47 | absorber_margin | 1000 | 0.7192 | 0.3875 | 0.8947 | -0.0001 | 0.2693 | -0.0012 | 60 |

## Static Baselines

| period | macro-F1 | collapse F1 | stable F1 | collapsed count |
|---|---:|---:|---:|---:|
| W-2022-46 | 0.7135 | 0.1627 | 0.8925 | 0 |
| W-2022-47 | 0.7194 | 0.1182 | 0.8959 | 1 |

## Figures

- Collapsed F1 by period: `outputs/collapse_active_replay_quic22_summary_all_r5_tr2_distill0.5/active_replay_collapse_f1_by_period.png`
- Macro-F1 by period: `outputs/collapse_active_replay_quic22_summary_all_r5_tr2_distill0.5/active_replay_macro_f1_by_period.png`
- Trade-off W-2022-46: `outputs/collapse_active_replay_quic22_summary_all_r5_tr2_distill0.5/active_replay_tradeoff_W-2022-46.png`
- Trade-off W-2022-47: `outputs/collapse_active_replay_quic22_summary_all_r5_tr2_distill0.5/active_replay_tradeoff_W-2022-47.png`

## Reading

- The key question is whether the M-2022-12 signal repeats on M-2022-7 and M-2022-10.
- A method-oriented result should improve collapsed-class F1 while keeping stable-class F1 close to the static baseline.
