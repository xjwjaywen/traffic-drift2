# TLS22 Normalization Drift-Type Ablation

- Baseline norm: `gn`
- Metrics are static predictions from each trained checkpoint.

## Overall Metrics

| norm | period | macro-F1 | final-collapsed F1 | stable F1 | collapsed count | severe count |
|---|---|---:|---:|---:|---:|---:|
| gn | M-2022-7 | 0.7402 | 0.4647 | 0.8322 | 5 | 3 |
| gn | M-2022-10 | 0.6836 | 0.2393 | 0.9090 | 8 | 5 |
| gn | M-2022-12 | 0.6286 | 0.0255 | 0.9028 | 13 | 9 |
| in | M-2022-7 | 0.6710 | 0.4249 | 0.7946 | 5 | 3 |
| in | M-2022-10 | 0.6138 | 0.2044 | 0.8395 | 8 | 6 |
| in | M-2022-12 | 0.5657 | 0.0351 | 0.8541 | 12 | 8 |
| bn | M-2022-7 | 0.7043 | 0.4583 | 0.8278 | 5 | 3 |
| bn | M-2022-10 | 0.6381 | 0.2275 | 0.8848 | 8 | 7 |
| bn | M-2022-12 | 0.5817 | 0.0329 | 0.8692 | 12 | 8 |
| ln | M-2022-7 | 0.7460 | 0.4448 | 0.8368 | 5 | 3 |
| ln | M-2022-10 | 0.6867 | 0.2303 | 0.9020 | 8 | 6 |
| ln | M-2022-12 | 0.6274 | 0.0289 | 0.8905 | 12 | 8 |

## Group Deltas vs Baseline

| norm | period | group | ΔF1 | ΔRecall | Δcollapsed | Δsevere |
|---|---|---|---:|---:|---:|---:|
| in | M-2022-7 | stable | -0.0376 | -0.0434 | -1 | 0 |
| in | M-2022-7 | final_collapsed | -0.0398 | -0.0512 | 0 | 0 |
| in | M-2022-7 | abrupt_collapsed | -0.0616 | -0.0803 | 0 | 0 |
| in | M-2022-7 | gradual_collapsed | -0.0049 | -0.0045 | 0 | 0 |
| in | M-2022-7 | absorber | -0.0956 | -0.1530 | 0 | 0 |
| in | M-2022-7 | degraded_noncollapsed | -0.0665 | -0.0685 | 0 | 0 |
| in | M-2022-10 | stable | -0.0695 | -0.0785 | 0 | 0 |
| in | M-2022-10 | final_collapsed | -0.0348 | -0.0445 | 0 | 1 |
| in | M-2022-10 | abrupt_collapsed | -0.0660 | -0.0813 | 1 | 1 |
| in | M-2022-10 | gradual_collapsed | 0.0151 | 0.0143 | -1 | 0 |
| in | M-2022-10 | absorber | -0.0891 | -0.1238 | 0 | 0 |
| in | M-2022-10 | degraded_noncollapsed | -0.0869 | -0.0772 | 6 | 0 |
| in | M-2022-12 | stable | -0.0486 | -0.0585 | 0 | 0 |
| in | M-2022-12 | final_collapsed | 0.0097 | 0.0117 | -1 | -1 |
| in | M-2022-12 | abrupt_collapsed | -0.0057 | -0.0032 | 0 | 0 |
| in | M-2022-12 | gradual_collapsed | 0.0342 | 0.0355 | -1 | -1 |
| in | M-2022-12 | absorber | -0.0696 | -0.1171 | 0 | 0 |
| in | M-2022-12 | degraded_noncollapsed | -0.0822 | -0.0789 | 8 | 0 |
| bn | M-2022-7 | stable | -0.0043 | 0.0045 | -1 | 0 |
| bn | M-2022-7 | final_collapsed | -0.0064 | 0.0029 | 0 | 0 |
| bn | M-2022-7 | abrupt_collapsed | -0.0176 | 0.0106 | 0 | 0 |
| bn | M-2022-7 | gradual_collapsed | 0.0114 | -0.0094 | 0 | 0 |
| bn | M-2022-7 | absorber | -0.0685 | -0.0243 | 0 | 0 |
| bn | M-2022-7 | degraded_noncollapsed | -0.0514 | -0.0527 | 3 | 0 |
| bn | M-2022-10 | stable | -0.0242 | -0.0196 | 0 | 0 |
| bn | M-2022-10 | final_collapsed | -0.0118 | 0.0029 | 0 | 2 |
| bn | M-2022-10 | abrupt_collapsed | -0.0215 | 0.0056 | 0 | 1 |
| bn | M-2022-10 | gradual_collapsed | 0.0038 | -0.0016 | 0 | 1 |
| bn | M-2022-10 | absorber | -0.0650 | -0.0271 | 0 | 0 |
| bn | M-2022-10 | degraded_noncollapsed | -0.0961 | -0.0637 | 6 | 0 |
| bn | M-2022-12 | stable | -0.0336 | -0.0217 | 0 | 0 |
| bn | M-2022-12 | final_collapsed | 0.0075 | 0.0066 | -1 | -1 |
| bn | M-2022-12 | abrupt_collapsed | 0.0038 | 0.0062 | 0 | 0 |
| bn | M-2022-12 | gradual_collapsed | 0.0133 | 0.0070 | -1 | -1 |
| bn | M-2022-12 | absorber | -0.0545 | -0.0214 | 0 | 0 |
| bn | M-2022-12 | degraded_noncollapsed | -0.0759 | -0.0534 | 2 | 0 |
| ln | M-2022-7 | stable | 0.0047 | 0.0165 | -1 | 0 |
| ln | M-2022-7 | final_collapsed | -0.0199 | 0.0027 | 0 | 0 |
| ln | M-2022-7 | abrupt_collapsed | -0.0296 | 0.0007 | 0 | 0 |
| ln | M-2022-7 | gradual_collapsed | -0.0044 | 0.0060 | 0 | 0 |
| ln | M-2022-7 | absorber | 0.0021 | -0.0141 | 0 | 0 |
| ln | M-2022-7 | degraded_noncollapsed | 0.0242 | 0.0116 | -1 | 0 |
| ln | M-2022-10 | stable | -0.0070 | -0.0073 | 0 | 0 |
| ln | M-2022-10 | final_collapsed | -0.0090 | -0.0078 | 0 | 1 |
| ln | M-2022-10 | abrupt_collapsed | -0.0153 | -0.0148 | 0 | 1 |
| ln | M-2022-10 | gradual_collapsed | 0.0012 | 0.0034 | 0 | 0 |
| ln | M-2022-10 | absorber | -0.0002 | -0.0108 | 0 | 0 |
| ln | M-2022-10 | degraded_noncollapsed | 0.0045 | 0.0132 | 1 | 0 |
| ln | M-2022-12 | stable | -0.0122 | -0.0019 | 0 | 0 |
| ln | M-2022-12 | final_collapsed | 0.0034 | 0.0032 | -1 | -1 |
| ln | M-2022-12 | abrupt_collapsed | 0.0018 | 0.0028 | 0 | 0 |
| ln | M-2022-12 | gradual_collapsed | 0.0060 | 0.0039 | -1 | -1 |
| ln | M-2022-12 | absorber | -0.0043 | -0.0095 | 0 | 0 |
| ln | M-2022-12 | degraded_noncollapsed | 0.0081 | 0.0073 | 1 | 0 |
