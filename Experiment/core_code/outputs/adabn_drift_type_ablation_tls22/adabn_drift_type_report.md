# TLS22 AdaBN Drift-Type Ablation

- BatchNorm layers adapted: `3`
- AdaBN recomputes BN running statistics on unlabeled target-period data.
- Learned model weights are not updated.

## Overall Metrics

| method | period | macro-F1 | final-collapsed F1 | stable F1 | collapsed count | severe count |
|---|---|---:|---:|---:|---:|---:|
| bn_static | M-2022-7 | 0.7043 | 0.4583 | 0.8278 | 5 | 3 |
| bn_adabn | M-2022-7 | 0.7436 | 0.4865 | 0.8324 | 5 | 3 |
| bn_static | M-2022-10 | 0.6381 | 0.2275 | 0.8848 | 8 | 7 |
| bn_adabn | M-2022-10 | 0.6746 | 0.2423 | 0.9072 | 8 | 6 |
| bn_static | M-2022-12 | 0.5817 | 0.0329 | 0.8692 | 12 | 8 |
| bn_adabn | M-2022-12 | 0.6287 | 0.0325 | 0.8851 | 12 | 6 |

## AdaBN Deltas vs BN Static

| period | group | Delta F1 | Delta Recall | Delta collapsed | Delta severe |
|---|---|---:|---:|---:|---:|
| M-2022-7 | stable | 0.0046 | 0.0126 | 0 | 0 |
| M-2022-7 | final_collapsed | 0.0282 | 0.0168 | 0 | 0 |
| M-2022-7 | abrupt_collapsed | 0.0294 | 0.0087 | 0 | 0 |
| M-2022-7 | gradual_collapsed | 0.0264 | 0.0297 | 0 | 0 |
| M-2022-7 | absorber | 0.0739 | 0.0251 | 0 | 0 |
| M-2022-7 | degraded_noncollapsed | 0.0504 | 0.0480 | -1 | 0 |
| M-2022-10 | stable | 0.0225 | 0.0212 | 0 | 0 |
| M-2022-10 | final_collapsed | 0.0148 | -0.0103 | 0 | -1 |
| M-2022-10 | abrupt_collapsed | 0.0245 | -0.0154 | 0 | 0 |
| M-2022-10 | gradual_collapsed | -0.0007 | -0.0023 | 0 | -1 |
| M-2022-10 | absorber | 0.0583 | 0.0078 | 0 | 0 |
| M-2022-10 | degraded_noncollapsed | 0.0615 | 0.0381 | -5 | 0 |
| M-2022-12 | stable | 0.0159 | 0.0204 | 0 | 0 |
| M-2022-12 | final_collapsed | -0.0004 | 0.0017 | 0 | -2 |
| M-2022-12 | abrupt_collapsed | -0.0004 | -0.0032 | 0 | 0 |
| M-2022-12 | gradual_collapsed | -0.0004 | 0.0097 | 0 | -2 |
| M-2022-12 | absorber | 0.0611 | 0.0231 | 0 | 0 |
| M-2022-12 | degraded_noncollapsed | 0.0750 | 0.0592 | -1 | 0 |
