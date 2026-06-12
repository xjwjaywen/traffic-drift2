# Collapse Statistics Summary

- Final period: `M-2022-12`
- `recall_lt_0_01` is the number of classes with recall < 0.01.
- `recall_lt_0_05` is the number of classes with recall < 0.05.
- `recall_lt_0_1` is the number of classes with recall < 0.1.

## Normalization / AdaBN Threshold Counts

| group | method | n | mean recall | mean F1 | <0.01 | <0.05 | <0.1 |
|---|---|---:|---:|---:|---:|---:|---:|
| final_collapsed | GN | 13 | 0.0157 | 0.0255 | 9 | 11 | 13 |
| final_collapsed | IN | 13 | 0.0274 | 0.0351 | 8 | 10 | 12 |
| final_collapsed | BN | 13 | 0.0223 | 0.0329 | 8 | 10 | 12 |
| final_collapsed | LN | 13 | 0.0189 | 0.0289 | 8 | 11 | 12 |
| final_collapsed | BN Static | 13 | 0.0223 | 0.0329 | 8 | 10 | 12 |
| final_collapsed | BN + AdaBN | 13 | 0.0240 | 0.0325 | 6 | 10 | 12 |
| abrupt_collapsed | GN | 8 | 0.0102 | 0.0159 | 6 | 7 | 8 |
| abrupt_collapsed | IN | 8 | 0.0069 | 0.0102 | 6 | 8 | 8 |
| abrupt_collapsed | BN | 8 | 0.0164 | 0.0196 | 6 | 6 | 8 |
| abrupt_collapsed | LN | 8 | 0.0130 | 0.0177 | 6 | 7 | 8 |
| abrupt_collapsed | BN Static | 8 | 0.0164 | 0.0196 | 6 | 6 | 8 |
| abrupt_collapsed | BN + AdaBN | 8 | 0.0132 | 0.0192 | 6 | 7 | 8 |
| gradual_collapsed | GN | 5 | 0.0246 | 0.0408 | 3 | 4 | 5 |
| gradual_collapsed | IN | 5 | 0.0601 | 0.0750 | 2 | 2 | 4 |
| gradual_collapsed | BN | 5 | 0.0316 | 0.0541 | 2 | 4 | 4 |
| gradual_collapsed | LN | 5 | 0.0284 | 0.0468 | 2 | 4 | 4 |
| gradual_collapsed | BN Static | 5 | 0.0316 | 0.0541 | 2 | 4 | 4 |
| gradual_collapsed | BN + AdaBN | 5 | 0.0413 | 0.0537 | 0 | 3 | 4 |

## Class-Level Delta Counts

| group | method vs baseline | improved | harmed | unchanged | mean delta recall |
|---|---|---:|---:|---:|---:|
| final_collapsed | IN vs GN | 4 | 6 | 3 | 0.0117 |
| final_collapsed | BN + AdaBN vs BN | 3 | 5 | 5 | 0.0017 |
| final_collapsed | BN + AdaBN vs BN Static | 3 | 5 | 5 | 0.0017 |
| abrupt_collapsed | IN vs GN | 2 | 4 | 2 | -0.0032 |
| abrupt_collapsed | BN + AdaBN vs BN | 0 | 3 | 5 | -0.0032 |
| abrupt_collapsed | BN + AdaBN vs BN Static | 0 | 3 | 5 | -0.0032 |
| gradual_collapsed | IN vs GN | 2 | 2 | 1 | 0.0355 |
| gradual_collapsed | BN + AdaBN vs BN | 3 | 2 | 0 | 0.0097 |
| gradual_collapsed | BN + AdaBN vs BN Static | 3 | 2 | 0 | 0.0097 |

## TTA Drift-Type Group Summary

| group | method | n | mean recall | mean F1 | severe | collapsed |
|---|---|---:|---:|---:|---:|---:|
| stable | Static | 20 | 0.9007 | 0.9028 | 0 | 0 |
| final_collapsed | Static | 13 | 0.0157 | 0.0255 | 9 | 13 |
| abrupt_collapsed | Static | 8 | 0.0102 | 0.0159 | 6 | 8 |
| gradual_collapsed | Static | 5 | 0.0246 | 0.0408 | 3 | 5 |
| stable | EATA | 20 | 0.9008 | 0.9029 | 0 | 0 |
| final_collapsed | EATA | 13 | 0.0157 | 0.0255 | 9 | 13 |
| abrupt_collapsed | EATA | 8 | 0.0101 | 0.0157 | 6 | 8 |
| gradual_collapsed | EATA | 5 | 0.0248 | 0.0411 | 3 | 5 |
| stable | CoTTA | 20 | 0.9019 | 0.9033 | 0 | 0 |
| final_collapsed | CoTTA | 13 | 0.0155 | 0.0251 | 9 | 13 |
| abrupt_collapsed | CoTTA | 8 | 0.0097 | 0.0151 | 6 | 8 |
| gradual_collapsed | CoTTA | 5 | 0.0247 | 0.0410 | 3 | 5 |
| stable | SAR | 20 | 0.9008 | 0.9013 | 0 | 0 |
| final_collapsed | SAR | 13 | 0.0138 | 0.0228 | 9 | 13 |
| abrupt_collapsed | SAR | 8 | 0.0086 | 0.0140 | 6 | 8 |
| gradual_collapsed | SAR | 5 | 0.0220 | 0.0369 | 3 | 5 |
| stable | TTA-TC | 20 | 0.9040 | 0.8984 | 0 | 0 |
| final_collapsed | TTA-TC | 13 | 0.0343 | 0.0565 | 6 | 11 |
| abrupt_collapsed | TTA-TC | 8 | 0.0333 | 0.0559 | 4 | 7 |
| gradual_collapsed | TTA-TC | 5 | 0.0358 | 0.0575 | 2 | 4 |

## Interpretation

The key diagnostic is not whether a method moves macro-F1 by a small amount, but whether it reduces the number of collapsed classes with near-zero recall. If IN, AdaBN, or TTA methods improve only a few individual classes while most abrupt/gradual collapsed classes remain below 0.05 recall, the result supports a negative finding: normalization-statistics adaptation and generic TTA do not solve class-conditional collapse.
