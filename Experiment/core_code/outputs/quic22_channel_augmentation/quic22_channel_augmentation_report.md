# QUIC22 Targeted Channel Augmentation Summary

This is frozen-model test-time augmentation averaging; no model parameters are updated.

## Best Setting By Period

| period | raw macro-F1 | best setting | best macro-F1 | delta |
|---|---:|---|---:|---:|
| W-2022-46 | 0.7135 | size_noise_0.02 | 0.7135 | +0.0000 |
| W-2022-47 | 0.7194 | direction_front_dropout_0.02 | 0.7194 | +0.0001 |

## All Settings

| period | setting | accuracy | macro-F1 | delta macro-F1 |
|---|---|---:|---:|---:|
| W-2022-46 | direction_front_dropout_0.02 | 0.7172 | 0.7133 | -0.0002 |
| W-2022-46 | direction_front_dropout_0.05 | 0.7195 | 0.7130 | -0.0005 |
| W-2022-46 | ipt_noise_0.05 | 0.7138 | 0.6837 | -0.0298 |
| W-2022-46 | packet_mask_0.02 | 0.7151 | 0.7106 | -0.0029 |
| W-2022-46 | raw | 0.7157 | 0.7135 | +0.0000 |
| W-2022-46 | size_noise_0.02 | 0.7157 | 0.7135 | +0.0000 |
| W-2022-47 | direction_front_dropout_0.02 | 0.7130 | 0.7194 | +0.0001 |
| W-2022-47 | direction_front_dropout_0.05 | 0.7153 | 0.7193 | -0.0001 |
| W-2022-47 | ipt_noise_0.05 | 0.7097 | 0.6845 | -0.0349 |
| W-2022-47 | packet_mask_0.02 | 0.7109 | 0.7159 | -0.0035 |
| W-2022-47 | raw | 0.7116 | 0.7194 | +0.0000 |
| W-2022-47 | size_noise_0.02 | 0.7116 | 0.7194 | -0.0000 |

## Reading

- Positive deltas indicate that robustness to that channel perturbation helps target-period prediction.
- If broad packet/channel perturbations hurt but one targeted channel helps, QUIC drift should be handled with channel-specific augmentation rather than generic TTA.
