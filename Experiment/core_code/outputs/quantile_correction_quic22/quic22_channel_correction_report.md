# QUIC22 Channel Drift and Correction Summary

This report checks whether QUIC22 temporal drift is tied to a specific PPI channel and whether channel-level correction can recover frozen-model performance.

## Best Channel Correction

| period | raw macro-F1 | best setting | best macro-F1 | delta |
|---|---:|---|---:|---:|
| W-2022-46 | 0.7135 | ipt_tail_20_29 | 0.7082 | -0.0053 |
| W-2022-47 | 0.7194 | ipt_tail_20_29 | 0.7127 | -0.0067 |

## All Correction Settings

| period | setting | macro-F1 | delta vs raw | corrected-region W1 reduction |
|---|---|---:|---:|---:|
| W-2022-46 | all | 0.3182 | -0.3953 | 384.8209 |
| W-2022-46 | direction_front_0_9 | 0.5883 | -0.1252 | -2.9557 |
| W-2022-46 | ipt_tail_20_29 | 0.7082 | -0.0053 | 63.7623 |
| W-2022-46 | raw | 0.7135 | +0.0000 | 0.0000 |
| W-2022-46 | size_all | 0.6904 | -0.0231 | 208.0802 |
| W-2022-46 | size_direction_front_ipt_tail | 0.5534 | -0.1601 | 268.8868 |
| W-2022-47 | all | 0.3140 | -0.4054 | 434.2626 |
| W-2022-47 | direction_front_0_9 | 0.5960 | -0.1234 | -2.9702 |
| W-2022-47 | ipt_tail_20_29 | 0.7127 | -0.0067 | 65.8337 |
| W-2022-47 | raw | 0.7194 | +0.0000 | 0.0000 |
| W-2022-47 | size_all | 0.6942 | -0.0252 | 254.8199 |
| W-2022-47 | size_direction_front_ipt_tail | 0.5596 | -0.1598 | 317.6833 |

## Channel Drift Magnitude

| period | macro-F1 | W1 size | W1 direction | W1 IPT | drifted positions |
|---|---:|---:|---:|---:|---:|
| W-2022-45 | 0.7750 | 217.4035 | 0.3015 | 317.2155 | 2 |
| W-2022-46 | 0.7135 | 222.8789 | 0.4070 | 305.5938 | 3 |
| W-2022-47 | 0.7194 | 270.3471 | 0.3887 | 335.1351 | 3 |

## Reading

- If one channel correction gives a positive delta while others do not, QUIC drift is likely channel/position-specific rather than class-collapse dominated.
- If correcting all channels hurts, broad augmentation can be worse than targeted augmentation.
- A positive channel-correction result is diagnostic; deployment still needs a practical augmentation or calibration rule that does not use labels.
