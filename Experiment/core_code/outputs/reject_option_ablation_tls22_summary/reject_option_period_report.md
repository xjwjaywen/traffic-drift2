# Reject-Option Multi-Period Summary

This summary uses the final-collapsed class set from the collapse report as the risk group across periods.

## Best Rule Rows

| period | rule | coverage | collapsed reject | stable false reject | absorber error reduction | accepted macro-F1 |
|---|---|---:|---:|---:|---:|---:|
| M-2022-10 | absorber_distance | 0.9592 | 0.0783 | 0.0021 | 0.2754 | 0.6874 |
| M-2022-10 | absorber_proto_disagree | 0.9668 | 0.0696 | 0.0018 | 0.2451 | 0.6883 |
| M-2022-10 | confidence | 0.5470 | 0.8273 | 0.1692 | 0.9762 | 0.8728 |
| M-2022-10 | hybrid | 0.5168 | 0.8362 | 0.1756 | 0.9983 | 0.8633 |
| M-2022-10 | margin | 0.5494 | 0.8083 | 0.1726 | 0.9742 | 0.8696 |
| M-2022-10 | prototype_distance | 0.6572 | 0.2530 | 0.1027 | 0.2754 | 0.7272 |
| M-2022-12 | absorber_distance | 0.9562 | 0.0887 | 0.0029 | 0.2403 | 0.6339 |
| M-2022-12 | absorber_proto_disagree | 0.9625 | 0.0817 | 0.0028 | 0.2215 | 0.6347 |
| M-2022-12 | confidence | 0.5142 | 0.8951 | 0.1800 | 0.9752 | 0.8201 |
| M-2022-12 | hybrid | 0.4849 | 0.9036 | 0.1866 | 0.9963 | 0.8125 |
| M-2022-12 | margin | 0.5171 | 0.8581 | 0.1834 | 0.9732 | 0.8167 |
| M-2022-12 | prototype_distance | 0.6462 | 0.2589 | 0.1063 | 0.2403 | 0.6771 |
| M-2022-7 | absorber_distance | 0.9567 | 0.0270 | 0.0023 | 0.1066 | 0.7423 |
| M-2022-7 | absorber_proto_disagree | 0.9646 | 0.0226 | 0.0021 | 0.0891 | 0.7431 |
| M-2022-7 | confidence | 0.6088 | 0.7564 | 0.1685 | 0.9678 | 0.9003 |
| M-2022-7 | hybrid | 0.5693 | 0.7687 | 0.1771 | 0.9985 | 0.8921 |
| M-2022-7 | margin | 0.6099 | 0.7351 | 0.1749 | 0.9678 | 0.8988 |
| M-2022-7 | prototype_distance | 0.6675 | 0.1844 | 0.1028 | 0.1066 | 0.7807 |

## Mean Across Periods

| rule | mean coverage | mean collapsed reject | mean stable false reject | mean absorber error reduction |
|---|---:|---:|---:|---:|
| absorber_distance | 0.9574 | 0.0647 | 0.0024 | 0.2075 |
| absorber_proto_disagree | 0.9646 | 0.0580 | 0.0022 | 0.1852 |
| confidence | 0.5567 | 0.8263 | 0.1726 | 0.9731 |
| hybrid | 0.5236 | 0.8361 | 0.1798 | 0.9977 |
| margin | 0.5588 | 0.8005 | 0.1770 | 0.9717 |
| prototype_distance | 0.6570 | 0.2321 | 0.1039 | 0.2075 |

## Figures

- Absorber-distance periods: `outputs/reject_option_ablation_tls22_summary/reject_absorber_distance_periods.png`
- Rule trade-off by period: `outputs/reject_option_ablation_tls22_summary/reject_rule_period_collapsed_vs_stable.png`

## Reading

- A practical rule should keep coverage high and stable false reject low.
- A high collapsed reject rate with low coverage is diagnostic but may be too conservative for deployment.
- If absorber-distance remains stable over periods, it supports a lightweight collapse-aware selective rejection mechanism.
