# Advisor-Facing Result Visualizations

This folder contains figures generated from existing TTA, normalization, AdaBN, and collapse outputs.

## Generated Figures

- **tls22_tta/curve**: `outputs/teacher_result_visuals/tls22_tta_macro_f1_curve.png`
- **tls22_tta/bar**: `outputs/teacher_result_visuals/tls22_tta_final_macro_f1_bar.png`
- **quic22_tta/curve**: `outputs/teacher_result_visuals/quic22_tta_macro_f1_curve.png`
- **quic22_tta/bar**: `outputs/teacher_result_visuals/quic22_tta_final_macro_f1_bar.png`
- **norm/curve**: `outputs/teacher_result_visuals/tls22_norm_adabn_macro_f1_curve.png`
- **norm/bar**: `outputs/teacher_result_visuals/tls22_norm_adabn_final_macro_f1_bar.png`
- **norm/group_bar**: `outputs/teacher_result_visuals/tls22_norm_adabn_m12_group_f1.png`
- **collapse/heatmap**: `outputs/teacher_result_visuals/tls22_collapse_recall_heatmap.png`
- **tta_drift_groups/group_bar**: `outputs/teacher_result_visuals/tls22_tta_m12_drift_type_group_f1.png`
- **tta_drift_groups/focused_bar**: `outputs/teacher_result_visuals/tls22_tta_m12_stable_abrupt_gradual_f1.png`

## tls22_tta Final-Period Table

| method | macro_f1 | accuracy | aurc |
|---|---|---|---|
| TTA-TC | 0.6308 | 0.7547 | 0.8318 |
| CoTTA | 0.6299 | 0.7555 | 0.8323 |
| EATA | 0.6288 | 0.7521 | 0.8307 |
| SAR | 0.6288 | 0.7561 | 0.8326 |
| Static | 0.6286 | 0.7519 | 0.8306 |
| BN-Adapt | 0.6286 | 0.7519 | 0.8306 |
| Tent | 0.5915 | 0.7397 | 0.8222 |
| NOTE | 0.5891 | 0.7381 | 0.8247 |

## quic22_tta Final-Period Table

| method | macro_f1 | accuracy | aurc |
|---|---|---|---|
| TTA-TC | 0.7195 | 0.7144 | 0.5961 |
| Static | 0.7194 | 0.7116 | 0.5944 |
| BN-Adapt | 0.7194 | 0.7116 | 0.5944 |
| EATA | 0.7193 | 0.7115 | 0.5944 |
| CoTTA | 0.7183 | 0.7074 | 0.5918 |
| SAR | 0.7071 | 0.7188 | 0.5990 |
| NOTE | 0.6083 | 0.6488 | 0.5734 |
| Tent | 0.5444 | 0.6197 | 0.5531 |

## norm Final-Period Table

| method | macro_f1 | accuracy | aurc |
|---|---|---|---|
| BN + AdaBN | 0.6287 | 0.7674 | nan |
| GN | 0.6286 | 0.7519 | nan |
| LN | 0.6274 | 0.7592 | nan |
| BN | 0.5817 | 0.7322 | nan |
| BN Static | 0.5817 | 0.7322 | nan |
| IN | 0.5657 | 0.7064 | nan |
