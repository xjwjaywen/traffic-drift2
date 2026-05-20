# Advisor Follow-Up Summary: TTA, InstanceNorm, AdaBN, and Collapse-Type Analysis

## What Has Been Completed

The advisor's requested follow-up has been covered in three parts:

1. **Abrupt collapse / TTA comparison**
   - Added drift-type evaluation for representative TTA methods: Static, BN-Adapt, Tent, EATA, CoTTA, SAR, NOTE, and TTA-TC.
   - Added group-level reporting for stable, final-collapsed, abrupt-collapsed, gradual-collapsed, absorber, and degraded-noncollapsed classes.
   - Added visualizations for final-period TTA macro-F1 and drift-type group F1.

2. **InstanceNorm / normalization and AdaBN comparison**
   - Added CNN normalization variants: GN, IN, BN, and LN-style normalization.
   - Added AdaBN evaluation on the BN checkpoint.
   - Added per-class collapsed-class tables and recall heatmaps comparing GN, IN, BN, LN, and BN + AdaBN.

3. **Result organization and visualization**
   - Added advisor-facing figures under `Experiment/core_code/outputs/teacher_result_visuals`.
   - Added a compact collapse-statistics script to count how many collapsed classes remain below recall thresholds `<0.01`, `<0.05`, and `<0.1`.
   - Added a related-work table explaining why generic TTA and normalization-based adaptation are weak fits for encrypted traffic class collapse.

## Main Finding

The added experiments support a clear negative result:

**Generic TTA and normalization-statistics adaptation provide only marginal final-period macro-F1 changes and do not reliably recover collapsed classes.**

The important observation is class-level rather than global:

- Stable classes remain relatively strong.
- Collapsed classes, especially abrupt-collapsed classes, often keep recall near zero.
- IN and AdaBN can help individual classes slightly, but their effects are not stable across collapsed classes.
- TTA-TC improves collapsed-group F1 more than most generic TTA baselines, but collapsed-class recall remains very low in the final period.

This suggests that the main bottleneck is not a simple global covariate shift or BN-statistics mismatch. The failure mode is better described as **class-conditional representation collapse and absorber-class confusion**.

## What Still Needs To Be Done

No additional large-scale experiment is required to respond to the advisor's immediate comments. The current evidence is enough for a coherent follow-up response.

The only remaining low-cost item is to regenerate and send the compact collapse-statistics table after all server outputs are present:

```bash
cd /data/xjw/traffic-drift2/Experiment/core_code
python scripts/summarize_teacher_collapse_stats.py \
  --output-dir outputs/teacher_result_visuals
```

The key file to send is:

```text
outputs/teacher_result_visuals/collapse_stat_summary.md
```

## Optional Future Experiments

These are not necessary for the current advisor response, but may be useful if the advisor asks for a broader domain-adaptation comparison:

1. **CORAL or MMD feature alignment**
   - Lightweight domain-adaptation baseline.
   - Useful if the advisor wants an explicit non-TTA DA baseline.

2. **DANN-style adversarial domain adaptation**
   - Heavier training-time DA baseline.
   - More costly and less urgent because the current evidence already shows that normalization/TTA is insufficient.

3. **Collapse-aware detection instead of adaptation**
   - Detect abrupt collapse events and flag classes requiring retraining or active intervention.
   - This may be more realistic than claiming zero-label TTA can recover fully absorbed classes.

## Suggested Message To Advisor

老师，我按您的建议补了三部分：一是对 abrupt / gradual / stable 这些漂移类型分别统计了 TTA 方法效果；二是补了 InstanceNorm、BatchNorm、LayerNorm、GroupNorm 和 AdaBN 的对照；三是整理了 final collapsed classes 的 per-class recall heatmap 和统计表。

初步结果显示，TTA-TC 在整体 macro-F1 上只有边际提升，IN/AdaBN 也没有稳定恢复 collapsed classes。更明显的问题是，stable classes 表现仍然较好，但 collapsed / abrupt collapsed classes 的 recall 很多仍接近 0。因此目前结果更支持一个负面发现：现有 TTA 和 normalization-based adaptation 不能有效解决类别级塌缩问题，后续方法需要显式建模 class-conditional drift / absorber confusion，而不是只做全局归一化或熵最小化。
