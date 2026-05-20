# Experiment Inventory and Current Conclusions

This document summarizes the main experiment lines completed so far for encrypted-traffic temporal drift, especially the TLS22 class-collapse diagnosis. It is meant as an advisor-facing inventory rather than a full paper section.

## Coverage Note

The conclusions below are based on two evidence sources:

- **Repository-available artifacts**: files currently committed under `Experiment/core_code/outputs/`, `Experiment/core_code/scripts/`, and `Survey/reports/`.
- **Server-run summaries**: outputs that were run on the GPU server and discussed in the experiment log, but whose full raw directories are not all committed to the repository.

Therefore, this inventory covers the main experiment lines related to TLS22/QUIC22 drift and class collapse, but it is not a byte-by-byte audit of every historical `outputs/` directory on the server.

## High-Level Result

The strongest current conclusion is negative but clear:

**Existing generic TTA, normalization-based adaptation, static recalibration, and simple target-prototype adaptation do not reliably recover collapsed TLS22 classes.**

The main failure mode is not only global distribution shift. The dominant bottleneck is **class-conditional collapse**, where some classes are absorbed by specific high-confidence confusion targets. This is especially severe for abrupt collapse.

## Experiment Lines

| Line | Methods / variants | Main artifacts | Key result | Current conclusion |
|---|---|---|---|---|
| Base source models | TLS22 CNN, QUIC22 CNN | `outputs/tls22_cnn/train_results.json`, `outputs/quic22_cnn/train_results.json` | TLS22 source test macro-F1 `0.8348`; QUIC22 source test macro-F1 `0.7788` | Base classifiers are reasonable, so later failure is not simply a broken source model |
| Sequential TTA baselines | Static, BN-Adapt, Tent, EATA, CoTTA, SAR, NOTE, TTA-TC | `outputs/eval_tls22/results_sequential.json`, `outputs/eval_quic22/results_sequential.json`, `outputs/teacher_result_visuals/teacher_result_visuals_summary.md` | TLS22 M12: Static `0.6286`, TTA-TC `0.6308`; QUIC22 final: Static `0.7194`, TTA-TC `0.7195` | Generic TTA gives only marginal overall macro-F1 gains |
| Drift-type TTA analysis | Static/EATA/CoTTA/SAR/TTA-TC grouped by stable/final-collapsed/abrupt/gradual | `scripts/tta_drift_type_ablation_tls22.py`, server output `outputs/tta_drift_type_ablation_tls22`, `collapse_stat_summary.md` | TLS22 M12 final-collapsed F1: Static `0.0255`, TTA-TC `0.0565`; 11/13 collapsed classes still below recall `0.1` | TTA-TC has a weak collapsed-class signal, but does not solve collapse |
| Monthly collapse diagnosis | Per-class recall timeline, first collapse month, absorber class | `scripts/summarize_per_class_collapse.py`, server output `outputs/per_class_collapse_tls22_monthly` | Collapsed classes increase from `0` in M4 to about `12-13` in M12; many have recall near zero | Long-term TLS22 drift is class-conditional and accumulates over time |
| Absorber/confusion analysis | Top confusion targets for collapsed classes | `collapse_classes.csv`, `collapse_pairs.csv`, server summaries | Examples: `56 -> 96`, `48 -> 14`, `47 -> 5`, `109 -> 71` | Collapse is not random; classes are absorbed by specific dominant classes |
| Collapse-pair source distance | Source/reference prototype distance between future collapsed pair and absorber | `scripts/analyze_collapse_pair_distances.py`, server output `outputs/collapse_pair_distances_tls22` | Many future absorber pairs were not nearest pairs in source space, e.g. `56 -> 96` rank `39`, `109 -> 71` rank `95` | Source-space margin alone cannot predict or prevent all later collapse |
| Normalization ablation | GN, IN, BN, LN-style | `scripts/norm_drift_type_ablation_tls22.py`, `outputs/teacher_result_visuals/m12_collapsed_norm_adabn_per_class.md` | M12 macro-F1: GN `0.6286`, IN `0.5657`, BN `0.5817`, LN `0.6274` | IN/BN do not improve overall robustness; GN/LN are safer overall |
| AdaBN adaptation | BN Static vs BN + AdaBN | `scripts/adabn_drift_type_ablation_tls22.py`, `outputs/teacher_result_visuals/collapse_stat_summary.md` | M12 final-collapsed mean recall: BN `0.0223`, BN+AdaBN `0.0240`; improved 3 classes, harmed 5 classes | AdaBN has no stable collapsed-class recovery effect |
| Drift-type normalization effects | Stable vs abrupt vs gradual collapse under IN/AdaBN | `outputs/teacher_result_visuals/collapse_stat_summary.md` | IN helps gradual mean recall (`0.0246 -> 0.0601` vs GN) but hurts abrupt (`0.0102 -> 0.0069`) | InstanceNorm is only mildly helpful for gradual drift and weak/harmful for abrupt collapse |
| Global / marginal correction | Quantile or global correction | server output `outputs/quantile_correction_tls22` | Reported as ineffective in prior experiment summaries | Global distribution correction is not enough for class-conditional collapse |
| Static prototype recalibration | Source/reference prototype score correction | server outputs `outputs/prototype_recalibration_tls22`, `outputs/prototype_recalibration_tls22_v2` | Only extremely weak macro-F1 change | Static source prototypes provide weak prior but cannot recover collapsed target classes |
| Oracle pair recalibration | Target-label-derived absorber pairs, beta/top sweep | server output `outputs/oracle_pair_recalibration_tls22` | Static macro-F1 `0.628647`; best oracle pair macro-F1 `0.628840`; bad macro-F1 `0.1544 -> 0.1546` | Even oracle pairwise score correction has a very low upper bound |
| CAPS target prototype | Confidence-gated target prototype update | `scripts/caps_target_prototype_tls22.py`, server outputs `outputs/caps_target_prototype_*` | M7 `0.7402 -> 0.7404`; M10 `0.6836 -> 0.6861`; M12 `0.6286 -> 0.6319`; M12 bad F1 `0.1544 -> 0.1639` | Target-adaptive prototypes give small, more visible gains under stronger drift, but only recover part of the bad classes |
| CAPS recovery bins | Recovered/unchanged/harmed class split | server output `outputs/caps_target_prototype_summary` | M12 bad recovered: 5 classes, mean delta F1 about `+0.046`; many bad classes unchanged | Some bad classes are recoverable; fully collapsed classes often remain unsolved |
| CAPS++ adapter | Target-side representation adapter with prototype/anchor ideas | `scripts/capspp_adapter_tls22.py`, server output `outputs/capspp_adapter_tls22_M-2022-12` | Best run collapsed badly: macro-F1 about `0.1839`, bad F1 about `0.0264` | Direct target-side representation adaptation is unstable in this setting |
| Training-time temporal invariance | Pooled ERM, class-balanced ERM, risk-weighted ERM, temporal prototype loss | `scripts/train_temporal_invariance_tls22.py`, server outputs `outputs/titc_validation_tls22_*` | Warm-start pooled ERM around M12 macro-F1 `0.646`; temporal prototype loss did not clearly beat pooled ERM | Training with more months helps somewhat, but the tested temporal prototype regularizer is not yet a method contribution |
| SSL task ablation | MPFP-only, POP-only, all SSL tasks | `outputs/verify/ablation_ssl_tasks.json` | All settings around accuracy `0.0977`, macro-F1 `0.0178` in verify output | Current SSL path is not validated and should not be claimed as effective |
| Active learning / labeled adaptation sweep | FT head, kNN labeled, TTA-TC sample strategies | server outputs `outputs/al_sweep/*` | Raw files exist on server but have not been fully summarized in this inventory | Useful if reframing as labeled/active maintenance, but not directly comparable to zero-label TTA |
| Other branches | CA_TTA, DT_TTA, TIF_extension, WST_validation | server-side output directories mentioned in logs | Not fully audited here | Should not be cited as final evidence until summarized separately |

## Most Important Numeric Takeaways

### Generic TTA

| Dataset / period | Static | Best observed generic TTA | Difference | Interpretation |
|---|---:|---:|---:|---|
| TLS22 M12 macro-F1 | `0.6286` | TTA-TC `0.6308` | `+0.0022` | Marginal overall gain |
| QUIC22 final macro-F1 | `0.7194` | TTA-TC `0.7195` | `+0.0001` | Essentially unchanged |
| TLS22 M12 final-collapsed F1 | Static `0.0255` | TTA-TC `0.0565` | `+0.0310` | Weak collapsed-class signal, still far from recovery |

### Normalization / AdaBN

| Group | Method | Mean recall | Classes below recall 0.1 | Interpretation |
|---|---|---:|---:|---|
| final collapsed | GN | `0.0157` | `13/13` | Collapsed classes nearly all unrecovered |
| final collapsed | IN | `0.0274` | `12/13` | Slightly fewer collapsed classes, poor overall |
| final collapsed | BN + AdaBN | `0.0240` | `12/13` | Small and inconsistent |
| abrupt collapsed | IN | `0.0069` | `8/8` | Worse than GN on abrupt collapse |
| gradual collapsed | IN | `0.0601` | `4/5` | Some help for gradual collapse only |

### Static Correction / Prototype-Based Correction

| Method line | Best observed effect | Interpretation |
|---|---|---|
| Static prototype recalibration | Extremely weak | Source prototypes are not enough once target class structure has shifted |
| Oracle pair recalibration | Macro-F1 `0.628647 -> 0.628840` | Even target-label-derived pair correction has very low ceiling |
| CAPS target prototype | M12 macro-F1 `0.6286 -> 0.6319`; bad F1 `0.1544 -> 0.1639` | Weak but real signal; insufficient for fully collapsed classes |

## Consolidated Interpretation

The experiments support the following conclusions:

1. **The main problem is class-conditional collapse, not only global covariate shift.**
   Stable classes can remain strong while a small set of classes collapses almost completely.

2. **Generic TTA is not enough.**
   Entropy-minimization and standard continual TTA methods mostly change overall macro-F1 by tiny amounts and do not reliably recover collapsed classes.

3. **Normalization-statistics adaptation is not enough.**
   IN and AdaBN show small class-specific effects, especially for gradual collapse, but they do not solve abrupt collapse.

4. **Static post-hoc correction is not enough.**
   Global correction, source prototype recalibration, and even oracle pair-aware recalibration have very low upper bounds.

5. **Target-adaptive prototypes show a direction but not a final method.**
   CAPS produces small improvements and helps some recoverable bad classes, but fully collapsed classes remain a major unsolved case.

6. **A realistic next direction is collapse-aware selective intervention.**
   The current evidence motivates detecting abrupt/gradual collapse and deciding when to use light adaptation, when to abstain, and when to trigger active maintenance or retraining.

## Suggested Advisor-Facing Summary

老师，我把目前主要实验线整理了一遍。整体结论不是某一个 TTA 方法失败，而是几类自然路线都显示出类似问题：普通 TTA、IN/AdaBN 这类归一化自适应、静态 prototype/pairwise correction 都只能带来很小的改善，不能稳定恢复 collapsed classes。尤其是 abrupt collapse，很多类在 M12 recall 仍接近 0。

目前最清楚的贡献点是诊断：TLS22 长期漂移的瓶颈是 class-conditional collapse 和 absorber-class confusion，而不是简单全局分布偏移。后续如果继续做方法，我认为更合理的方向不是再堆通用 TTA，而是做 collapse-aware detection 和 selective intervention：区分 stable / gradual / abrupt collapse，对不同类型采用不同处理策略。

## Remaining Work

No additional large-scale experiment is required to answer the advisor's immediate InstanceNorm/TTA/AdaBN comments. The main remaining work is organizational:

- Keep the server-side raw output directories synchronized if they need to be cited directly.
- If the advisor asks for explicit domain-adaptation baselines, run a lightweight CORAL/MMD feature-alignment baseline before considering heavier DANN-style training.
- If the paper direction shifts from zero-label TTA to maintenance, summarize `outputs/al_sweep/*` and compare active/few-shot maintenance strategies.
