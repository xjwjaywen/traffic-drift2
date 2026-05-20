# Teacher-Suggested Literature Survey: TTA, AdaBN, and Encrypted-Traffic Drift

This note summarizes the papers and directions suggested by the advisor:

1. continual / drift-aware test-time adaptation;
2. InstanceNorm / Adaptive Batch Normalization;
3. domain adaptation and drift robustness for encrypted traffic classification.

The goal is not only to list papers, but to connect them to our current TLS22 findings: long-term temporal drift, abrupt/gradual class collapse, weak gains from generic TTA and static prototype correction, and normalization-type ablation results.

## 1. Continual Test-Time Adaptation for Drift

### 1.1 CoTTA: Continual Test-Time Domain Adaptation

**Paper:** Qin Wang, Olga Fink, Luc Van Gool, Dengxin Dai. "Continual Test-Time Domain Adaptation." CVPR 2022.  
**Link:** https://arxiv.org/abs/2203.13591  
**Code:** https://github.com/qinenergy/cotta

**Problem setting.** Standard test-time adaptation usually assumes a fixed target domain. CoTTA considers a more realistic setting where the target distribution changes continuously over time. The paper explicitly points out two failure modes under non-stationary target streams:

- pseudo-labels become unreliable as the target distribution moves;
- noisy self-training can cause error accumulation and catastrophic forgetting.

**Method.**

- **Teacher-student / weight-averaged predictions:** maintain a teacher model as an exponential moving average of the adapted model to make pseudo-labels more stable.
- **Augmentation-averaged predictions:** use multiple augmented views to reduce prediction noise.
- **Stochastic restoration:** randomly restore a small subset of weights to the original source model during adaptation, which slows forgetting.
- **Full-network adaptation:** unlike Tent, which mainly updates normalization affine parameters, CoTTA adapts all model parameters.

**Relevance to our work.**

CoTTA is directly relevant because our TLS22 evaluation is a continual temporal stream. It gives a clean vocabulary for our problem: *error accumulation*, *catastrophic forgetting*, and *non-stationary target domain*. However, our current results suggest an extra issue beyond the CoTTA setting: some classes collapse almost completely, so their pseudo-labels become nearly unusable. In such classes, teacher-student smoothing may preserve wrong predictions rather than recover the class.

**How to position it.**

CoTTA should be treated as a key baseline and related work for continual TTA. If CoTTA only improves weakly on TLS22, this supports the claim that encrypted traffic drift contains class-conditional / concept-drift components that are harder than image corruption streams.

### 1.2 Tent: Fully Test-Time Adaptation by Entropy Minimization

**Paper:** Dequan Wang et al. "Tent: Fully Test-time Adaptation by Entropy Minimization." ICLR 2021.  
**Link:** https://arxiv.org/abs/2006.10726

**Core idea.** Tent adapts at test time by minimizing prediction entropy. It updates normalization statistics and channel-wise affine parameters online.

**Relevance.**

Tent is the basic entropy-minimization baseline behind many later TTA methods. Its assumption is that making predictions more confident on target data improves target accuracy. This can fail under class collapse: if the model already maps a collapsed class into a wrong absorber class with high confidence, entropy minimization will reinforce the wrong decision.

**Connection to our results.**

Our TLS22 results are consistent with this limitation: generic entropy-based TTA methods do not recover collapsed classes. The failure is not just "not enough adaptation"; it is that the unsupervised adaptation signal points in the wrong direction for collapsed classes.

### 1.3 EATA: Efficient Test-Time Model Adaptation without Forgetting

**Paper:** Shuaicheng Niu et al. "Efficient Test-Time Model Adaptation without Forgetting." ICML 2022.  
**Link:** https://proceedings.mlr.press/v162/niu22a.html

**Core idea.**

- Filter out unreliable or redundant samples for entropy minimization.
- Add an anti-forgetting regularizer to preserve important model weights.

**Relevance.**

EATA is a stronger version of Tent. It addresses noisy gradients and forgetting, which are relevant to long streams. But it still depends on unsupervised confidence/entropy criteria. If collapsed-class samples are confidently assigned to absorber classes, they may not be corrected.

### 1.4 SAR: Towards Stable Test-Time Adaptation in Dynamic Wild World

**Paper:** Shuaicheng Niu et al. "Towards Stable Test-Time Adaptation in Dynamic Wild World." ICLR 2023.  
**Link:** https://openreview.net/pdf?id=g2YraF75Tj  
**Code:** https://github.com/mr-eggplant/SAR

**Core idea.**

SAR studies "wild" test-time settings: mixed shifts, small batches, and imbalanced online streams. It argues that BN can be unstable under these settings, and that GN/LN are often more stable. SAR uses reliable entropy filtering and sharpness-aware minimization to prevent model collapse during TTA.

**Relevance.**

This is highly relevant to the advisor's InstanceNorm / normalization suggestion. SAR gives a reason why normalization choice matters. It also matches our own norm ablation:

- GN and LN are more stable overall.
- IN/BN slightly help a few late collapsed groups but hurt stable and degraded noncollapsed classes.
- Normalization alone does not solve class collapse.

**Important distinction.**

SAR's "model collapse" refers to adaptation collapse, e.g., entropy minimization making the model predict one class. Our "class collapse" refers to certain true classes being absorbed by other classes under temporal drift. They are related but not identical. This distinction should be explicit in writing.

### 1.5 NOTE: Robust Continual Test-Time Adaptation Against Temporal Correlation

**Paper:** Taesik Gong et al. "NOTE: Robust Continual Test-time Adaptation Against Temporal Correlation." NeurIPS 2022.  
**Link:** https://arxiv.org/abs/2208.05117

**Core idea.**

NOTE observes that real test streams are not i.i.d.; temporal correlation can break existing TTA methods. It proposes:

- Instance-Aware Batch Normalization (IABN);
- Prediction-balanced Reservoir Sampling (PBRS), which tries to build a more class-balanced memory from a non-i.i.d. stream.

**Relevance.**

NOTE is useful because network traffic streams can be temporally correlated and class-imbalanced. PBRS is conceptually close to our earlier buffering ideas. However, NOTE still assumes that prediction-balanced pseudo-labels are meaningful. For fully collapsed classes, pseudo-label balance may still be wrong if the collapsed class is never predicted.

## 2. AdaBN / InstanceNorm / Normalization Adaptation

### 2.1 AdaBN: Revisiting Batch Normalization for Practical Domain Adaptation

**Paper:** Yanghao Li, Naiyan Wang, Jianping Shi, Jiaying Liu, Xiaodi Hou. "Revisiting Batch Normalization For Practical Domain Adaptation." 2016.  
**Link:** https://arxiv.org/abs/1603.04779  
**PyTorch reference from advisor:** https://github.com/sainatarajan/adabn-pytorch

**Core idea.**

AdaBN keeps learned model weights fixed and replaces source-domain BN running statistics with target-domain statistics. It is parameter-free and does not require target labels.

In implementation terms:

1. run the BN model on unlabeled target data;
2. estimate target-domain running mean and variance for each BN layer;
3. replace the old BN statistics;
4. evaluate with all learned weights unchanged.

**What it can solve.**

AdaBN is suitable for covariate shift where domain change appears as feature mean/variance shift, similar to style or sensor distribution changes.

**What it probably cannot solve.**

AdaBN cannot directly repair class-conditional concept drift. If class 56 has become indistinguishable from class 96 in the learned representation, replacing global BN statistics will not introduce class-specific separation.

**Connection to our current experiments.**

Our default TLS22 model uses GN, so AdaBN cannot be applied to it directly. We already trained/evaluated BN and IN/LN variants. The current normalization ablation shows:

- IN/BN slightly improve final-collapsed F1 in M-2022-12, but only by about +0.007 to +0.010 over GN.
- IN severely hurts overall macro-F1, stable classes, absorber classes, and degraded noncollapsed classes.
- LN is the least harmful alternative and is close to GN, but still does not solve collapse.

Therefore, AdaBN should be tested as the advisor suggested, but our expected conclusion should be conservative: it is a useful sanity check for covariate-shift normalization, not a complete answer to abrupt class collapse.

### 2.2 InstanceNorm

InstanceNorm normalizes each sample independently. In images, it is often useful for style-like shifts because it removes per-instance intensity/contrast statistics. In encrypted traffic, however, per-flow statistics such as packet-size scale, burst strength, and timing magnitude can themselves be discriminative.

**Our current observation.**

InstanceNorm hurts overall performance and stable classes:

- M-2022-12 macro-F1: GN 0.6286 vs IN 0.5657.
- M-2022-12 stable F1: GN 0.9028 vs IN 0.8541.
- M-2022-12 final-collapsed F1: GN 0.0255 vs IN 0.0351.

So IN gives a tiny late benefit on collapsed classes, especially gradual collapse, but removes too much useful traffic information. This is a strong analysis point: traffic classification is not like image style transfer; instance-level statistics can be label information.

## 3. Encrypted-Traffic Drift and Domain Adaptation

### 3.1 Deep learning for encrypted traffic classification in the face of data drift

**Paper:** Navid Malekghaini et al. "Deep learning for encrypted traffic classification in the face of data drift: An empirical study." Computer Networks 2023.  
**Link:** https://www.sciencedirect.com/science/article/pii/S1389128623000932

**Core idea.**

This paper studies data drift in encrypted traffic classification using real-world ISP datasets. It shows that model degradation occurs when old models are tested on newer traffic and analyzes how TLS header bytes and flow time-series features drift differently.

**Key relevance.**

This is one of the closest prior works to our diagnosis. It supports our motivation that temporal evaluation is necessary and that encrypted traffic models decay in production. Their emphasis on feature types is useful: traffic-shape features may be more robust than TLS-header-dependent features.

**Gap for our work.**

They study drift and architecture/feature robustness; our analysis goes deeper into class-conditional collapse, abrupt vs gradual collapse, absorber classes, and normalization effects.

### 3.2 CESNET-TLS-Year22

**Paper:** "CESNET-TLS-Year22: A year-spanning TLS network traffic dataset from backbone lines." Scientific Data 2024.  
**Link:** https://www.nature.com/articles/s41597-024-03927-4

**Core idea.**

CESNET-TLS-Year22 spans the full year of 2022 and is explicitly designed for time-aware evaluation of traffic classifiers. The paper emphasizes that evaluation should respect time order and that random/time-inconsistent splits can overestimate performance.

**Relevance.**

This paper justifies our M-2022-4 to M-2022-12 evaluation setup. It also gives dataset-level support for the idea that long-span traffic data enables studies of model stability, retraining strategies, and temporal robustness.

### 3.3 FDAN: Zero-relabelling mobile-app identification over drifted encrypted network traffic

**Paper:** Minghao Jiang et al. "Zero-relabelling mobile-app identification over drifted encrypted network traffic." Computer Networks 2023.  
**Link:** https://www.sciencedirect.com/science/article/pii/S1389128623001731

**Core idea.**

FDAN treats drifted encrypted traffic as a domain adaptation problem. It uses a feature generator, app predictor, and domain discriminators to learn domain-invariant features without target relabeling. The paper reports F1 improvements under app version, platform, and region shifts.

**Relevance.**

FDAN is important because it is an encrypted-traffic-specific zero-relabeling/domain-adaptation method. It is closer to our problem than generic CV TTA.

**Difference from our setup.**

FDAN is more like unsupervised domain adaptation with source and target data available during adaptation/training. Pure online TTA is stricter. If we use this line, we should be clear whether our method is:

- pure test-time adaptation;
- source-free target adaptation;
- or train-time / offline unsupervised domain adaptation.

### 3.4 FG-Net: Flow-level relationship with GNN for drifted traffic

**Paper:** "Accurate mobile-app fingerprinting using flow-level relationship with graph neural networks." Computer Networks 2022/2023.  
**Link:** https://www.sciencedirect.com/science/article/pii/S1389128622003577

**Core idea.**

FG-Net argues that packet-level information can have low stability under drift, while flow-level burst relationships are more stable. It builds a flow relationship graph and uses GNNs to improve robustness to ambiguous and drifted traffic.

**Relevance.**

This gives a different solution direction: do not only adapt the model; change the representation to use more stable traffic context. It supports the interpretation that abrupt collapse may happen because the model relies on unstable low-level features.

### 3.5 Deep Adaptation Network with Smooth Characteristic Function

**Paper:** Van Tong et al. "Encrypted Traffic Classification Through Deep Domain Adaptation Network With Smooth Characteristic Function." IEEE TNSM 2025.  
**Link:** https://pure.psu.edu/en/publications/encrypted-traffic-classification-through-deep-domain-adaptation-n-2/

**Core idea.**

This work uses a deep adaptation network and smooth characteristic functions to reduce source-target domain discrepancy for encrypted traffic classification, aiming to handle limited or unlabeled target data.

**Relevance.**

It supports the domain-adaptation framing for encrypted traffic classification. It is less directly about temporal class collapse, but useful in related work as evidence that source-target discrepancy is a recognized problem in encrypted traffic classification.

### 3.6 Drift-oriented self-evolving encrypted traffic classification

**Paper:** Zihan Chen et al. "Drift-oriented Self-evolving Encrypted Traffic Application Classification for Actual Network Environment." arXiv 2025.  
**Link:** https://arxiv.org/abs/2501.04246

**Core idea.**

This paper explicitly frames real encrypted traffic classification as feature concept drift caused by application updates. It proposes drift detection and self-evolving fine-tuning without exact labeled samples, reporting improved F1 and extended classifier lifetime.

**Relevance.**

This paper is close to the advisor's "abrupt collapse" concern because it links application updates to feature concept drift. It supports our claim that some drift is event-driven and may require detection plus model update, not only passive normalization.

## 4. What These Papers Mean for Our Current Direction

### 4.1 The advisor is not saying our current result is useless

The advisor's suggestion is more likely:

1. connect abrupt collapse to the continual TTA / drift literature;
2. test whether simple normalization adaptation, especially AdaBN, explains or mitigates part of the drift;
3. organize the existing results visually, especially by drift type.

This is consistent with our current norm ablation. We are not expected to magically solve all collapse with InstanceNorm. The useful contribution is to show which drift types each adaptation family can or cannot handle.

### 4.2 Clear taxonomy emerging from literature + our results

| Drift / failure type | Typical method family | Expected effect on TLS22 |
|---|---|---|
| Global covariate shift | AdaBN, BN adaptation, normalization statistics | May help if feature mean/variance shifts globally; unlikely to fix class collapse |
| Non-i.i.d. target stream | NOTE/PBRS, CoTTA memory/teacher mechanisms | Helps avoid unstable adaptation; still depends on useful pseudo-labels |
| Entropy/noisy-gradient instability | EATA, SAR | Helps prevent adaptation-induced collapse; not enough when source model already collapses classes |
| Class-conditional temporal collapse | Class-pair analysis, collapse-aware training/adaptation | Our main diagnosis; generic TTA weak |
| App-version / event-driven feature concept drift | Drift detection + domain adaptation / self-evolving update | Likely needed for abrupt collapse |

### 4.3 How to write the related-work narrative

Suggested paragraph logic:

1. Encrypted traffic classifiers suffer from temporal drift in production; prior empirical studies and CESNET-TLS-Year22 motivate time-aware evaluation.
2. Generic TTA methods such as Tent, CoTTA, EATA, SAR, and NOTE address unlabeled adaptation under distribution shift, non-stationary streams, forgetting, and noisy test data.
3. However, encrypted traffic drift includes class-conditional collapse and application-update-induced concept drift. In this setting, pseudo-label-based TTA can be unreliable because collapsed classes are confidently assigned to absorber classes.
4. Normalization-based adaptation such as AdaBN is a natural lightweight remedy for covariate shift. Our ablation shows it is insufficient for severe collapse, though it reveals different behavior across abrupt and gradual collapse.
5. Encrypted-traffic-specific domain adaptation works such as FDAN show that zero-relabeling adaptation is promising, but they often assume source-target access or offline adaptation rather than strict online TTA.

## 5. Concrete Next Experiments Suggested by This Survey

1. **AdaBN on BN checkpoint.**
   - Use `outputs/tls22_cnn_bn/best_model.pt`.
   - For each target month, recompute BN statistics on unlabeled target data, then evaluate.
   - Compare BN static vs BN + AdaBN on overall, stable, abrupt, gradual, absorber, degraded-noncollapsed groups.

2. **Visualization package for advisor.**
   - Line plot: macro-F1 over M7/M10/M12 for GN/IN/BN/LN/AdaBN.
   - Bar plot: M12 group delta vs GN.
   - Heatmap: per-class recall for collapse classes over months.
   - Pair plot/table: major absorber pairs and whether each method changes them.

3. **Literature table in paper.**
   - Columns: method, setting, target labels, source data needed, source domain available, online/offline, handles continual drift, handles class collapse.

4. **Explicit negative-result framing.**
   - Generic TTA and normalization adaptation handle certain covariate or stream-level shifts, but they do not repair severe class-conditional collapse in TLS22.
   - This motivates either drift detection + response, or domain-adaptive training/update methods with stronger supervision assumptions.

## 6. Short Takeaway

The most important reading conclusion is:

**CoTTA / NOTE / EATA / SAR explain why continual TTA is hard under non-stationary streams, while AdaBN explains a lightweight normalization-based adaptation baseline. Encrypted-traffic DA papers such as FDAN show that zero-relabeling adaptation is possible, but usually under a less strict offline DA setting. Our current TLS22 results suggest that abrupt and gradual class collapse are not solved by generic TTA or normalization alone; the useful contribution is to characterize this gap and show which drift types each method can or cannot handle.**

