# Failure Modes and Pitfalls of Test-Time Adaptation (TTA) Applied to Network Traffic Classification

## Executive Summary

This report synthesizes findings from the TTA literature on systematic failure modes of test-time adaptation methods, and analyzes how each failure mode manifests---often with amplified severity---in the domain of network traffic classification. The analysis draws primarily from Zhao et al. (ICML 2023), NOTE (NeurIPS 2022), SAR (ICLR 2023), CoTTA (CVPR 2022), EATA (ICML 2022), and related work. For each failure mode, traffic-specific mitigations are proposed.

---

## 1. General TTA Failure Modes

### 1.1 Three Pitfalls Identified by Zhao et al. (ICML 2023)

Zhao et al. conducted a systematic benchmarking study (TTAB) that revealed three pervasive pitfalls in prior TTA work:

| Pitfall | Description |
|---------|-------------|
| **Model selection difficulty** | Hyperparameters of TTA methods (learning rate, batch size, update frequency) are highly sensitive to the specific test scenario. Online batch dependency makes model selection exceedingly difficult because there is no held-out validation set at test time. |
| **Pre-trained model dependence** | TTA effectiveness varies greatly depending on the quality and architecture of the pre-trained model. A method that works well with one backbone may fail with another. |
| **Distribution shift limitations** | Even with oracle-based hyperparameter tuning, no existing TTA method can address all common classes of distribution shift. Mixed distribution shifts, where multiple types co-occur, are particularly problematic. |

**Key finding**: TENT (Wang et al., ICLR 2021) exhibits empirical sensitivity and batch dependency; it performs worse than the unadapted source model when batch size = 1. TTA may fail to improve or actively harm model performance under: (1) mixed distribution shifts, (2) small batch sizes, and (3) online imbalanced label distributions.

### 1.2 Entropy Minimization Collapse

Entropy minimization is the dominant self-supervised loss in TTA (used by TENT, EATA, SAR, and others). Its failure modes are well-documented:

**Mechanism of collapse**: The entropy objective is minimized even when the model predicts a single class for all inputs. This is a trivial solution: setting f(x)_y = 1 for a fixed class y minimizes entropy without meaningful learning. During the collapse process, model gradients undergo an initial explosion followed by rapid degradation.

**Conditions triggering collapse**:
- **Batch size = 1**: With a single sample per iteration, the model simply amplifies the largest logit until the output becomes one-hot. TENT at batch size 1 is consistently worse than the unadapted source model.
- **Noisy / out-of-distribution samples**: Samples with large gradient norms disrupt adaptation and drive the model toward trivial solutions.
- **Severe distribution shift**: Under severity level 5 corruptions (e.g., ImageNet-C), entropy minimization on group-normalized models tends to collapse.
- **Irreducible uncertainty**: The contradiction between pursuing zero entropy and the existence of genuinely ambiguous samples creates instability.

### 1.3 Batch Size Requirements

TTA methods that rely on batch normalization statistics estimation require sufficiently large and diverse batches:

- **Class diversity, not just size, is the critical factor** (AAAI 2024, "Unraveling Batch Normalization for Realistic TTA"): Inaccurate target statistics stem primarily from substantially reduced class diversity within mini-batches, not merely from small sample counts.
- **Minimum practical batch size**: Most TTA methods require batch sizes of at least 64--200 to produce stable BN statistics. Below this threshold, the estimated mean and variance are unreliable.
- **Batch-agnostic normalization** (group norm, layer norm) provides greater stability but does not fully eliminate collapse.

### 1.4 Class-Imbalanced Test Batches

When test batches have skewed label distributions:

- BN statistics become biased toward the majority class's activation patterns, producing normalization that actively harms minority-class representations.
- Entropy minimization adapts primarily to majority-class samples, since they dominate the gradient signal. Minority classes are increasingly misclassified.
- Pseudo-labels become self-reinforcing for majority classes: correct pseudo-labels for the dominant class strengthen its representation, while incorrect pseudo-labels for minority classes accumulate.
- SAR's sample selection (entropy threshold tau = 0.4 * ln(C)) partially addresses this but does not explicitly account for class balance.
- ROID explicitly maintains prediction diversity to prevent collapse toward a subset of classes.

---

## 2. Non-i.i.d. Test Data (NOTE, NeurIPS 2022)

### 2.1 The Problem

Most TTA methods assume i.i.d. test data. Real-world test streams are temporally correlated (non-i.i.d.): consecutive samples share similar characteristics because they arise from the same context (e.g., the same driving scene, the same user session, the same application burst).

**NOTE's key finding**: Most existing TTA methods fail dramatically under temporal correlation. Adapting to temporally correlated batches causes overfitting to the temporal distribution, destroying generalization. Error rates of baseline methods deteriorate as the degree of temporal correlation increases.

### 2.2 Two Mechanisms of Failure

1. **Batch normalization bias**: When a batch is dominated by samples from one class or one temporal context, BN statistics reflect that context rather than the true test distribution. This removes instance-wise variations that are actually useful for prediction and introduces class-prior bias.
2. **Entropy minimization on correlated samples**: Minimizing entropy on a batch of similar samples reinforces a narrow region of the feature space, causing the model to overfit to that local distribution.

### 2.3 NOTE's Solutions

- **Instance-Aware Batch Normalization (IABN)**: Corrects normalization on a per-instance basis for out-of-distribution samples.
- **Prediction-Balanced Reservoir Sampling (PBRS)**: Maintains a reservoir buffer that simulates i.i.d. data streams from non-i.i.d. sources in a class-balanced manner.
- NOTE achieves 21.1% error on CIFAR10-C (15.1% lower than the best baseline) and supports single-sample inference without batching.

---

## 3. SAR: Handling Unreliable Samples (ICLR 2023)

### 3.1 Root Cause Analysis

Niu et al. (SAR) identified two root causes of TTA instability:

1. **Batch normalization is a crucial stability obstacle**: BN layers are inherently sensitive to batch composition and size, making them a primary source of TTA failure.
2. **Noisy samples with large gradients cause collapse**: Even a few outlier samples can produce gradient norms that overwhelm the adaptation signal, pushing the model toward trivial solutions.

### 3.2 SAR's Two-Pronged Approach

| Component | Mechanism |
|-----------|-----------|
| **Reliable sample filtering** | Exclude samples with entropy > tau = 0.4 * ln(C) from the adaptation loss. These high-entropy samples are deemed unreliable and produce noisy gradients. |
| **Sharpness-aware optimization** | Optimize model weights toward flat minima in the entropy landscape. Flat minima are more robust to perturbations from remaining noisy samples. Uses a SAM-style inner maximization step before the outer minimization. |

### 3.3 Practical Implications

- SAR demonstrates that replacing BN with batch-agnostic normalization (GN, LN) improves stability but does not eliminate failure.
- The entropy filtering threshold is tied to the number of classes C. For traffic classification with, e.g., 100+ application classes, the threshold becomes more permissive (tau = 0.4 * ln(100) = 1.84 nats), potentially admitting more noisy samples.

---

## 4. Error Accumulation in Continual TTA

### 4.1 The Core Problem

In continual TTA, the model adapts online over extended periods without resetting. Pseudo-labels derived from the model's own predictions introduce errors that compound over time:

1. Incorrect pseudo-labels update the model in the wrong direction.
2. The shifted model produces even more incorrect pseudo-labels.
3. This positive feedback loop leads to exponential error growth and eventual model collapse.

**Empirical timelines for collapse**:
- TENT on CIFAR-10-C: error rises from 19.3% at visit 1 to 87.8% at visit 20 in a continual setting.
- On the CCC (Continually Changing Corruptions) benchmark, many methods collapse to single-digit accuracy.
- Accuracy drops dramatically after approximately 60,000 adaptation steps without mitigation.
- RDumb requires periodic resets every ~1,000 batches to remain functional.

### 4.2 CoTTA's Stochastic Restoration (CVPR 2022)

CoTTA addresses error accumulation through two mechanisms:

| Mechanism | Details |
|-----------|---------|
| **Weight-averaged and augmentation-averaged pseudo-labels** | A teacher model (exponential moving average of student weights) provides smoothed pseudo-labels. Multiple augmentations of each sample are averaged to reduce prediction noise. This reduces error from 20.7% to 17.4% on CIFAR10-C. |
| **Stochastic restoration** | At each iteration, each neuron has a probability p = 0.01 of being reset to its source pre-trained value. This prevents catastrophic forgetting by continuously injecting source knowledge. Reduces error further to 16.2%. |

**Limitations of CoTTA**:
- The stochastic nature of restoration may randomly discard parameters that have adapted well, losing useful adaptation.
- Assumes i.i.d. test data within each domain.
- Updates the entire model, increasing computational cost.

### 4.3 EATA's Fisher Regularization (ICML 2022)

EATA addresses both efficiency and forgetting:

| Component | Mechanism |
|-----------|-----------|
| **Active sample selection** | Only update on low-entropy (reliable) samples. High-entropy samples produce noisy gradients and are excluded. Also excludes redundant samples similar to previously seen ones. |
| **Fisher regularization** | Estimates parameter importance via the Fisher information matrix (computed from test samples with pseudo-labels). Important parameters are constrained from large changes, preventing catastrophic forgetting of source knowledge. Loss: L = L_entropy + lambda * sum_i F_i * (theta_i - theta_i^source)^2 |

**Practical insight**: EATA's Fisher regularization acts as an elastic weight consolidation mechanism, anchoring the model to its source configuration while allowing adaptation in less critical parameter subspaces.

### 4.4 Long-Term Degradation Timeline

| Adaptation Duration | Typical Outcome |
|---------------------|-----------------|
| 1--1,000 batches | Generally beneficial adaptation; error decreases. |
| 1,000--10,000 batches | Without mitigation, error begins increasing. Periodic resets (e.g., every 1,000 batches) can maintain performance. |
| 10,000--60,000 batches | Significant degradation without regularization or resets. Fisher regularization or stochastic restoration can extend this window. |
| >60,000 batches | Full model collapse likely without aggressive mitigation. Models predict a single class for all inputs. |

---

## 5. Traffic-Specific Challenges for TTA

Network traffic classification presents a unique combination of challenges that amplify nearly every known TTA failure mode.

### 5.1 Non-i.i.d. Batch Composition

**The problem**: Network traffic is inherently bursty. A user watching a YouTube video generates hundreds of consecutive flows from that application. A corporate VPN session produces long bursts of VPN-encapsulated traffic. A bulk download generates a sustained stream of one application's flows.

**Impact on TTA mechanisms**:

| TTA Component | Effect of Bursty Traffic |
|---------------|-------------------------|
| **BN statistics estimation** | A batch of 64 flows where 58 are YouTube produces mean/variance estimates dominated by YouTube's activation patterns. These statistics are inaccurate for other applications and will degrade classification of the minority flows in the batch. |
| **Entropy minimization convergence** | The loss is dominated by YouTube flows. The model adapts its parameters to minimize entropy on YouTube predictions, potentially at the cost of other applications. This is equivalent to a severe label shift within each batch. |
| **Pseudo-label quality** | Pseudo-labels for the dominant application are likely correct (high confidence from many similar samples). Pseudo-labels for minority flows in the batch are unreliable because BN statistics are skewed and the model has over-adapted to the dominant class. |

**Severity assessment**: This is arguably the most critical failure mode for traffic TTA. Unlike image classification where test batches can be shuffled, traffic flows arrive in temporal order and cannot be arbitrarily reordered without introducing latency.

### 5.2 Extreme Class Imbalance

**The problem**: In real networks, traffic follows a heavy-tailed distribution. A small number of applications (Google, Facebook/Meta, Netflix, Amazon, Microsoft) generate the vast majority of traffic volume. Long-tail applications (specialized enterprise tools, niche streaming services, IoT protocols) produce very few flows.

**Quantitative context**: Malekghaini et al. (2022, 2023) studied ISP-scale traffic and found that class imbalance ratios can exceed 1000:1 between the most and least common applications.

**Impact on TTA when minority classes drift but majority classes do not**:
- The adaptation signal is overwhelmed by majority-class flows, which do not need adaptation. The model wastes its update budget reinforcing already-correct majority-class predictions.
- Minority-class flows that have genuinely drifted are either excluded by entropy-based filtering (because they appear as high-entropy outliers) or drowned out by the dominant gradient signal.
- Over time, the model becomes increasingly biased toward majority classes, and minority-class accuracy degrades---the opposite of what adaptation should achieve.

**Impact when majority class dominates the adaptation signal**:
- BN statistics reflect majority-class activations, normalizing features in a way that is optimal for Google/Facebook traffic but suboptimal for everything else.
- Entropy minimization converges to a state where majority-class predictions are extremely confident while minority-class predictions become increasingly uncertain.
- Pseudo-label noise is concentrated in minority classes, creating a vicious cycle of degradation for exactly the classes that most need adaptation.

### 5.3 Abrupt vs. Gradual Drift

Network traffic experiences both types of drift, but from distinct causes:

| Drift Type | Causes in Traffic | Characteristics |
|------------|-------------------|-----------------|
| **Abrupt** | TLS certificate rotation; CDN migration (an app moves from Akamai to Cloudflare); protocol version upgrade (TLS 1.2 to 1.3); JA3/JA4 fingerprint change due to browser update | Feature values change suddenly across all flows of the affected application. The change is permanent (not transient). May affect handshake features, packet-length sequences, or SNI patterns simultaneously. |
| **Gradual** | Application version updates (incremental UI/API changes); user behavior shifts (seasonal patterns); network infrastructure changes (new routing, capacity upgrades) | Feature distributions evolve slowly over weeks/months. Individual flow features remain within historical ranges but the aggregate distribution shifts. |

**How TTA should handle each type differently**:

- **For abrupt drift**: The model needs to rapidly recalibrate affected parameters. Stochastic restoration (CoTTA-style) is counterproductive here because it fights against legitimate adaptation. Instead, a triggered reset of affected class representations or a burst of high-learning-rate updates is appropriate. The challenge is detecting that an abrupt shift has occurred (vs. a few outlier flows).

- **For gradual drift**: Slow, continuous adaptation is appropriate. Fisher regularization (EATA) or EMA-based teacher models (CoTTA) work well here because they prevent over-correction while allowing incremental adjustment. Stochastic restoration can help prevent long-term forgetting.

**The challenge for traffic**: Both types co-occur. A certificate rotation (abrupt) for one app may happen while another app undergoes gradual feature drift. A single TTA strategy optimized for one type will fail on the other.

### 5.4 Open-Set Issues: New Applications

**The problem**: New applications continuously appear on the network. A model trained on 100 applications will encounter flows from applications 101, 102, etc. at test time. These are truly unknown classes with no representation in the training data.

**How TTA behaves with unknown classes**:

1. **Entropy minimization on unknown classes is harmful**: The model will attempt to minimize entropy on flows from unknown applications, forcing them into the closest known class. This corrupts the model's representation of that known class.

2. **BN statistics contamination**: Unknown-class flows have feature distributions that differ from all training classes. Including them in BN statistics estimation shifts the normalization in unpredictable ways, degrading classification of known classes.

3. **Pseudo-label poisoning**: Unknown-class flows receive incorrect pseudo-labels (they are assigned to some known class). These incorrect labels then update the model, reinforcing the wrong association and degrading the decision boundary for the incorrectly assigned class.

4. **Confidence miscalibration**: Standard TTA has no mechanism to say "I don't know." Entropy minimization actively fights against appropriate uncertainty, making the model confidently wrong on unknown classes.

**Severity for traffic classification**: This is a persistent, ongoing challenge. The Internet application ecosystem is continuously evolving. Unlike image classification where the set of objects is relatively stable, the set of network applications changes monthly. New apps, app versions using new protocols, and ephemeral services constantly appear.

---

## 6. Mitigation Strategies for Traffic Classification

### 6.1 Sample Filtering Strategies

| Strategy | Mechanism | Traffic-Specific Implementation |
|----------|-----------|-------------------------------|
| **Entropy-based filtering (SAR)** | Exclude samples with entropy > tau = 0.4 * ln(C). | Set C to the number of application classes. For C=100, tau = 1.84 nats. Flows with higher entropy are excluded from adaptation. |
| **Gradient-norm filtering (SAR)** | Exclude samples whose gradient norms exceed a threshold. | Monitor gradient norms during adaptation and exclude flows that would produce outsized parameter updates. |
| **PLPD-based selection (DeYO)** | Use Pseudo-Label Probability Difference to assess whether predictions rely on meaningful features rather than shortcuts. | Adapt to traffic classification by measuring prediction stability under feature perturbation (e.g., masking packet-length features and checking if the prediction changes). |
| **Energy-score filtering (UniEnt)** | Use energy scores to separate in-distribution from OOD samples. | Compute energy scores for each flow; flows with energy below threshold are in-distribution and safe for adaptation. High-energy flows may be from unknown applications and should be excluded. |
| **Redundancy filtering (EATA)** | Exclude samples that are highly similar to previously seen ones. | For traffic, skip adaptation on flows that are near-duplicates of recently processed flows (common in bursty traffic). This reduces computational cost and prevents over-adaptation to one application. |

### 6.2 Batch Composition Normalization

| Strategy | Mechanism | Traffic-Specific Implementation |
|----------|-----------|-------------------------------|
| **Prediction-Balanced Reservoir Sampling (PBRS, from NOTE)** | Maintain a buffer that stores samples in a class-balanced manner, using predicted labels to balance. | Maintain a reservoir of recent flows balanced across predicted application classes. When adapting, sample from the reservoir rather than using the raw temporal stream. Buffer size: at least 10 * C flows (e.g., 1,000 for 100 classes). |
| **Test-time EMA (TEMA)** | Use exponential moving average of BN statistics across batches to incorporate historical class diversity. | Maintain running EMA of BN mean/variance with a slow momentum (e.g., 0.999). This smooths out the burstiness of traffic and prevents any single burst from dominating statistics. |
| **Memory-based BN (MemBN)** | Maintain a queue of recent batch statistics and aggregate them for more stable normalization. | Store the last K batches' statistics (K = 50--100). Aggregate via weighted average, giving less weight to batches with low class diversity (detectable via pseudo-label distribution). |
| **Batch-agnostic normalization** | Replace BN with Group Norm (GN) or Layer Norm (LN) at training time. | Preferred approach for traffic classifiers. GN/LN do not depend on batch statistics and are inherently robust to bursty, non-i.i.d. test streams. Requires training the model with GN/LN from scratch. |

### 6.3 Class-Balanced Adaptation

| Strategy | Mechanism | Traffic-Specific Implementation |
|----------|-----------|-------------------------------|
| **Per-class learning rates** | Scale the learning rate inversely with class frequency in the adaptation buffer. | Track the number of flows per application class in a sliding window. Weight each flow's contribution to the loss by 1/sqrt(n_class), where n_class is the number of flows from that class in the window. |
| **Diversity-aware weighting (ROID)** | Explicitly maintain prediction diversity to prevent collapse toward dominant classes. | Monitor the entropy of the predicted class distribution across recent flows. If diversity drops (one class dominates predictions), increase the weight of minority-class flows and decrease majority-class weight. |
| **Class-conditional BN statistics** | Maintain separate BN statistics per predicted class. | For each application class, maintain independent running mean/variance. During inference, use the statistics corresponding to the predicted class. This prevents majority-class statistics from corrupting minority-class normalization. |
| **Label distribution shift correction (DART)** | Estimate and correct for label shift at test time. | Estimate the current test-time class prior using a sliding window of pseudo-labels. Apply Bayesian correction: P(y|x)_corrected = P(y|x)_model * P_test(y) / P_train(y). This counteracts the natural imbalance of traffic. |

### 6.4 Confidence Thresholding

| Strategy | Mechanism | Traffic-Specific Implementation |
|----------|-----------|-------------------------------|
| **Entropy threshold for adaptation** | Only adapt on flows with entropy below a class-count-dependent threshold. | Use SAR's tau = 0.4 * ln(C). For a 100-class traffic classifier, only adapt on flows with entropy < 1.84 nats (roughly, flows where the model assigns >50% probability to one class). |
| **Softmax confidence threshold** | Only adapt when max(softmax(logits)) exceeds a threshold. | Set threshold empirically (e.g., 0.7). Flows with lower confidence are classified using the current model but do not contribute to parameter updates. |
| **Conformal prediction sets (CUI)** | Generate prediction sets with guaranteed coverage; only adapt when the prediction set is small (high certainty). | For each flow, produce a conformal prediction set. If the set contains more than k classes (e.g., k = 3), the prediction is too uncertain for safe adaptation. |
| **Open-set detection threshold** | Use energy scores or OOD detectors to identify flows from unknown applications; exclude them from adaptation entirely. | Train an OOD detector alongside the traffic classifier. Flows flagged as OOD are logged for human review but do not update the model. This prevents pseudo-label poisoning from unknown applications. |

### 6.5 Reset / Restoration Mechanisms

| Strategy | Mechanism | Traffic-Specific Implementation |
|----------|-----------|-------------------------------|
| **Periodic reset (RDumb)** | Reset all parameters to source values every T batches. | Set T based on the expected rate of drift. For traffic, T = 500--2,000 batches (approximately hours to a day of traffic, depending on volume). Simple but effective as a baseline. |
| **Stochastic restoration (CoTTA)** | Reset each neuron with probability p per iteration. | Use p = 0.01 (1% of neurons per iteration). This continuously injects source knowledge, preventing long-term forgetting. However, it fights against rapid adaptation to abrupt drift. |
| **Adaptive selective reset (ASR)** | Dynamically determine when and where to reset based on collapse risk indicators. | Monitor: (1) prediction diversity (entropy of class distribution over recent flows), (2) gradient norm trends, (3) loss trajectory. Trigger a reset when diversity drops below threshold or gradient norms spike. Only reset parameters with high Fisher information change. |
| **Drift-type-aware reset** | Use different strategies for abrupt vs. gradual drift. | Detect abrupt drift via sudden increase in average entropy or sudden change in feature statistics (e.g., a jump in mean packet length for one class). For abrupt drift: high learning rate, no stochastic restoration. For gradual drift: low learning rate, Fisher regularization, stochastic restoration. |
| **Class-conditional reset** | Reset parameters associated with specific classes that show signs of degradation. | Monitor per-class entropy over time. If a class's average entropy increases sharply (suggesting drift or confusion), reset only the classifier head weights for that class while preserving the shared feature extractor. |

---

## 7. Integrated Mitigation Framework for Traffic TTA

Based on the analysis above, the following integrated framework addresses the unique challenges of network traffic classification:

### Architecture-Level Requirements (Training Time)
1. **Use Group Normalization or Layer Normalization** instead of Batch Normalization. This eliminates the single largest source of TTA instability for bursty, non-i.i.d. traffic.
2. **Train with an OOD detection head** (e.g., energy-based) to enable open-set rejection at test time.
3. **Use a modular classifier head** where per-class weights can be independently reset or updated.

### Online Adaptation Pipeline (Test Time)

```
For each incoming batch of flows:
  1. OOD Filtering: Compute energy scores. Exclude flows with energy > E_threshold.
     -> These may be unknown applications. Log them separately.

  2. Entropy Filtering: Compute entropy for remaining flows. Exclude flows with
     entropy > 0.4 * ln(C).
     -> These are unreliable for adaptation.

  3. Reservoir Buffering: Add surviving flows to a prediction-balanced reservoir
     (PBRS). Sample a balanced mini-batch from the reservoir for adaptation.
     -> This addresses burstiness and class imbalance.

  4. Class-Weighted Loss: Compute entropy loss with per-class weighting
     (inverse frequency in the reservoir).
     -> This prevents majority classes from dominating adaptation.

  5. Sharpness-Aware Update: Apply SAR-style optimization (inner max + outer min)
     with Fisher regularization to prevent forgetting.
     -> This stabilizes adaptation and prevents collapse.

  6. Drift Monitoring: Track per-class average entropy and prediction diversity.
     - If prediction diversity drops below threshold -> trigger selective reset.
     - If a class's entropy spikes abruptly -> increase that class's learning rate
       and disable stochastic restoration for related parameters.
     - If adaptation has run > T_max batches without reset -> trigger periodic reset.

  7. Teacher Update: Update EMA teacher model for pseudo-label generation
     (CoTTA-style). Use teacher predictions for pseudo-labeling rather than
     the student model directly.
```

### Monitoring and Safety Rails

- **Collapse detection**: Monitor the entropy of the predicted class distribution. If H(predicted class distribution) < 0.5 * H(uniform), the model is collapsing toward a few classes. Trigger a full reset.
- **Performance estimation**: Use the teacher-student prediction agreement rate as a proxy for accuracy. A sudden drop in agreement suggests degradation.
- **Periodic checkpoint**: Save model weights every T_save batches. If collapse is detected, restore from the last known-good checkpoint rather than the source model.

---

## 8. Summary Table: Failure Modes and Traffic-Specific Severity

| Failure Mode | General TTA Severity | Traffic-Specific Severity | Why Worse for Traffic |
|---|---|---|---|
| Entropy minimization collapse | High | Very High | Small effective class diversity per batch due to burstiness |
| Small batch size | High | Moderate | Traffic batches can be made large (high flow rates), but class diversity remains low |
| Non-i.i.d. test data | High | Critical | Traffic is inherently temporally correlated and bursty; flows from the same app/session arrive in clusters |
| Class imbalance | Moderate | Critical | Power-law distribution of traffic; majority apps dominate 80-90% of flows |
| Abrupt distribution shift | Moderate | High | Certificate rotations, CDN migrations, and protocol upgrades cause sudden feature changes |
| Gradual distribution shift | Low-Moderate | Moderate | App updates cause slow drift; well-handled by EMA-based methods |
| Open-set / unknown classes | Moderate | Critical | Continuous appearance of new applications; no closed-world assumption is valid for the Internet |
| Error accumulation | High | Very High | Traffic classification runs 24/7 with no natural stopping point; adaptation must be indefinitely stable |
| Catastrophic forgetting | High | Very High | Source knowledge about stable apps must be preserved while adapting to drifting ones |

---

## References

1. Zhao, H., Liu, Y., Alahi, A., & Lin, T. (2023). On Pitfalls of Test-Time Adaptation. ICML 2023. [Paper](https://proceedings.mlr.press/v202/zhao23d/zhao23d.pdf) | [Code](https://github.com/LINs-lab/ttab)

2. Gong, T., Jeong, J., Kim, T., Kim, Y., Shin, J., & Lee, S.-J. (2022). NOTE: Robust Continual Test-time Adaptation Against Temporal Correlation. NeurIPS 2022. [Paper](https://papers.neurips.cc/paper_files/paper/2022/file/ae6c7dbd9429b3a75c41b5fb47e57c9e-Paper-Conference.pdf) | [arXiv](https://arxiv.org/abs/2208.05117)

3. Niu, S., Wu, J., Zhang, Y., Wen, Z., Chen, Y., Zhao, P., & Tan, M. (2023). Towards Stable Test-Time Adaptation in Dynamic Wild World. ICLR 2023 (Oral). [Paper](https://openreview.net/pdf?id=g2YraF75Tj) | [Code](https://github.com/mr-eggplant/SAR)

4. Wang, Q., Fink, O., Van Gool, L., & Dai, D. (2022). Continual Test-Time Domain Adaptation. CVPR 2022. [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Continual_Test-Time_Domain_Adaptation_CVPR_2022_paper.pdf) | [Code](https://github.com/qinenergy/cotta)

5. Niu, S., Wu, J., Zhang, Y., Chen, Y., Zheng, S., Zhao, P., & Tan, M. (2022). Efficient Test-Time Model Adaptation without Forgetting. ICML 2022. [Paper](https://proceedings.mlr.press/v162/niu22a.html) | [Code](https://github.com/mr-eggplant/EATA)

6. Wang, D., Shelhamer, E., Liu, S., Olshausen, B., & Darrell, T. (2021). Tent: Fully Test-Time Adaptation by Entropy Minimization. ICLR 2021. [Paper](https://arxiv.org/abs/2006.10726) | [Code](https://github.com/DequanWang/tent)

7. Su, Y., & Xu, Z. (2024). Unraveling Batch Normalization for Realistic Test-Time Adaptation. AAAI 2024. [Paper](https://arxiv.org/html/2312.09486v3)

8. Lee, J., Jung, D., Lee, S., Park, J., Shin, J., Hwang, U., & Yoon, S. (2024). Entropy is not Enough for Test-Time Adaptation: From the Perspective of Disentangled Factors (DeYO). ICLR 2024 Spotlight. [Project](https://whitesnowdrop.github.io/DeYO/) | [Code](https://github.com/Jhyun17/DeYO)

9. Marsden, R. A., Dobler, M., & Yang, B. (2024). Universal Test-Time Adaptation through Weight Ensembling, Diversity Weighting, and Prior Correction (ROID). WACV 2024.

10. Malekghaini, M. R., et al. (2023). Deep Learning for Encrypted Traffic Classification in the Face of Data Drift: An Empirical Study. Computer Networks. [Paper](https://www.sciencedirect.com/science/article/abs/pii/S1389128623000932)

11. Zheng, Z., et al. (2025). Drift-oriented Self-evolving Encrypted Traffic Application Classification for Actual Network Environment. [arXiv](https://arxiv.org/html/2501.04246v1)

12. Wei, M., et al. (2025). When and Where to Reset Matters for Long-Term Test-Time Adaptation. [arXiv](https://arxiv.org/abs/2603.03796)

13. UniEnt: Unified Entropy Optimization for Open-Set Test-Time Adaptation. [arXiv](https://arxiv.org/abs/2404.06065)

14. Lee, J., et al. (2024). Ranked Entropy Minimization for Continual Test-Time Adaptation. ICML 2025. [arXiv](https://arxiv.org/html/2505.16441v1)
