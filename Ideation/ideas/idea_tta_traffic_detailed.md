# TTA-TC: Test-Time Adaptation for Temporally Robust Encrypted Traffic Classification

## Detailed Technical Proposal

---

## 1. Problem Formalization

**Setting**: A classifier $f_\theta: \mathcal{X} \to \mathcal{Y}$ trained on flows from time period $T_0$ is deployed to classify traffic at times $T_1, T_2, \ldots$. Due to concept drift, the data distribution shifts: $P_{T_0}(X, Y) \neq P_{T_k}(X, Y)$.

**Goal**: Adapt $f_\theta$ at test time using only unlabeled flows from $T_k$, without any access to training data from $T_0$ or labels from $T_k$.

**Input representation**: Each flow $x$ is represented as a tuple of three sequences from the first $N=30$ packets:
- **Packet sizes**: $\mathbf{s} = (s_1, s_2, \ldots, s_N) \in \mathbb{R}^N$
- **Directions**: $\mathbf{d} = (d_1, d_2, \ldots, d_N) \in \{-1, +1\}^N$
- **Inter-packet times**: $\mathbf{t} = (t_1, t_2, \ldots, t_N) \in \mathbb{R}_{\geq 0}^N$

Plus flow-level statistics: $\mathbf{z} = (\text{DURATION}, \text{BYTES}, \text{BYTES\_REV}, \text{PACKETS}, \text{PACKETS\_REV}, \text{DST\_ASN}, \ldots)$

This matches the CESNET DataZoo PPI (Per-Packet Information) format exactly.

---

## 2. Architecture: TTT-TC (Test-Time Training for Traffic Classification)

### 2.1 Y-Shaped Architecture

```
Input flow x = (s, d, t, z)
         │
    ┌─────┴──────┐
    │  Encoder    │ θ_e (shared, updated at test time)
    │  f_enc      │
    └──┬─────┬───┘
       │     │
  ┌────┴──┐ ┌┴────────┐
  │ Class │ │   SSL    │
  │ Head  │ │  Head    │ θ_s (updated at test time)
  │ g_cls │ │  g_ssl   │
  └───┬───┘ └────┬────┘
      │          │
   ŷ ∈ Y    L_ssl (self-supervised loss)
```

**Training phase** (on $T_0$ with labels):
$$\mathcal{L}_{\text{train}} = \mathcal{L}_{\text{cls}}(\theta_e, \theta_c) + \lambda \cdot \mathcal{L}_{\text{ssl}}(\theta_e, \theta_s)$$

**Test phase** (on $T_k$ without labels):
1. For each test batch $B$, compute $\mathcal{L}_{\text{ssl}}(B; \theta_e, \theta_s)$
2. Update $\theta_e \leftarrow \theta_e - \alpha \nabla_{\theta_e} \mathcal{L}_{\text{ssl}}$
3. Classify with updated encoder: $\hat{y} = g_{\text{cls}}(f_{\text{enc}}(x; \theta_e); \theta_c)$

### 2.2 Encoder Architecture Options

**Option A: 1D-CNN (lightweight, compatible with CESNET models)**

```
PPI Branch:           Flow Stats Branch:
s,d,t → Concat(3,30)  z → FC(64) → ReLU → FC(32)
  → Conv1d(3,64,k=3)         │
  → BN → ReLU               │
  → Conv1d(64,128,k=3)       │
  → BN → ReLU               │
  → GlobalAvgPool             │
  → FC(128)                  │
       └──────┬──────────────┘
              │ Concat
         FC(256) → FC(C)  [classification head]
         FC(256) → ...    [SSL head]
```

**Option B: Transformer (compatible with ET-BERT / YaTC style)**

```
PPI → Token Embedding (per-packet tokens)
   → Positional Encoding
   → Transformer Encoder (L=4, d=256, h=4)
   → [CLS] token → FC(C)  [classification head]
   → Masked tokens → FC(V) [SSL reconstruction head]
```

**Critical design decision**: Use **Group Normalization** or **Layer Normalization** instead of Batch Normalization. The failure mode analysis shows BN is the primary instability source for non-i.i.d. traffic.

---

## 3. Self-Supervised Auxiliary Tasks

Based on the comprehensive SSL task analysis, we propose three tasks ranked by feasibility:

### 3.1 Masked Packet Feature Prediction (MPFP) — PRIMARY

**Mechanism**: Randomly mask 15% of positions in the PPI sequences and predict the masked values.

**Formal definition**:
- Let $M \subset \{1, \ldots, N\}$ be the set of masked positions, $|M| = \lfloor 0.15 \cdot N \rfloor$
- Replace features at masked positions with a learnable [MASK] embedding
- SSL head predicts: packet size $\hat{s}_i$, direction $\hat{d}_i$, discretized IPT $\hat{t}_i$ for $i \in M$

**Loss**:
$$\mathcal{L}_{\text{MPFP}} = \frac{1}{|M|} \sum_{i \in M} \left[ \text{MSE}(\hat{s}_i, s_i) + \text{BCE}(\hat{d}_i, d_i) + \text{CE}(\hat{t}_i, \text{bin}(t_i)) \right]$$

where $\text{bin}(t_i)$ discretizes inter-packet times into 8 logarithmic bins (matching CESNET PHIST bins: 0-15ms, 16-31ms, 32-63ms, 64-127ms, 128-255ms, 256-511ms, 512-1024ms, >1024ms).

**Why this works for TTA**: When traffic distribution shifts (e.g., Google certificate change alters packet size at position 5), the reconstruction error on masked packet sizes increases, providing gradient signal to re-calibrate the encoder's representation. The encoder learns to model the current statistical relationships between packets — exactly what shifts under drift.

**Computational cost**: One forward + backward pass through encoder + lightweight MLP head. Comparable to a classification pass.

### 3.2 Packet Order Prediction (POP) — SECONDARY

**Mechanism**: Divide each flow's PPI into 3 segments. Shuffle two randomly chosen segments. Predict the correct ordering (6 permutations for 3 segments).

**Loss**:
$$\mathcal{L}_{\text{POP}} = \text{CE}(\hat{\pi}, \pi_{\text{true}})$$

where $\hat{\pi}$ is the predicted permutation class (1 of 6).

**Why this complements MPFP**: POP captures temporal flow structure (request-response patterns) that is orthogonal to the per-position feature statistics captured by MPFP. Different applications have distinct packet ordering patterns that may shift independently of packet size distributions.

**Computational cost**: Low — 6-way classification on the [CLS] token. Negligible compared to encoder forward pass.

### 3.3 Flow Statistics Reconstruction (FSR) — TERTIARY

**Mechanism**: From the PPI sequences alone, predict aggregated flow statistics (total bytes, duration, packet count ratios).

**Loss**:
$$\mathcal{L}_{\text{FSR}} = \text{MSE}(\hat{\mathbf{z}}, \mathbf{z})$$

where $\hat{\mathbf{z}}$ is the predicted flow statistics vector and $\mathbf{z}$ is the actual statistics.

**Why useful**: Forces the encoder to learn the relationship between micro-level (per-packet) and macro-level (flow) features. This relationship may shift when applications update.

### 3.4 Combined SSL Objective

$$\mathcal{L}_{\text{ssl}} = \mathcal{L}_{\text{MPFP}} + \alpha \cdot \mathcal{L}_{\text{POP}} + \beta \cdot \mathcal{L}_{\text{FSR}}$$

Recommended: $\alpha = 0.2$, $\beta = 0.1$ (MPFP dominates as it provides the densest gradient signal).

---

## 4. Adaptation Strategy: Class-Aware Selective TTA

The key empirical finding from Soukup et al. [3] is that drift is **highly non-uniform** across classes: 83/159 classes drifted at least once over one year, with drift events clustered in certain classes. This motivates selective, class-aware adaptation.

### 4.1 Per-Class Drift Detection

For each class $c$, maintain a running estimate of the class's entropy trajectory:

$$\bar{H}_c(t) = \frac{1}{|B_c|} \sum_{x \in B_c} H(f_\theta(x))$$

where $B_c = \{x : \arg\max f_\theta(x) = c\}$ is the set of flows predicted as class $c$ in the current window.

**Drift signal**: Compare $\bar{H}_c(t)$ against a baseline $\bar{H}_c(0)$ (from training data):

$$\Delta H_c(t) = \bar{H}_c(t) - \bar{H}_c(0)$$

If $\Delta H_c(t) > \tau_H$ for a class $c$, flag it as drifted.

### 4.2 Selective Adaptation Pipeline

```
For each incoming batch B of flows:

  Step 1: OOD Filtering
    Compute energy score E(x) = -log Σ_c exp(z_c(x)) for each flow
    Exclude flows where E(x) > E_threshold (potential unknown apps)

  Step 2: Entropy Filtering (SAR-style)
    Compute prediction entropy H(x) for each flow
    Exclude flows where H(x) > 0.4 * ln(C)

  Step 3: Prediction-Balanced Reservoir Sampling (NOTE-style)
    Add surviving flows to a PBRS buffer (size = 10*C)
    Sample a class-balanced mini-batch B' from the buffer

  Step 4: Class-Weighted SSL Loss
    For flows in B', compute MPFP loss
    Weight each flow's loss by inverse class frequency in the buffer:
      w_c = 1 / sqrt(n_c)  where n_c = count of class c in buffer

  Step 5: Selective Update
    For classes flagged as drifted (ΔH_c > τ_H):
      Update encoder with higher learning rate (α * γ, γ > 1)
    For stable classes:
      Apply Fisher regularization to constrain updates

  Step 6: Anti-Forgetting
    With probability p=0.01, stochastically restore each parameter
    to its source (T_0) value (CoTTA-style)

  Step 7: Teacher Update
    Update EMA teacher: θ_teacher ← m * θ_teacher + (1-m) * θ_e
    where m = 0.999
```

### 4.3 Handling Abrupt vs. Gradual Drift

Monitor the **rate of change** of $\Delta H_c(t)$:
- If $|\frac{d}{dt} \Delta H_c| > \tau_{\text{abrupt}}$: **abrupt drift** detected
  - Disable stochastic restoration for class $c$'s parameters
  - Increase learning rate by factor $\gamma = 5$
  - Increase adaptation steps to 10 per batch
- Otherwise: **gradual drift**
  - Maintain stochastic restoration
  - Normal learning rate
  - 1-3 adaptation steps per batch

---

## 5. Experimental Design

### 5.1 Datasets and Temporal Protocol

**Dataset 1: CESNET-QUIC22** (short-term evaluation)

| Config | Train | Val | Test |
|--------|-------|-----|------|
| Temporal-1W | W-2022-44 | 20% split from W-44 | W-2022-45 |
| Temporal-2W | W-2022-44 | 20% split from W-44 | W-2022-46 |
| Temporal-3W | W-2022-44 | 20% split from W-44 | W-2022-47 |
| Sequential | W-2022-44 | — | W-45 → W-46 → W-47 (continual) |

Key test: W-45 contains the Google certificate rotation event. The "Sequential" config tests continual TTA across 3 weeks.

**Dataset 2: CESNET-TLS-Year22** (long-term evaluation)

| Config | Train | Val | Test |
|--------|-------|-----|------|
| Temporal-1M | M-2022-1 to M-2022-3 | M-2022-4 | M-2022-5 to M-2022-12 |
| Temporal-6M | M-2022-1 to M-2022-3 | M-2022-4 | M-2022-9 |
| Temporal-11M | M-2022-1 to M-2022-3 | M-2022-4 | M-2022-12 |
| Sequential | M-2022-1 to M-2022-3 | — | M-4 → M-5 → ... → M-12 (continual) |

Use size "S" (25M samples) for development, "M" (75M) for final evaluation.

**API usage**:
```python
from cesnet_datazoo.datasets import CESNET_QUIC22
from cesnet_datazoo.config import DatasetConfig, AppSelection

dataset = CESNET_QUIC22("/data/", size="S")
config = DatasetConfig(
    dataset=dataset,
    apps_selection=AppSelection.ALL_KNOWN,
    train_period_name="W-2022-44",
    test_period_name="W-2022-45",
)
dataset.set_dataset_config_and_initialize(config)
train_loader = dataset.get_train_dataloader()
test_loader = dataset.get_test_dataloader()
```

### 5.2 Baselines

| # | Method | Category | Key Reference |
|---|--------|----------|---------------|
| B1 | Static (no adaptation) | Lower bound | — |
| B2 | Periodic retraining (weekly/monthly) | Upper bound (needs labels) | — |
| B3 | BN Adaptation (BN-1) | Statistics update | Schneider et al. 2020 |
| B4 | Tent | TTA (entropy) | Wang et al. ICLR 2021 |
| B5 | EATA | TTA (efficient) | Niu et al. ICML 2022 |
| B6 | CoTTA | TTA (continual) | Wang et al. CVPR 2022 |
| B7 | SAR | TTA (robust) | Niu et al. ICLR 2023 |
| B8 | NOTE | TTA (non-i.i.d.) | Gong et al. NeurIPS 2022 |
| B9 | Self-evolving | Pseudo-label | Chen et al. arXiv 2025 |
| B10 | TTA-TC (ours) | TTA (traffic-specific SSL) | This work |

B1-B3 need no training modification. B4-B8 are general TTA methods applied to the traffic classifier as-is. B9 is the closest traffic-specific prior work. B10 is our method.

### 5.3 Evaluation Metrics

| Metric | Definition | What it measures |
|--------|-----------|------------------|
| Accuracy@$T_k$ | Accuracy on test data from time $T_k$ | Point-in-time performance |
| Macro-F1@$T_k$ | Macro-averaged F1 at $T_k$ | Class-balanced performance |
| Accuracy Retention Ratio (ARR) | $\frac{\text{Acc}(T_k)}{\text{Acc}(T_0)}$ | How well performance is preserved |
| Area Under Retention Curve (AURC) | $\int_0^K \text{ARR}(T_k) dk$ | Overall temporal robustness |
| Per-Class Recall Decay | $\text{Recall}_c(T_k) - \text{Recall}_c(T_0)$ per class | Class-level drift impact |
| Adaptation Latency | Wall-clock time per batch adaptation | Deployment feasibility |
| GPU Memory | Peak memory during adaptation | Resource requirements |

### 5.4 Ablation Studies

| # | Ablation | Variables | Purpose |
|---|----------|-----------|---------|
| A1 | SSL task selection | MPFP alone / POP alone / FSR alone / MPFP+POP / all three | Best auxiliary task |
| A2 | Masking ratio | 5% / 10% / 15% / 25% / 50% | Optimal reconstruction difficulty |
| A3 | Adaptation depth | LN params only / last 2 layers / last 4 layers / full encoder | Cost-performance tradeoff |
| A4 | Class-aware vs. global | With/without per-class drift detection | Value of selectivity |
| A5 | Anti-forgetting | None / Fisher reg / stochastic restore / both | Error accumulation prevention |
| A6 | Batch size sensitivity | 16 / 32 / 64 / 128 / 256 | Robustness to small batches |
| A7 | Backbone architecture | 1D-CNN / Transformer (4L) / Transformer (8L) | Architecture sensitivity |
| A8 | Normalization type | BN / GN / LN | BN vs alternatives under drift |
| A9 | Adaptation frequency | Every batch / every 10 batches / every 100 batches | Cost reduction |
| A10 | Gradient alignment | Measure cos(∇L_ssl, ∇L_cls) across training | Validate theoretical condition |

### 5.5 Key Hypotheses

- **H1**: TTA-TC with MPFP achieves higher ARR than all general TTA methods (B4-B8) because traffic-specific SSL provides better-aligned gradients than generic entropy minimization.
- **H2**: Class-aware selective adaptation outperforms global adaptation because drift is non-uniform across classes.
- **H3**: Group/Layer Normalization is essential for traffic TTA due to bursty, non-i.i.d. batch composition.
- **H4**: Combined MPFP+POP objective outperforms either task alone by capturing both feature-level and structure-level shifts.
- **H5**: TTA-TC maintains ARR > 0.95 across 3 weeks on CESNET-QUIC22 and ARR > 0.90 across 6 months on CESNET-TLS-Year22.

---

## 6. Novelty Claims

1. **First TTA for traffic classification**: No prior work applies test-time adaptation to encrypted traffic classification. TTA is mature in CV (20+ methods) but completely unexplored in traffic analysis.

2. **Traffic-specific SSL auxiliary tasks for adaptation**: We design MPFP — a masked packet feature prediction task tailored to the PPI (packet size/direction/IPT) representation used by traffic classifiers. This is distinct from both ET-BERT's byte-level MBM (which operates on raw payload bytes) and generic entropy minimization.

3. **Class-aware selective TTA**: We combine per-class drift detection with selective adaptation, addressing the empirically confirmed non-uniform drift distribution across application classes.

4. **Comprehensive temporal evaluation**: First work to evaluate drift mitigation on CESNET-TLS-Year22 (1 year), enabling assessment of long-term TTA stability.

---

## 7. Implementation Plan

### Phase 1: Baseline Reproduction
- Set up cesnet-datazoo, load CESNET-QUIC22
- Reproduce Luxemburk et al. TMA 2023 baseline (train W-44, test W-45/46/47)
- Confirm 13.48% accuracy drop
- Implement 1D-CNN and Transformer backbones with GN/LN

### Phase 2: SSL Pre-training Tasks
- Implement MPFP, POP, FSR auxiliary tasks
- Joint training: $\mathcal{L}_{\text{cls}} + \lambda \mathcal{L}_{\text{ssl}}$
- Measure gradient alignment: $\cos(\nabla_{\theta_e} \mathcal{L}_{\text{cls}}, \nabla_{\theta_e} \mathcal{L}_{\text{ssl}})$
- Ablate across tasks, masking ratios

### Phase 3: TTA Integration
- Implement test-time SSL adaptation loop
- Integrate SAR-style sample filtering
- Implement PBRS buffer for non-i.i.d. handling
- Implement Fisher regularization + stochastic restoration

### Phase 4: Class-Aware Extension
- Implement per-class entropy tracking
- Implement selective adaptation with drift-type detection
- Integrate OOD filtering for unknown apps

### Phase 5: Full Evaluation
- Run all baselines on CESNET-QUIC22 (4 weeks)
- Run all baselines on CESNET-TLS-Year22 (12 months)
- Complete ablation studies
- Per-class analysis and visualization

---

## 8. Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| SSL gradient not aligned with classification | Medium | High | Ablate multiple tasks; measure alignment explicitly; fallback to entropy-only TTA |
| Error accumulation destroys model after months | Medium | High | Periodic reset + stochastic restoration + Fisher reg; checkpoint system |
| GN/LN reduces baseline accuracy vs. BN | Low | Medium | Compare BN vs GN/LN at training time; accept small accuracy tradeoff for stability |
| CESNET-QUIC22 drift too mild (only 4 weeks) | Low | Low | Primary long-term eval on TLS-Year22 (1 year) |
| Class imbalance corrupts adaptation | Medium | Medium | PBRS buffer + class-weighted loss; EATA redundancy filtering |
| Computational overhead too high | Low | Medium | Adapt only LN/GN params + SSL head; batch adaptation every K batches |

---

## 9. References

[1] Luxemburk, J. & Hynek, K. "Encrypted traffic classification: the QUIC case." TMA 2023.
[2] Malekghaini, N. et al. "Deep learning for encrypted traffic classification in the face of data drift." Computer Networks, 2023.
[3] Soukup, D. et al. "Drift-Based Dataset Stability Benchmark." arXiv:2512.23762, 2025.
[4] Chen, Z. et al. "Drift-oriented Self-evolving Encrypted Traffic Application Classification." arXiv:2501.04246, 2025.
[5] Jiang, M. et al. "Zero-relabelling mobile-app identification over drifted encrypted network traffic." Computer Networks, 2023.
[6] Yang, L. et al. "CADE: Detecting and Explaining Concept Drift Samples." USENIX Security, 2021.
[7] Han, D. et al. "OWAD: Anomaly Detection in the Open World." NDSS, 2023.
[8] Lin, X. et al. "ET-BERT." WWW, 2022.
[9] Zhao, R. et al. "YaTC: Yet Another Traffic Classifier." AAAI, 2023.
[10] Sun, Y. et al. "Test-Time Training with Self-Supervision." ICML, 2020.
[11] Wang, D. et al. "Tent: Fully Test-Time Adaptation by Entropy Minimization." ICLR, 2021.
[12] Niu, S. et al. "EATA: Efficient Test-Time Model Adaptation without Forgetting." ICML, 2022.
[13] Wang, Q. et al. "CoTTA: Continual Test-Time Domain Adaptation." CVPR, 2022.
[14] Niu, S. et al. "SAR: Towards Stable Test-Time Adaptation in Dynamic Wild World." ICLR, 2023.
[15] Gong, T. et al. "NOTE: Robust Continual Test-time Adaptation Against Temporal Correlation." NeurIPS, 2022.
[16] Zhao, H. et al. "On Pitfalls of Test-Time Adaptation." ICML, 2023.
[17] Hynek, K. et al. "CESNET-TLS-Year22." Scientific Data, 2024.
[18] Gandelsman, Y. et al. "Test-Time Training with Masked Autoencoders." NeurIPS, 2022.
[19] Hang, Z. et al. "Flow-MAE." RAID, 2023.
[20] Wang, Q. et al. "Lens: A Foundation Model for Network Traffic." 2024.
