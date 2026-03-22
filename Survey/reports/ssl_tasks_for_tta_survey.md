# Self-Supervised Learning Tasks in Network Traffic Classification: Feasibility as TTA Auxiliary Objectives

## Technical Survey Report

---

## 1. ET-BERT (WWW 2022)

**Paper**: Lin et al., "ET-BERT: A Contextualized Datagram Representation with Pre-training Transformers for Encrypted Traffic Classification," The Web Conference 2022.
**Source**: https://dl.acm.org/doi/10.1145/3485447.3512217 | https://arxiv.org/abs/2202.06335
**Code**: https://github.com/linwhitehat/ET-BERT

### 1.1 Datagram2Token Representation

ET-BERT introduces a three-stage tokenization pipeline:

1. **BURST Generator**: Extracts unidirectional packet sequences (client-to-server or server-to-client) from session flows, identified by the 5-tuple (IP_src:PORT_src, IP_dst:PORT_dst, Protocol). A BURST represents time-adjacent packets in a single direction within one session.

2. **BURST2Token (Bi-gram Model)**: Hexadecimal packet payloads are encoded using a bi-gram model where each unit consists of two adjacent bytes. For example, the byte sequence `0x45 0x00 0x00 0x34` is tokenized as `4500, 0000, 0034, ...`.

3. **Vocabulary Construction via BPE**: Byte-Pair Encoding builds a vocabulary of size |V| = 65,536 (covering the 0x0000 to 0xFFFF range). Special tokens: `[CLS]`, `[SEP]`, `[PAD]`, `[MASK]`.

**Input format**: Each BURST is divided into two sub-BURSTs (segments A and B), separated by `[SEP]` tokens. Maximum sequence length: 512 tokens. For flow-level fine-tuning, M=5 consecutive packets are stitched together.

### 1.2 Masked BURST Modeling (MBM)

MBM is directly analogous to BERT's Masked Language Model (MLM):

- **Masking ratio**: 15% of tokens are randomly selected.
- **Replacement strategy**: Of the selected tokens, 80% are replaced with `[MASK]`, 10% replaced with a random token from the vocabulary, 10% left unchanged.
- **Prediction objective**: Predict the original token at each masked position using bidirectional context from the transformer encoder.
- **Loss function**: Negative log-likelihood (cross-entropy over the vocabulary):
  ```
  L_MBM = -sum_i log P(MASK_i = token_i | X_masked; theta)
  ```

### 1.3 Same-origin BURST Prediction (SBP)

SBP is analogous to BERT's Next Sentence Prediction (NSP):

- **Pair construction**: Each BURST is equally divided into sub-BURST_A and sub-BURST_B. In 50% of cases, sub-BURST_B is the actual second half of the same BURST (positive). In 50%, sub-BURST_B is sampled from a different BURST (negative).
- **Classification**: Binary classification (same-origin vs. different-origin) using the `[CLS]` token representation.
- **Loss function**: Binary cross-entropy:
  ```
  L_SBP = -sum_j log P(y_j | B_j; theta)
  ```
  where y_j in {0, 1} indicates whether the two sub-BURSTs originate from the same BURST.

### 1.4 Architecture and Training

- **Architecture**: 12 transformer layers, 12 attention heads, hidden dimension H=768, max sequence length 512.
- **Pre-training**: Batch size 32, 500,000 steps, learning rate 2e-5, warmup ratio 0.1. Pre-training data: ~30 GB unlabeled traffic (15 GB public datasets + 15 GB CSTNET).
- **Fine-tuning**: 10 epochs, batch size 32, dropout 0.5. Flow-level learning rate 6e-5; packet-level learning rate 2e-5.

---

## 2. YaTC (AAAI 2023)

**Paper**: Zhao et al., "Yet Another Traffic Classifier: A Masked Autoencoder Based Traffic Transformer with Multi-Level Flow Representation," AAAI 2023.
**Source**: https://ojs.aaai.org/index.php/AAAI/article/view/25674
**Code**: https://github.com/NSSL-SJTU/YaTC

### 2.1 Multi-level Flow Representation (MFR)

YaTC treats traffic as images rather than language:

- **Traffic matrix**: Raw packet bytes from the first 5 packets of a flow are arranged into a **40 x 40 matrix** (1,600 bytes total). This 2D matrix captures byte-level, packet-level, and flow-level information simultaneously.
- **Byte allocation**: 80 bytes for header + 240 bytes for payload (data after IP/TCP headers) per packet, totaling 320 bytes per packet across 5 packets = 1,600 bytes.
- **Key insight**: Traffic bytes lack explicit high-level semantic units, making traffic representation extraction more suitable as a CV task than an NLP task.

### 2.2 Masked Autoencoder Pre-training

YaTC directly applies the MAE framework from computer vision:

- **Patch construction**: The 40x40 traffic matrix is divided into **4x4 patches**, yielding 100 patches (10x10 grid).
- **Masking ratio**: **75%** of patches are randomly masked. This high ratio is consistent with image MAE and supports the view that traffic bytes have high information redundancy (unlike NLP tokens where 15% masking is optimal).
- **Encoder**: Processes only the 25% visible patches (25 of 100 patches). This is a ViT-based encoder.
- **Decoder**: Lightweight decoder that takes the encoder output plus mask tokens and attempts to reconstruct the original pixel values of the masked patches.
- **Reconstruction loss**: Mean Squared Error (MSE) between reconstructed and original patch pixel values:
  ```
  L_rec = MSE(y_rec, y_real)
  ```

### 2.3 Traffic Transformer Architecture

YaTC incorporates hierarchical attention:

- **Packet-level attention**: Captures intra-packet byte dependencies within each packet's token representation.
- **Flow-level attention**: Captures inter-packet temporal dependencies across the flow sequence.
- **Parameter sharing**: Encoders share parameters between the pre-training and fine-tuning stages, reducing model size and accelerating convergence.

### 2.4 Training Details

- **Pre-training**: 300 epochs on large-scale unlabeled traffic data, MSE reconstruction loss.
- **Fine-tuning**: Classifier head added on top of encoder for downstream classification tasks. Consistent SOTA performance across 5 datasets (ISCXVPN2016, ISCXTor2016, USTC-TFC2016, CICIoT2022, Cross-Platform). F1/accuracy ~0.963.

---

## 3. Lens (arXiv 2024, under review)

**Paper**: Wang et al., "Lens: A Foundation Model for Network Traffic."
**Source**: https://arxiv.org/abs/2402.03646
**Architecture**: T5-v1.1-base (~250M parameters), encoder-decoder.

### 3.1 Tokenization

- **Hex conversion**: All session flow content is converted to hexadecimal format for data uniformity.
- **Tokenizer**: WordPiece with predefined vocabulary. Vocabulary size: 65,536. Special tokens: `</s>`, `<pad>`, `<unk>`, `<tsk>` (task marker), `<head>` (header separator), `<pkt>` (packet boundary).
- **Input embeddings**: Sum of four 768-dimensional embedding vectors:
  1. **Token embedding**: Unique identifier for each token.
  2. **Positional embedding**: Preserves contextual order.
  3. **Header embedding**: Differentiates header fields from payload.
  4. **Packet segment embedding**: Differentiates packets within a flow and preserves temporal sequence.

### 3.2 Masked Span Prediction (MSP)

Adapted from T5's span denoising objective:

- **Span selection**: Contiguous spans of 1-5 tokens (averaging 3) are selected with 15% probability.
- **Masking**: Each selected span is replaced with a unique sentinel token `<extra_id_i>` (i in [0, 99]).
- **Reconstruction**: The decoder generates the original tokens for each masked span, using the sentinel token as prefix.
- **Loss function**: Negative log-likelihood over all masked spans:
  ```
  L_MSP = -sum_k sum_i log P(MSPAN_i = SPAN_i | x_in, MSPAN_<i; theta)
  ```

### 3.3 Packet Order Prediction (POP)

- **Shuffling**: 15% of input flows have their internal packet order randomly shuffled.
- **Classification**: 3-way classification task using `<pkt>` suffix tokens. For each of the first 3 packets, predicts whether the packet occupies its original position.
- **Loss function**: Cross-entropy:
  ```
  L_POP = -sum_i z_i sum_j y_ij log p_ij
  ```
  where z_i indicates whether flow i undergoes POP (not HTP), y_ij is whether packet j is in its original position, p_ij is softmax probability.

### 3.4 Homologous Traffic Prediction (HTP)

- **Pair construction**: 30% of flows not used in POP are selected. Each flow is split into two sub-flows (sf_i^1, sf_i^2). Heterologous (negative) pairs are formed by combining sf_i^1 with sf_j^2 (j != i).
- **Binary classification**: Uses the `</s>` token's hidden state to classify homologous (0) vs. heterologous (1).
- **Loss function**: Binary cross-entropy:
  ```
  L_HTP = -sum_i log P(y_i | x_in; theta)
  ```

### 3.5 Combined Objective

```
L_total = L_MSP + alpha * L_POP + beta * L_HTP
```
Optimal hyperparameters: alpha = 0.2, beta = 0.2 (determined via grid search).

### 3.6 Training Details

- **Pre-training**: Batch size 128 with 4-step gradient accumulation, LR 2e-2, 56,000 steps, warmup 8,000 steps, dropout 0.1. Data: 645 GB across six datasets.
- **Fine-tuning**: Batch size 32, LR 3e-5, 10 epochs. Uses 50-95% less labeled data than ET-BERT.
- **Input**: First 3 packets per flow (flow-level tasks); first 5 packets (packet-level tasks).

---

## 4. TrafficFormer (IEEE S&P 2025)

**Paper**: Zhou et al., "TrafficFormer: An Efficient Pre-trained Model for Traffic Data," IEEE S&P 2025.
**Source**: https://www.researchgate.net/publication/392750930

### 4.1 Pre-training Tasks

TrafficFormer retains ET-BERT's MBM task and replaces SBP with a more sophisticated objective:

1. **Masked Burst Modeling (MBM)**: Same as ET-BERT -- 15% masking with the standard 80/10/10 replacement strategy.

2. **Same Origin-Direction-Flow (SODF)**: Extends ET-BERT's SBP by requiring the model to additionally predict:
   - Whether two sub-sequences originate from the same BURST (origin)
   - The direction of each sub-sequence (client-to-server vs. server-to-client)
   - The sequential ordering within the flow
   - Classification uses the `[CLS]` token representation from the last transformer layer.

### 4.2 Fine-tuning: Random Initialization Field Augmentation (RIFA)

RIFA generates augmented copies of traffic data by randomly altering "randomly initialized fields" (e.g., TCP sequence numbers) in the first packet while preserving the change pattern across subsequent packets. For instance, RIFA replaces the TCP sequence number of the first packet with a random number and assigns subsequent sequence numbers as (random number + original difference). This preserves protocol semantics while diversifying training data.

### 4.3 Architecture and Training

- **Base model**: BERT architecture, 12 layers, hidden 768.
- **Tokenization**: 4-hex Bigram Encoding via BPE (same as ET-BERT).
- **Input**: First 5 packets per flow, 64 bytes per packet after the Ethernet layer.
- **Pre-training**: Adam optimizer, LR 2e-5, batch size 64 (effective 192 with 3 GPUs), 500,000 steps (model selected at step 120,000 where loss stabilizes).
- **Results**: F1 improvements of 10.05-10.09% over ET-BERT.

---

## 5. netFound (arXiv 2023)

**Paper**: Guthula et al., "netFound: Foundation Model for Network Security."
**Source**: https://arxiv.org/abs/2310.17025
**Code**: https://github.com/SNL-UCSB/netFound
**Model**: https://huggingface.co/snlucsb/netFound-640M-base (~640M parameters)

### 5.1 Self-Supervised Pre-training

netFound uses **masked token prediction** with domain-specific tokenization:

- **Masking**: 30% of tokens are randomly selected. Of these, 80% are replaced with `[MASK]`, 10% left unchanged, 10% replaced randomly (BERT-style but at 30% rate).
- **Loss**: Negative log-likelihood:
  ```
  L = min (1/l) sum_i -X_i^p log(X_hat_i^p)
  ```

### 5.2 Protocol-Aware Tokenization

Unlike generic BPE, netFound parses packet headers according to protocol specifications:

- Parses TCP, UDP, IP, ICMP headers field by field.
- Expands each field to 2-byte tokens, preserving semantic boundaries.
- Splits 32-bit TCP sequence/acknowledgment numbers into two tokens each.
- Includes first 12 payload bytes (excludes SNI/domain names to avoid learning shortcuts).
- **Result**: Up to 18 tokens per packet.

### 5.3 Hierarchical Transformer Architecture

Two-layer hierarchy:

1. **Burst Encoder**: Processes tokens within individual bursts (up to 108 tokens per burst: 6 packets x 18 tokens).
2. **Flow Encoder**: Processes burst-level `[CLS-B]` tokens plus a flow-level `[CLS-F]` token via skip connections.

### 5.4 Input Embeddings

Three embedding types, combined additively:

1. **Token embedding**: Vocabulary size 65,539.
2. **Positional embedding**: Token positions within bursts (1-108) and burst positions within flows (1-13).
3. **Metadata embedding**: 5 attributes -- direction (inbound/outbound), packet count, byte count, inter-arrival time difference, protocol type.

---

## 6. NetGPT (arXiv 2023)

**Paper**: Meng et al., "NetGPT: Generative Pretrained Transformer for Network Traffic."
**Source**: https://arxiv.org/abs/2304.09513
**Code**: https://github.com/ict-net/NetGPT

### 6.1 Architecture and Pre-training

- **Base model**: GPT-2-base (~100M parameters), autoregressive.
- **Tokenization**: Each byte is converted to hex representation. Vocabulary constructed via WordPiece, expanding beyond the basic 256 hex values.
- **Pre-training objective**: Autoregressive next-token prediction:
  ```
  P(t_k | t_1, t_2, ..., t_{k-1})
  ```
- **Max sequence length**: 512 tokens.
- **Packet delimiter**: `[pck]` token separates packets within a flow.

### 6.2 Fine-tuning

- **Classification**: Task name used as a prompt, translated to hex, attached to input beginning.
- **Generation**: Text-to-text methods with hex prompts to generate targeted tokens.
- **Header field shuffling**: During fine-tuning, header field order is randomized to improve one-directional learning robustness.

---

## 7. Flow-MAE (RAID 2023)

**Paper**: Hang et al., "Flow-MAE: Leveraging Masked AutoEncoder for Accurate, Efficient and Robust Malicious Traffic Classification," RAID 2023.
**Source**: https://dl.acm.org/doi/10.1145/3607199.3607206
**Code**: https://github.com/NLear/Flow-MAE

### 7.1 Approach

- Uses burst representation as input (same concept as ET-BERT).
- Applies patch embedding to accommodate variable traffic lengths.
- Pre-training task: **Masked Patch Model** -- randomly masks patches from the burst representation and reconstructs them.
- **Advantage over ET-BERT**: Processes longer input sequences due to MAE's asymmetric encoder-decoder (only 25% of patches pass through the heavy encoder).
- **Results**: Accuracy > 0.99, throughput > 900 samples/s, 7.8-10.3x faster than ET-BERT, requiring only 0.2% FLOPs and 44% memory.

---

## 8. CL-ViME (IEEE 2024/2025)

**Paper**: "CL-ViME: Contrastive Learning and Vision Mixture of Experts for Encrypted Traffic Classification."
**Source**: https://ieeexplore.ieee.org/document/11320841

### 8.1 Architecture

Three main components:

1. **Packet-Temporal Matrix**: Preserves fine-grained packet headers and flow-level temporal structure in a 2D representation.
2. **Vertical Vision Transformer with MoE**: Extracts dual-view features through vertical patching and dynamic expert routing.
3. **Dual-Granularity Contrastive Learning**: Aligns packet-level and flow-level representations via an MoE projector, followed by lightweight classifier-head fine-tuning.

### 8.2 Contrastive Learning Objective

Dual-granularity contrastive loss that:
- Creates augmented views of the same traffic sample (positive pairs).
- Pushes apart representations from different samples (negative pairs).
- Operates at both packet-level and flow-level granularity simultaneously.

---

## 9. PERT (ITU 2020)

**Paper**: He et al., "PERT: Payload Encoding Representation from Transformer for Encrypted Traffic Classification," ITU Kaleidoscope 2020.

- **Pioneer work**: First to apply transformer pre-training to encrypted traffic classification.
- **Tokenization**: Byte-level bigrams in hexadecimal format (precursor to ET-BERT's approach).
- **Pre-training**: Masked Language Model (MLM) task at the packet level.
- **Fine-tuning**: Flow-level supervised learning via concatenation of individual packet representations.
- **Limitation**: Flow-level classification uses concatenated outputs rather than joint encoding, losing inter-packet dependencies. ET-BERT's flow-level approach outperforms by ~4.26%.

---

## 10. Other Contrastive and Semi-Supervised Approaches

### 10.1 CL-ETC (IEEE 2022)

Contrastive learning encoder for encrypted traffic:
- Creates multiple augmentation samples per input using a traffic-specific augmenter.
- Narrows representation vectors among similar augmentations, alienates dissimilar ones.
- Semi-supervised: Encoder trained on unlabeled data, classifier fine-tuned on small labeled set.

### 10.2 ET-SSL (Nature Scientific Reports 2025)

Self-supervised contrastive learning for encrypted traffic anomaly detection:
- Operates on flow-level statistical features: packet length, inter-arrival time, flow duration, protocol metadata.
- Contrastive objective: Normal traffic clustered closely, anomalies pushed apart.
- No payload inspection needed.
- Achieves 96.8% accuracy, 92.7% TPR, real-time detection (15-25ms latency, 10 Gbps throughput).

---

## 11. TTT/TTA Auxiliary Task Design Principles

### 11.1 Original TTT Framework (Sun et al., ICML 2020)

**Source**: https://arxiv.org/abs/1909.13231

- **Y-shaped architecture**: Shared feature extractor (theta_e) with two heads: main task head (theta_m) and self-supervised head (theta_s).
- **Joint training**: L_total = L_main(theta_e, theta_m) + lambda * L_self(theta_e, theta_s).
- **Test-time procedure**: Given a test sample x, update theta_e by minimizing L_self(x; theta_e, theta_s), then predict using the main task head with updated theta_e.
- **Original auxiliary task**: Rotation prediction (0, 90, 180, 270 degrees) for images.

**Key theoretical result (Theorem 1)**: For convex models, TTT reduces the main task loss when there exists positive gradient correlation between the main and self-supervised tasks:
```
inner_product(grad_{theta_e} L_main, grad_{theta_e} L_self) > 0
```
Empirically validated across 75 corruption types: stronger gradient correlations correspond to larger TTT performance gains.

### 11.2 TTT-MAE (Gandelsman et al., NeurIPS 2022)

**Source**: https://yossigandelsman.github.io/ttt_mae

- **Auxiliary task**: Masked autoencoding (reconstruct 75% masked patches).
- **Separate training**: Unlike original TTT, uses a pre-trained MAE encoder (frozen), trains only a ViT-Base main task head (86M parameters).
- **Test-time**: Updates the MAE encoder via reconstruction loss on the test sample using SGD for 20 steps (momentum 0.9, weight decay 0.2, LR 5e-3).
- **Key advantage**: MAE provides a richer self-supervised signal than rotation prediction; main task loss keeps decreasing even after 500 reconstruction steps.

### 11.3 Gradient Alignment (GraTa, AAAI 2025)

**Source**: https://arxiv.org/abs/2408.07343

- **Core insight**: Measures cosine similarity between auxiliary gradient and pseudo gradient. Higher cosine similarity = better alignment = better TTA.
- **Implicit alignment**: Two-step optimization: (1) theta' = theta - grad(L_ent), (2) minimize L_con(theta'). Via Taylor expansion, this approximately maximizes the inner product between gradients without computing expensive Hessians.
- **Result**: Certain auxiliary losses (rotation prediction, denoising) yield suboptimal results compared to entropy loss, confirming that task-specific auxiliary objectives matter significantly.

### 11.4 CTA: Cross-Task Alignment (Barbeau et al., 2025)

**Source**: https://arxiv.org/abs/2507.05221

- **Problem**: Multi-task TTT suffers from negative transfer during joint training where conflicting tasks produce gradient interference.
- **Solution**: Train self-supervised and supervised encoders independently, then align via cross-encoder contrastive loss. Test-time updates modify only the self-supervised encoder while keeping the classifier frozen.
- **Result**: 4.51% improvement over prior TTT SOTA on CIFAR-10-C.

### 11.5 Pitfalls of TTA (Zhao et al., ICML 2023)

**Source**: https://arxiv.org/abs/2306.03536 | https://github.com/LINs-lab/ttab

Three key pitfalls identified:
1. **Hyperparameter sensitivity**: Model selection is exceedingly difficult due to online batch dependency.
2. **Model quality dependence**: TTA effectiveness varies greatly with the quality of the pre-trained model.
3. **No universal method**: No existing method addresses all common types of distribution shifts.

### 11.6 When TTA Fails

Based on comprehensive survey (https://link.springer.com/article/10.1007/s11263-024-02181-w):

- **BN-based methods fail**: With small batches, non-i.i.d. streams, severe label shifts.
- **Entropy minimization fails**: When source model predictions are incorrect (error accumulation), can collapse to trivial predictions.
- **Auxiliary self-supervised methods fail**: When the pretext task is not gradient-aligned with the main task; when overfitting occurs to the auxiliary task.
- **General failure**: Correlated data streams (non-i.i.d. batches), temporal correlation in online settings.

---

## 12. Feasibility Analysis for TTA

For each SSL task, we evaluate four criteria:
1. **Label-free**: Can it be computed without ground-truth labels? (Required for TTA)
2. **Feature availability**: Does it use the same features available at inference time?
3. **Computational cost**: Is it lightweight enough for online adaptation?
4. **Gradient alignment**: Is the task likely to produce gradients aligned with traffic classification?

### 12.1 Masked Token/Burst Modeling (MBM) -- ET-BERT / TrafficFormer

| Criterion | Assessment |
|-----------|-----------|
| Label-free | YES. Predicts masked tokens from context; purely self-supervised. |
| Feature availability | YES. Operates on the same raw byte sequences used for classification. |
| Computational cost | MODERATE. Requires a forward-backward pass through the transformer encoder. With 512 tokens and 12 layers, this is comparable to a classification forward pass. The 15% masking means the prediction head operates over ~77 tokens. |
| Gradient alignment | HIGH LIKELIHOOD. MBM learns byte-level contextual dependencies, which are the same features the classifier uses to distinguish traffic types. Byte patterns (protocol fields, payload structure) are directly relevant to classification. |

**TTA verdict: STRONG CANDIDATE.** MBM is the most naturally suited task for TTA in traffic classification. It directly forces the encoder to model the byte distribution of the test-time traffic, which is exactly what shifts under distribution drift. The reconstruction signal provides dense gradients across all encoder layers. At test time, simply mask 15% of tokens in the input, compute the reconstruction loss, and take 1-20 gradient steps on the shared encoder before classification.

**Implementation sketch**:
```
# At test time, for each batch:
x_masked, mask_indices = random_mask(x, ratio=0.15)
logits_recon = model.mlm_head(model.encoder(x_masked))
loss_mlm = cross_entropy(logits_recon[mask_indices], x[mask_indices])
loss_mlm.backward()
optimizer.step()  # Update encoder only
# Then classify with updated encoder:
output = model.classifier(model.encoder(x))
```

### 12.2 Same-origin BURST Prediction (SBP) / SODF -- ET-BERT / TrafficFormer

| Criterion | Assessment |
|-----------|-----------|
| Label-free | YES. Pairs are constructed from the same input flow by splitting/swapping sub-BURSTs; no external labels needed. |
| Feature availability | PARTIAL. Requires access to BURSTs (directional packet sequences), which may not be available if the inference pipeline only provides pre-processed features. If raw packets are available, this is feasible. |
| Computational cost | LOW-MODERATE. Binary classification on `[CLS]` token; main cost is the shared encoder forward pass. |
| Gradient alignment | MODERATE. SBP captures flow-level coherence and temporal structure. This is relevant to classification (e.g., distinguishing application protocols by their interaction patterns), but the signal is coarser than byte-level MBM. SODF additionally captures directionality, which is more informative. |

**TTA verdict: MODERATE CANDIDATE.** SBP/SODF could serve as a complementary auxiliary loss alongside MBM, but is less ideal as the sole TTA objective because: (1) it requires constructing burst pairs at test time, adding complexity; (2) the binary classification signal is sparser than per-token reconstruction; (3) gradient alignment with fine-grained class discrimination is less direct. However, SODF's directional and ordering signals could help maintain structural understanding during adaptation.

### 12.3 Masked Patch Reconstruction (MAE) -- YaTC / Flow-MAE

| Criterion | Assessment |
|-----------|-----------|
| Label-free | YES. Reconstructs masked patches from the traffic matrix; purely self-supervised. |
| Feature availability | YES. Requires the same raw bytes used for classification, arranged as a 40x40 matrix. |
| Computational cost | LOW. The MAE asymmetric design means the encoder only processes 25% of patches (25 of 100). The lightweight decoder handles reconstruction. This is significantly cheaper than a full encoder pass. Flow-MAE reports only 0.2% FLOPs compared to ET-BERT. |
| Gradient alignment | HIGH LIKELIHOOD. Reconstructing traffic byte patterns forces the encoder to model the statistical properties of the current traffic distribution. The 75% masking ratio works because traffic data has high redundancy, but requires the model to learn meaningful structural features. |

**TTA verdict: STRONG CANDIDATE.** MAE-based reconstruction is particularly attractive for TTA because: (1) the asymmetric encoder-decoder makes it computationally efficient; (2) the dense pixel-level MSE loss provides strong gradients; (3) 75% masking creates a challenging pretext task that exercises the encoder's representation capacity. The 2D traffic matrix naturally captures both byte-level and packet-level patterns.

**Implementation sketch**:
```
# At test time, for each batch:
x_matrix = reshape_to_40x40(raw_bytes)
patches = patchify(x_matrix, patch_size=4)
visible, masked, mask = random_mask_patches(patches, ratio=0.75)
encoded = encoder(visible)
decoded = decoder(encoded, mask)
loss_mae = MSE(decoded, patches[masked])
loss_mae.backward()
optimizer.step()  # Update encoder only
# Classify:
full_encoded = encoder(patchify(x_matrix, patch_size=4))
output = classifier(full_encoded)
```

### 12.4 Masked Span Prediction (MSP) -- Lens

| Criterion | Assessment |
|-----------|-----------|
| Label-free | YES. Masks and reconstructs token spans; self-supervised. |
| Feature availability | YES. Operates on the same hex-tokenized traffic data. |
| Computational cost | MODERATE-HIGH. Requires both encoder and decoder passes in the T5 architecture. The decoder must autoregressively generate tokens for each masked span. At ~250M parameters, this is heavier than MBM. |
| Gradient alignment | HIGH. Span prediction captures multi-token dependencies and structural patterns in traffic. The sentinel-based mechanism forces understanding of byte sequence semantics. |

**TTA verdict: MODERATE CANDIDATE.** MSP provides rich self-supervised signal but the encoder-decoder architecture makes it computationally expensive for online TTA. If the decoder is kept small or shared parameters are limited to the encoder, this becomes more feasible. Best suited for batch TTA (adapt on accumulated test batches) rather than per-sample adaptation.

### 12.5 Packet Order Prediction (POP) -- Lens

| Criterion | Assessment |
|-----------|-----------|
| Label-free | YES. Shuffles packets and predicts original order; no labels needed. |
| Feature availability | YES. Requires multi-packet flow input, which is typically available. |
| Computational cost | LOW. 3-way classification on packet position tokens; lightweight head on top of encoder. |
| Gradient alignment | MODERATE-HIGH. Packet ordering captures temporal interaction patterns that are highly relevant to application identification. Different applications exhibit distinct request-response patterns. Under distribution shift, packet ordering characteristics may change (new application versions, different network conditions). |

**TTA verdict: MODERATE CANDIDATE.** POP is computationally cheap and captures complementary temporal information to byte-level tasks. However, it provides a sparse gradient signal (only 3 classification decisions per flow). Best used as a secondary objective combined with MSP or MBM, not as the sole TTA task. The Lens paper's alpha=0.2 weighting confirms its auxiliary nature.

### 12.6 Homologous Traffic Prediction (HTP) -- Lens

| Criterion | Assessment |
|-----------|-----------|
| Label-free | YES. Constructs pairs from flow splits; purely self-supervised. |
| Feature availability | PARTIAL. Requires multiple flows simultaneously to construct negative pairs. In online TTA, this means maintaining a buffer of recent flows. |
| Computational cost | MODERATE. Requires encoding two sub-flows and performing binary classification. Buffer management adds memory overhead. |
| Gradient alignment | MODERATE. Learns flow-level coherence, which relates to application behavior consistency. Less directly tied to byte-level class-discriminative features. |

**TTA verdict: WEAK CANDIDATE for per-sample TTA, MODERATE for batch TTA.** The requirement for negative pair construction from multiple flows makes it impractical for single-sample TTA. In batch TTA settings (adapting on a batch of test flows), HTP is feasible and provides a flow-level structural signal. However, the binary classification provides limited gradient information per pair.

### 12.7 Autoregressive Next-Token Prediction -- NetGPT

| Criterion | Assessment |
|-----------|-----------|
| Label-free | YES. Predicts each token from preceding context; purely self-supervised. |
| Feature availability | YES. Same byte sequence as classification input. |
| Computational cost | MODERATE. Requires sequential autoregressive generation. Cannot parallelize prediction across positions like MLM. |
| Gradient alignment | MODERATE. Learns forward-directional byte dependencies. However, the unidirectional nature means it only captures causal relationships, missing bidirectional context that may be important for classification. |

**TTA verdict: WEAK-MODERATE CANDIDATE.** Autoregressive prediction is less ideal than bidirectional masked modeling for classification-oriented TTA because: (1) it captures only forward dependencies; (2) sequential generation is slower than parallel masked prediction; (3) the GPT-2 architecture is not designed for classification tasks. However, it could be useful if the traffic model is already GPT-based.

### 12.8 Contrastive Learning -- CL-ViME / CL-ETC / ET-SSL

| Criterion | Assessment |
|-----------|-----------|
| Label-free | YES. Creates positive pairs via augmentation; no labels needed. |
| Feature availability | DEPENDS ON AUGMENTATION. Traffic-specific augmentations (e.g., packet dropping, byte masking, header field perturbation) operate on available features. However, traditional image augmentations (rotation, crop) are not meaningful for traffic data. |
| Computational cost | HIGH. Requires encoding two augmented views per sample, plus a contrastive loss over the batch. Memory-intensive due to the need for a large batch (or memory bank) of negatives. |
| Gradient alignment | VARIABLE. Depends heavily on the augmentation strategy. Good augmentations (that preserve class-relevant features while diversifying class-irrelevant ones) produce well-aligned gradients. Bad augmentations can destroy class-discriminative features. |

**TTA verdict: WEAK CANDIDATE for online TTA.** Contrastive learning's reliance on large batches of negatives and careful augmentation design makes it impractical for online or per-sample TTA. However, in batch TTA settings with appropriate traffic augmentations, it could be effective. The CTA paper's student-teacher approach offers a promising alternative: use contrastive alignment to bridge self-supervised and supervised objectives without joint training.

### 12.9 Protocol-Aware Masked Prediction -- netFound

| Criterion | Assessment |
|-----------|-----------|
| Label-free | YES. Same masking mechanism as BERT; no labels needed. |
| Feature availability | YES. Uses protocol-parsed packet headers and first 12 payload bytes. |
| Computational cost | MODERATE-HIGH. 640M parameter model with hierarchical two-level transformer. Larger than ET-BERT or YaTC. |
| Gradient alignment | HIGH. Protocol-aware tokenization means the model explicitly learns field-level semantics. Masking protocol fields forces the model to understand protocol structure, which is highly relevant to traffic classification. |

**TTA verdict: STRONG CANDIDATE (if model size is acceptable).** netFound's protocol-aware masking is the most semantically meaningful masking strategy for network traffic. By masking at the protocol field level rather than arbitrary byte boundaries, the reconstruction task directly requires understanding of packet structure. The 30% masking rate (vs. ET-BERT's 15%) provides denser gradients. However, the 640M parameter model is expensive for online TTA. A smaller variant or selective layer updating would be needed.

---

## 13. Comparative Summary Table

| SSL Task | Model | Label-Free | Compute Cost | Gradient Alignment | TTA Feasibility | Notes |
|----------|-------|-----------|-------------|-------------------|----------------|-------|
| Masked Token Modeling (MBM) | ET-BERT, TrafficFormer | Yes | Moderate | High | **Strong** | Dense per-token gradients; directly models byte distribution shift |
| Same-origin Burst Pred. (SBP/SODF) | ET-BERT, TrafficFormer | Yes | Low-Moderate | Moderate | Moderate | Sparse binary signal; complementary to MBM |
| MAE Patch Reconstruction | YaTC, Flow-MAE | Yes | **Low** | High | **Strong** | Efficient asymmetric encoder-decoder; MSE loss; handles variable-length traffic |
| Masked Span Prediction (MSP) | Lens | Yes | Moderate-High | High | Moderate | Rich signal but expensive decoder; best for batch TTA |
| Packet Order Prediction (POP) | Lens | Yes | Low | Moderate-High | Moderate | Lightweight; sparse signal; good secondary objective |
| Homologous Traffic Pred. (HTP) | Lens | Yes | Moderate | Moderate | Weak-Moderate | Needs flow buffer for negatives; batch TTA only |
| Next-Token Prediction | NetGPT | Yes | Moderate | Moderate | Weak-Moderate | Unidirectional; sequential; not optimized for classification |
| Contrastive Learning | CL-ViME, CL-ETC | Yes | High | Variable | Weak | Needs large batches; augmentation-sensitive |
| Protocol-Aware Masking | netFound | Yes | High | **Very High** | Strong (with caveats) | Best semantic signal; expensive model |

---

## 14. Recommendations for TTA in Traffic Classification

### 14.1 Primary Recommendation: Masked Byte/Token Modeling (MBM)

MBM is the strongest candidate for a TTA auxiliary objective for the following reasons:

1. **Universality**: Works with any BERT-based traffic model (ET-BERT, TrafficFormer, PERT).
2. **Dense gradients**: Per-token reconstruction loss provides gradients across all encoder layers and positions.
3. **Direct distribution modeling**: When traffic distribution shifts (new application versions, changed encryption parameters, network condition changes), the byte distribution shifts correspondingly. MBM directly forces the encoder to re-model this distribution.
4. **Minimal overhead**: No additional data structures needed (no pair construction, no flow buffers). Just mask the input and reconstruct.
5. **Well-studied**: MBM is the traffic analog of BERT's MLM, which has been extensively validated for TTT (via TTT-MAE and related works).

**Recommended configuration for TTA**:
- Masking ratio: 15% (same as pre-training to avoid distribution mismatch).
- Gradient steps: 1-5 per test batch (more steps risk overfitting to individual samples).
- Updated parameters: Encoder only (freeze classification head).
- Optimizer: SGD with momentum (more stable than Adam for few-step updates, per TTT-MAE findings).

### 14.2 Secondary Recommendation: MAE Patch Reconstruction (for ViT-based models)

If the base model uses a ViT/MAE architecture (YaTC, Flow-MAE), patch reconstruction is the natural TTA task:

1. **Computational efficiency**: Asymmetric encoder processes only 25% of patches during reconstruction.
2. **Strong reconstruction signal**: MSE loss on pixel values provides continuous gradients (vs. discrete cross-entropy in MBM).
3. **High masking ratio tolerance**: 75% masking works because traffic has high byte redundancy.

**Recommended configuration for TTA**:
- Masking ratio: 75% (matching pre-training).
- Gradient steps: 10-20 per test batch (TTT-MAE showed MAE benefits from more steps than MLM).
- Updated parameters: Encoder only.
- Optimizer: SGD with momentum 0.9, weight decay 0.2, LR 5e-3.

### 14.3 Combined Objective (Multi-task TTA)

For maximum robustness, combine byte-level and flow-level objectives:

```
L_TTA = L_MBM + alpha * L_POP
```

where alpha = 0.1-0.2. This captures both byte-level distribution shift (via MBM) and temporal pattern shift (via POP). Important: follow CTA's insight and avoid conflicting gradient updates. Monitor gradient cosine similarity; if alignment drops below 0, disable the secondary task.

### 14.4 Safety Mechanisms

Based on TTA pitfall research (Zhao et al., ICML 2023):

1. **Entropy filtering**: Only adapt on samples where the classifier's prediction entropy exceeds a threshold (following EATA). High-confidence predictions indicate no adaptation is needed.
2. **Gradient magnitude clipping**: Prevent catastrophically large updates from anomalous test samples.
3. **EMA model**: Maintain an exponential moving average of the adapted model weights as a safety mechanism against error accumulation.
4. **Periodic reset**: In long-running online TTA, periodically reset to the original pre-trained weights to prevent gradual drift from the original task.

---

## 15. Open Research Questions

1. **Gradient alignment empirics for traffic**: No existing work has measured the gradient correlation between masked byte modeling and traffic classification objectives. This is a critical validation step before deploying MBM as a TTA auxiliary.

2. **Masking ratio for TTA vs. pre-training**: It is unknown whether the optimal masking ratio at test time should match the pre-training ratio or differ. Higher ratios provide more signal but may create too-difficult reconstruction problems; lower ratios provide less signal but more stable gradients.

3. **Batch size sensitivity**: Network traffic TTA will likely encounter small, potentially non-i.i.d. batches (e.g., flows from a single user or application cluster). BN-based methods will fail here; the auxiliary SSL approach may be more robust.

4. **Protocol-aware masking for TTA**: netFound's protocol-aware masking could produce semantically richer TTA gradients than random byte masking, but at higher computational cost. A hybrid approach (mask only certain protocol fields at test time) could balance cost and semantic relevance.

5. **Traffic-specific augmentations for contrastive TTA**: Developing augmentations that preserve application identity while diversifying network conditions (jitter, reordering, field randomization) could make contrastive learning feasible for traffic TTA.

---

## References

1. Lin, X., et al. "ET-BERT: A Contextualized Datagram Representation with Pre-training Transformers for Encrypted Traffic Classification." WWW 2022. https://dl.acm.org/doi/10.1145/3485447.3512217
2. Zhao, R., et al. "Yet Another Traffic Classifier: A Masked Autoencoder Based Traffic Transformer with Multi-Level Flow Representation." AAAI 2023. https://ojs.aaai.org/index.php/AAAI/article/view/25674
3. Wang, Q., et al. "Lens: A Foundation Model for Network Traffic." arXiv:2402.03646, 2024. https://arxiv.org/abs/2402.03646
4. Zhou, G., et al. "TrafficFormer: An Efficient Pre-trained Model for Traffic Data." IEEE S&P 2025.
5. Guthula, S., et al. "netFound: Foundation Model for Network Security." arXiv:2310.17025, 2023. https://arxiv.org/abs/2310.17025
6. Meng, Z., et al. "NetGPT: Generative Pretrained Transformer for Network Traffic." arXiv:2304.09513, 2023. https://arxiv.org/abs/2304.09513
7. Hang, Z., et al. "Flow-MAE: Leveraging Masked AutoEncoder for Accurate, Efficient and Robust Malicious Traffic Classification." RAID 2023. https://dl.acm.org/doi/10.1145/3607199.3607206
8. Sun, Y., et al. "Test-Time Training with Self-Supervision for Generalization under Distribution Shifts." ICML 2020. https://arxiv.org/abs/1909.13231
9. Gandelsman, Y., et al. "Test-Time Training with Masked Autoencoders." NeurIPS 2022. https://yossigandelsman.github.io/ttt_mae
10. Zhao, H., et al. "On Pitfalls of Test-Time Adaptation." ICML 2023. https://arxiv.org/abs/2306.03536
11. Barbeau, M., et al. "CTA: Cross-Task Alignment for Better Test Time Training." arXiv:2507.05221, 2025.
12. GraTa: "Gradient Alignment Improves Test-Time Adaptation for Medical Image Segmentation." AAAI 2025. https://arxiv.org/abs/2408.07343
13. He, H.Y., et al. "PERT: Payload Encoding Representation from Transformer for Encrypted Traffic Classification." ITU Kaleidoscope 2020.
