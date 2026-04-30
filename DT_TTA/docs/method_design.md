# DT-TTA Method Design

## 核心论点

> **现有 TTA 方法对所有特征/参数采用相同的适应策略，忽略漂移在特征空间中的结构。我们提出 DT-TTA：基于 drift topology 自动选择适应策略族。**

## Topology 诊断（无标签）

### 度量
对每层 GroupNorm，hook 其输入激活，统计每个 channel 的 (mean, var)。漂移分数定义为源/目标 Gaussian 的对称 KL：

$$
s_c = \tfrac{1}{2}\left[ KL(\mathcal{N}_s \,\|\, \mathcal{N}_t) + KL(\mathcal{N}_t \,\|\, \mathcal{N}_s) \right]
$$

### 拓扑分类
对每层每个 channel 的 score 计算 Gini 系数，按层 channel 数加权得到全局 Gini：

| Gini 范围 | Topology |
|----------|----------|
| ≥ 0.6 | focal（少数 channel 集中漂移）|
| ≤ 0.3 | diffuse（漂移分散）|
| 0.3 ~ 0.6 | mixed |

### Channel mask 选择
对每层取分数最高的 top-K（默认 K = 40% × C）作为"漂移 channel"，构造布尔 mask。

## 策略族

### SelectiveNormAdapt（孤立组件，验证 channel mask 自身价值）
- 仅更新漂移 channel 的 GroupNorm γ/β
- 通过对每个 GN 层注册 backward hook 实现 gradient masking
- 不更新分类头

### FocalStrategy
- SelectiveNormAdapt + 分类头 **bias-only** 更新（仅 b，不动 W）
- 直觉：focal drift 主要是 feature distribution 局部偏移，head 微调只需要小幅修正决策阈值

### DiffuseStrategy
- 全部 GroupNorm γ/β + 分类头全部参数（W 和 b）
- 直觉：diffuse drift 是全局变化，需要 full norm + full head 联合调整

### 共用训练协议
- Optimizer：Adam, lr=1e-3, wd=1e-4
- 30 epoch, batch size 64
- 标签预算：500 / period（与 v9 active TTA 一致）
- 采样器：random（控制变量）

## 复用的 Two-Pass 框架
所有三个方法继承 `_PeriodLabeledBase`：
1. Pass 1：source 模型 forward，收集 features / labels / static_logits / 原始 PPI
2. Diagnose：用收集的 PPI 计算 target stats，得到 channel mask（无需标签）
3. Sample：用 sampler 选 500 indices
4. Adapt：在 sampled (PPI, label) 上做有监督训练，参数选择按策略
5. Pass 2：用适应后模型 re-forward 全部数据，输出预测

## 关键实验对照

### Setting 间的隔离逻辑

```
(0) static                ← 无适应 baseline
(1) ft_head               ← head 适应
(2) supervised_norm       ← norm 适应
(3) selective_norm        ← norm 适应 + drift-aware
(4) focal_strategy        ← (3) + bias-only head
(5) diffuse_strategy      ← (1) + (2)
```

### 消除替代解释

- (3) > (2)：drift-aware channel selection 优于 uniform → 否则 selection 没价值
- (4) > (3)：bias-only head 补强 focal 适应 → 否则 head 调整无用
- (5) > (1)：full norm 补强 head → 否则 norm 调整无用
- QUIC: (4) > (5) AND TLS: (5) > (4)：topology-conditioned 选择有价值 → 否则 framework 失败

## 论文叙事骨架

### Title
*Drift-Topology-Conditioned Test-Time Adaptation for Encrypted Traffic Classification*

### Section 结构
1. Introduction：现有 TTA 忽略漂移结构差异
2. Background：TTA in encrypted traffic, drift detection
3. **Drift Topology Characterization**（实证发现）
   - Focal vs diffuse 二分法
   - GroupNorm-based diagnosis
4. **DT-TTA Framework**（方法）
   - Topology classifier
   - Strategy selector
   - Per-strategy parameter selection
5. Experiments
   - QUIC22 (focal) + TLS22 (diffuse)
   - 与 7 个 baseline 对比
   - Ablation：去掉 topology selection → 退化为 supervised_norm
6. Discussion & Limitations

## 风险点 & Mitigation

| 风险 | Mitigation |
|------|-----------|
| QUIC topology 跨周期不稳 | Step 1 先验证；不稳则改为 per-period 动态分类 |
| (3) selective_norm ≈ (2) supervised_norm | 添加 random_mask 对照（已支持，cfg.dt_random_mask=True）|
| (4) focal_strategy ≈ (1) ft_head（说明 norm 没用）| 此时 framework 失败，回退 |
| N=2 协议 generalizability 质疑 | 在论文 limitation 老实承认，propose synthetic drift on additional datasets as future work |

## 决策树

```
Step 1 通过？
├── No: 改 per-period 动态分类
└── Yes: → Step 2
        Step 2 (4) > (5) on QUIC AND (5) > (4) on TLS？
        ├── Both: ✓ Framework 成立 → 写论文
        ├── One: 部分成立 → 重新设计另一 arm
        └── None: framework 失败 → 退回 trick paper
```
