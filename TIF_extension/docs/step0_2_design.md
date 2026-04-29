# Step 0.2 — Drift-Structure Temporal Stability

## 目的

验证 GIRM 方向的核心假设：**训练时观察到的漂移规律能预测测试时的漂移规律**。

如果该假设不成立，"用漂移诊断指导 feature grouping" 的整个机制就失去因果基础。

## 假设的形式化

记 $D(p_a, p_b) \in \mathbb{R}^{3 \times 30}$ 为周期 $p_a$ 与 $p_b$ 之间的漂移矩阵（每个 (channel, position) 上的 KS 统计量）。

**假设**：$D(p_{\text{train\_end}}, p_{\text{train\_end}+k})$ 与 $D(p_{\text{test\_start}}, p_{\text{test\_start}+k})$ 在结构上相似（高 Spearman 相关）。

## 实验设计

### QUIC22
- **Reference**：W-2022-44（源域）
- **训练期信号**：W-2022-45 vs W-44 的漂移
- **测试期信号**：W-2022-46, W-2022-47 vs W-44 的漂移平均

### TLS22
- **Reference**：M-2022-3（源域）
- **训练期信号**：M-4, M-5 vs M-3 的漂移平均
- **测试期信号**：M-10, M-11, M-12 vs M-3 的漂移平均

### 度量
1. **Spearman ρ**：训练期 vs 测试期的 90-dim 漂移向量的秩相关
2. **Cosine similarity**：同上向量的方向相似度
3. **Top-K Jaccard overlap**：top-K 漂移最大的 (channel, position) 单元的重合度
4. **Per-channel breakdown**：分通道的 ρ，看哪个通道最稳定

## 决策矩阵

| ρ (KS sig) | 含义 | 行动 |
|------------|------|------|
| **> 0.7** | 漂移结构在时间上稳定 | GIRM 假设成立，进入 Step 0.3 |
| **0.3 ~ 0.7** | 部分稳定 | GIRM 可行，但需要 robust 设计 |
| **< 0.3** | 不稳定 | GIRM 假设崩溃，方向死掉 |

## 运行

```bash
# 在服务器 Experiment/core_code/ 下（cesnet 数据已下载好）
cd /data/xjw/traffic-drift2
python TIF_extension/scripts/drift_stability.py \
    --datasets quic22 tls22 \
    --data-dir-quic ./Experiment/core_code/data/quic22 \
    --data-dir-tls ./Experiment/core_code/data/tls22 \
    --max-batches 200 \
    --output-dir ./TIF_extension/outputs/step0_2

# 只跑一个数据集（如 QUIC 数据没问题先验证）
python TIF_extension/scripts/drift_stability.py \
    --datasets quic22 \
    --data-dir-quic ./Experiment/core_code/data/quic22

# 全量样本（去掉 batch cap，慢但精度更高）
python TIF_extension/scripts/drift_stability.py --max-batches 0
```

## 预期输出

```
============================================================================
Dataset: quic22
All periods loaded: ['W-2022-44', 'W-2022-45', 'W-2022-46', 'W-2022-47']
Reference period:   W-2022-44
Training-era set:   ['W-2022-45']
Test-era set:       ['W-2022-46', 'W-2022-47']
============================================================================
[load] W-2022-44
  samples=51200
[load] W-2022-45
  samples=51200
[load] W-2022-46
  samples=51200
[load] W-2022-47
  samples=51200

[result] Comparing drift signatures
  --- KS-statistic signature (magnitude of drift) ---
   Spearman ρ = +0.XXXX  (p=...)
   Cosine sim = +0.XXXX
   Top-10 overlap (Jaccard) = X.XX
   Top-20 overlap (Jaccard) = X.XX

  --- Per-channel breakdown (KS Spearman ρ) ---
   channel 0 (size      ): ρ = +0.XXXX
   channel 1 (direction ): ρ = +0.XXXX
   channel 2 (ipt       ): ρ = +0.XXXX

  ==> DECISION (quic22): PASS / PARTIAL / FAIL
```

## 解读细则

- **看 ρ_total 决定整体走向**
- **看 per-channel ρ 找最稳定的通道**——即使整体不稳，某些通道（比如 size）可能稳定，可作为 GIRM 的 limited-scope 应用
- **看 Top-K overlap 检验"少数关键位置"假设**——如果 ρ 中等但 top-10 overlap 高，说明少数关键位置稳定，多数不重要的位置噪声大，仍然可以做 GIRM

## 计算成本

- CPU only
- 每个 period 约 50K-100K 样本
- 每个 dataset 总共 ~5 min
- 两个数据集 ~10 min

## 已知风险

1. **TLS22 的 reference period 是 M-3（训练期）但训练集本身只用了 M-3**，这是 OK 的——我们这里测的是相对 M-3 的漂移演化，不是模型性能
2. **W-44/M-3 内部样本量可能因 cesnet-datazoo 配置而不同**，使用 `max-batches` 上限可以保证比较公平
