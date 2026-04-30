# CA-TTA: Certification-Aware Test-Time Adaptation

## 方向定位

把 randomized smoothing certified robustness (CertTA, USENIX'25) 和 test-time adaptation 在加密流量场景下结合。**核心 hypothesis**：标准 TTA（如 Tent）会损害 certified robustness，需要 cert-aware 的 adaptation 目标。

## Phase 0：可行性验证 (1-2 周)

在投入 4-5 月正式做之前，先验证两个前提：

### Step 0.3：Vanilla cert-acc 是否随时间崩溃？
在 source 模型上跑 randomized smoothing，逐月测 certified accuracy。

### Step 0.4：Tent 适应后 cert-acc 是否变差？
跑标准 Tent 适应，再测 certified accuracy，对比 vanilla。

**判据矩阵**：

| Step 0.3 | Step 0.4 | Verdict |
|---------|---------|---------|
| cert acc 降 >20% | Tent 进一步损害 | ✅ **强 motivation**，进入 Phase 1 |
| cert acc 降 >20% | Tent 没影响 | ⚠️ Tent 不是右 baseline，考虑其他 TTA |
| cert acc 降 <5% | 任何 | ❌ **方向死**，certified 在漂移下不是问题 |

## 目录结构

```
CA_TTA/
├── README.md
├── methods/
│   ├── __init__.py
│   └── smoothing.py            # SmoothedClassifier + Cohen CERTIFY
├── scripts/
│   ├── phase0_cert_acc_per_month.py    # Step 0.2/0.3
│   ├── phase0_tent_then_certify.py     # Step 0.4
│   └── phase0_aggregate.py             # Verdict
├── outputs/                    # JSON results
└── docs/
```

## 快速运行

```bash
cd /data/xjw/traffic-drift2

# (a) Vanilla cert-acc per month, TLS22
python CA_TTA/scripts/phase0_cert_acc_per_month.py \
    --config Experiment/core_code/configs/eval_tls22.yaml \
    --checkpoint Experiment/core_code/outputs/tls22_cnn/best_model.pt \
    --sigma 0.25 \
    --max-samples-per-period 500

# (b) Tent then certify, TLS22
python CA_TTA/scripts/phase0_tent_then_certify.py \
    --config Experiment/core_code/configs/eval_tls22.yaml \
    --checkpoint Experiment/core_code/outputs/tls22_cnn/best_model.pt \
    --sigma 0.25 \
    --max-samples-per-period 500

# (c) Aggregate + verdict
python CA_TTA/scripts/phase0_aggregate.py
```

## 计算开销

每个 period 500 个样本 × (n0 + n) = 500 × 550 = 275K forward pass
- TLS22 9 个 period：~2.5M forward pass
- 1D-CNN 在 GPU 上每秒 ~5K forward → **~10 分钟/dataset/sigma**
- Tent 适应再加 ~5 分钟
- 总计 Phase 0 约 30 分钟

## 关键参数

| 参数 | 默认 | 说明 |
|------|------|------|
| sigma | 0.25 | smoothing noise std；可试 0.1, 0.25, 0.5 |
| n0 | 50 | 选 top class 的样本数 |
| n | 500 | 估计 p_A 的样本数 |
| alpha | 0.001 | Clopper-Pearson 置信参数 |
| radii | [0.05, 0.1, 0.25, 0.5] | 评估半径 |

## 论文方向 (Phase 1+，仅当 Phase 0 通过)

> **"Certification-Aware Test-Time Adaptation for Encrypted Traffic Classification"**
>
> Standard TTA (Tent/CoTTA/...) optimizes accuracy or entropy, which can collapse certified margins. We propose CA-TTA: a TTA loss that jointly optimizes accuracy and certified robustness via a differentiable surrogate (MACER-style margin loss).

不依赖之前 DT-TTA / TIF-extension 的失败结果——**这是独立的新方向**。
