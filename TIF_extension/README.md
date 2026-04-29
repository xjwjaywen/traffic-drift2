# TIF Extension: Drift-Structure-Aware Temporal Invariance Learning

## 方向定位

把 TIF (Learning Temporal Invariance in Android Malware Detectors, arXiv 2502.05098) 的训练时漂移不变性思路扩展到加密流量分类，并改进它的核心机制：

- **TIF**: 二分类 (Android malware) + uniform invariance
- **本工作**: 100+ 类细粒度分类 (encrypted traffic) + drift-structure-aware invariance

## 核心方法草案: Group-wise Invariant Risk Minimization (GIRM)

不对所有特征维度施加同等强度的不变性约束，而是根据漂移诊断结果，对漂移严重的特征通道/位置施加强约束，对稳定的维度保留判别能力。

## 阶段 0: 可行性验证 (1 周)

在投入完整方法实现前，先验证四件事：

| Step | 内容 | 时间 |
|------|------|------|
| 0.1 | 确认 CESNET 多 period 数据可下载 | 0.5 天 |
| 0.2 | **漂移结构时间稳定性诊断** | 1 天 |
| 0.3 | vanilla IGA 在 100+ 类上能否收敛 | 3 天 |
| 0.4 | vanilla IGA vs ERM 的 AURC 对比 | 2 天 |

**关键 gate: Step 0.2** — 如果训练期漂移规律和测试期漂移规律相关性低 (Spearman ρ < 0.3)，整个方向死掉。

## 目录结构

```
TIF_extension/
├── scripts/         # 阶段 0 各 step 的实验脚本
├── configs/         # 训练 / 评估配置
├── outputs/         # 实验结果 (logs, json, plots)
├── docs/            # 设计文档 / 论文笔记
└── README.md        # 本文件
```

## 与主项目的关系

复用主项目 (`Experiment/core_code/`) 的:
- 数据加载: `tta_tc.data.cesnet_loader`
- 模型: `tta_tc.models.TTATCModel`  (1D-CNN with GroupNorm)
- 漂移诊断: `scripts/diagnose_drift.py`
- 已训源模型: `outputs/quic22_cnn/`, `outputs/tls22_cnn/`

不重训现有源模型，新工作产出的所有 IGA / GIRM 模型放到 `TIF_extension/outputs/`。

## 参考文献

- TIF: https://arxiv.org/abs/2502.05098
- IRM: Arjovsky et al. (2019), Invariant Risk Minimization
- SoK Encrypted Traffic: https://arxiv.org/abs/2503.20093
