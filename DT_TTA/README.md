# DT-TTA: Drift-Topology-Conditioned Test-Time Adaptation

## 方向定位

加密流量分类的 test-time 适应方法选择应当**取决于漂移在特征空间中的拓扑结构**：
- **Focal drift**（漂移集中在少数 channel）→ 只更新这些 channel 的 norm 参数 + 轻量 head 调整
- **Diffuse drift**（漂移弥散于多 channel）→ 更新所有 norm 参数 + 完整 head 微调

不需要重训源模型。完全复用主项目的 `Experiment/core_code/` 中的 baseline 与评估管线。

## 48 小时实验计划

### Step 1（4h）：漂移拓扑跨周期稳定性
对每个 test period 计算 GroupNorm channel-level drift score，用 Gini 系数判定 focal/diffuse/mixed。要求同协议在所有 period 上分类一致。

### Step 2（8h）：6 setting sweep
| ID | 方法 | 描述 |
|----|------|------|
| (0) | static | 不适应（基线）|
| (1) | ft_head | 全分类头微调 |
| (2) | supervised_norm | 全 GroupNorm γ/β |
| (3) | selective_norm | 仅漂移 channel 的 γ/β |
| (4) | focal_strategy | selective_norm + bias-only head |
| (5) | diffuse_strategy | full norm + full head |

### Step 3（4h）：分析与决策

**判据**：
- QUIC22 上 (4) > (5)：focal arm 成立
- TLS22 上 (5) > (4)：diffuse arm 成立
- 两者皆成立 → DT-TTA 框架验证通过 → 进入完整论文阶段

## 目录结构

```
DT_TTA/
├── README.md
├── methods/
│   ├── __init__.py
│   ├── topology.py            # GroupNorm stats、drift score、Gini 分类、channel mask
│   └── strategies.py          # SelectiveNormAdapt / FocalStrategy / DiffuseStrategy
├── scripts/
│   ├── compute_source_stats.py    # 预计算源域 GroupNorm 输入统计
│   ├── run_step1_topology.py      # Step 1: per-period topology 分类
│   ├── run_step2_sweep.sh         # Step 2: 36 次 sequential 评估
│   └── aggregate_step2.py         # Step 2 聚合 + 框架验证
├── outputs/
│   ├── source_stats/          # 源域统计（pt 文件）
│   ├── step1_topology/        # Step 1 输出
│   └── step2_sweep/           # Step 2 输出
└── docs/
```

## 与主项目的关系

| 资产 | 复用方式 |
|------|---------|
| `tta_tc.models.TTATCModel` | 直接加载现有 `outputs/{quic,tls}22_cnn/best_model.pt` |
| `tta_tc.data.cesnet_loader` | 数据加载完全复用 |
| `tta_tc.baselines.labeled_baselines._PeriodLabeledBase` | DT-TTA 三个方法继承自它 |
| `evaluate_tta.py` | 已扩展支持 `selective_norm` / `focal_strategy` / `diffuse_strategy` 三个方法 key |

## 一键运行

```bash
cd /data/xjw/traffic-drift2

# (a) 预计算两个数据集的源域统计 (~5 min × 2)
python DT_TTA/scripts/compute_source_stats.py \
    --config Experiment/core_code/configs/eval_quic22.yaml \
    --checkpoint Experiment/core_code/outputs/quic22_cnn/best_model.pt
python DT_TTA/scripts/compute_source_stats.py \
    --config Experiment/core_code/configs/eval_tls22.yaml \
    --checkpoint Experiment/core_code/outputs/tls22_cnn/best_model.pt

# (b) Step 1: 跨周期拓扑稳定性 (~5 min × 2)
python DT_TTA/scripts/run_step1_topology.py \
    --config Experiment/core_code/configs/eval_quic22.yaml \
    --checkpoint Experiment/core_code/outputs/quic22_cnn/best_model.pt \
    --source-stats DT_TTA/outputs/source_stats/quic22_source_stats.pt
python DT_TTA/scripts/run_step1_topology.py \
    --config Experiment/core_code/configs/eval_tls22.yaml \
    --checkpoint Experiment/core_code/outputs/tls22_cnn/best_model.pt \
    --source-stats DT_TTA/outputs/source_stats/tls22_source_stats.pt

# (c) Step 2: 6-setting sweep (放 tmux, ~2-3h)
tmux new -s dt_tta_sweep
bash DT_TTA/scripts/run_step2_sweep.sh
# Ctrl+B D 离开

# (d) 聚合
python DT_TTA/scripts/aggregate_step2.py
```

## 关键实验结果速查

跑完后看 aggregate_step2.py 输出，重点：
1. **Step 1 决定**：QUIC22 是否 consistently focal、TLS22 是否 consistently diffuse
2. **Step 2 决定**：focal_strategy vs diffuse_strategy 在两个数据集上的反转

如果两步都通过，DT-TTA framework 成立，可以开始写论文。
