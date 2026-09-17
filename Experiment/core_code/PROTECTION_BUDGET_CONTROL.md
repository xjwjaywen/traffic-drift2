# 保护样本配额对照

## 要回答的问题

上一轮 200 个理想保护样本减少了误伤，但也损失了一部分崩溃类恢复。本轮仅检查减少保护配额后，恢复与损伤的取舍是否改善。修复器和训练设置保持原样。

每个种子比较两种保护来源：`oracle_probe_damage`（源模型正确、保存的预演模型错误）与 `oracle_source_correct`（所有源模型正确样本）。默认种子 **0、1、2**。

| 指定保护数 | BADGE 补足数 | 每个保护来源的处理 |
|---:|---:|---|
| 0 | 1000 | 复用同一个原 BADGE 结果 |
| 50 | 950 | 新训练 |
| 100 | 900 | 新训练 |
| 200 | 800 | 复用上一轮对应 oracle 结果 |

新增 **2 种来源 × 2 个配额 × 3 个种子 = 12 次训练**。不重训预演、纯 BADGE 或 200 个保护样本端点。比例按原协议总查询量计算，真实实验总量为 1000。

## 选样和评价固定规则

对每个种子和保护来源，将上一轮保存的 200 个保护样本按 `default_rng(seed + 62021)` 生成固定排列，取前 50 和前 100 个，形成嵌套子集。保留子集后，按原 BADGE 顺序取未重复的前 950 或 900 个样本补足。训练行顺序为 BADGE 部分在前、按 ID 排序的指定保护样本在后。原共同前 800 个查询保持不变；0 和 200 配额的行顺序与保存的端点完全一致。

指定保护样本可能与原完整 BADGE 集合重合，BADGE 补足部分也可能自然包含预演误伤样本。因此报告同时列出指定保护数、与原 BADGE 重合数、实际替换行数，以及后 200 个训练样本的真实组成。指定保护数不等于全部保护性样本数量。

新查询均属于上一轮六组查询的并集。评价直接沿用该并集，**不缩小排除集合**。所有旧模型在相同评估集上重算，并核对上一轮指标；新旧结果使用相同的种子和评估样本。

原目标 CE＋参考 KD、参考索引、源分类头初始化、学习率、损失权重、温度、训练步数、batch 和目标/参考抽样位置流全部继承。设备、线程数和推断 batch 也从原 manifest 继承。新脚本不修改旧代码、原查询、原模型或原报告。

本轮选择阶段只读取已保存的查询 ID 和对应标签，不重新查看完整目标真值；但这些查询来自上轮完整真值筛选，**仍继承 oracle 信息特权**，不能声称同标注成本或可部署。训练只使用冻结查询标签；所有选择冻结后才训练，全部新训练完成后才读取完整真值评价。

## 服务器运行

在原 `traffic-ncde` 环境、仓库根目录执行：

```bash
bash Experiment/core_code/scripts/run_kbs_protection_budget.sh preflight
bash Experiment/core_code/scripts/run_kbs_protection_budget.sh run
```

`run` 依次启动选择、训练、评价三个独立进程，任一阶段失败即停止。原修复实验及 oracle 实验的全部源种子、训练和评价必须完成。中断后重跑同一命令会跳过通过核验的结果；不要同时启动第二个实例。

需要后台运行时，用下面命令代替前台 `run`：

```bash
nohup bash Experiment/core_code/scripts/run_kbs_protection_budget.sh run > kbs-protection-budget.log 2>&1 &
tail -f kbs-protection-budget.log
```

完成后查看：

```bash
cat Experiment/core_code/outputs/kbs_protection_budget_v1/evaluation/report.md
cat Experiment/core_code/outputs/kbs_protection_budget_v1/evaluation/paired_by_seed.csv
```

支持 `plan`、`preflight`、`select`、`train`、`evaluate` 和启动器的 `run`。路径默认相对于 `Experiment/core_code`：

| 环境变量 | 默认值 | 用途 |
|---|---|---|
| `ORACLE_OUTPUT_DIR` | `outputs/kbs_oracle_protection_v1` | 上轮 oracle 输入 |
| `REPAIR_OUTPUT_DIR` | `outputs/kbs_repair_aware_v1` | 原预演实验输入 |
| `STUDY_DIR` | `outputs/kbs_supplement_v1` | 原协议和 BADGE 选择 |
| `CACHE_DIR` | `$STUDY_DIR/cache` | 原缓存 |
| `CHECKPOINT` | `outputs/tls22_cnn/best_model.pt` | 原 checkpoint |
| `PROTECTION_BUDGET_OUTPUT_DIR` | `outputs/kbs_protection_budget_v1` | 本轮独立输出 |
| `PROTECTION_BUDGET_SEEDS` | `0,1,2` | 已完成 oracle 实验的种子子集 |

`GPU_ID` 可以指定可见 GPU。旧 `SEEDS`、`ORACLE_SEEDS`、`REPAIR_SEEDS`、`OUTPUT_DIR`、`DEVICE`、`STEPS` 不控制本轮的新设置。不要通过改旧 manifest 绕过核验；种子、设置或来源身份变化需要独立输出目录。

## 输出与判读

`evaluation/report.md` 提供四个配额的均值/样本标准差、相同配额下定向保护减普通正确样本的配对差值，以及各配置相对 BADGE 的差值。两组 0 配额重复展示同一 BADGE 结果，不能算成独立证据。

`curve_by_seed.csv` / `curve_summary.csv` 保存全部工作点；`paired_by_seed.csv` / `paired_summary.csv` 保存配对变化；`by_seed.csv` / `summary.csv` 包含 static、原六组及新四组；`per_class.csv` 包含全部类别；`allocation_audit.csv` 核对新配额的实际组成。选择、训练和评价均记录来源与完成文件哈希，语义核验不一致会停止。

重点看能否在接近 BADGE 崩溃类 F1 的同时减少非崩溃类负翻转，并一起检查正翻转、稳定类 F1、新崩溃、残留崩溃及逐类支持数。保护带来的恢复下降仍是取舍。

本轮是看过上轮结果后提出的 M12 开发性实验。报告所有工作点，不插值或外推，不自动选最佳配置、设通过阈值或宣称显著性。三个种子共用同一源模型和月份，不能充当独立跨环境验证或创新性证明。
