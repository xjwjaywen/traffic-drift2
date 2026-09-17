# 实际保护选样与 oracle 的差距诊断

本轮只分析已完成结果，不新增训练、标注、选样、阈值扫描或全池模型推断。
目的是检查三个可能的解释：实际选样的类别/特征覆盖较窄，预演误伤与最终误伤不一致，
或者选中样本在共同分类头下的 CE 梯度较弱。任何一个诊断指标都不能独立证明因果关系。

## 运行

在原 `traffic-ncde` 环境、仓库根目录执行：

```bash
bash Experiment/core_code/scripts/run_kbs_protection_gap_diagnosis.sh plan
bash Experiment/core_code/scripts/run_kbs_protection_gap_diagnosis.sh preflight
bash Experiment/core_code/scripts/run_kbs_protection_gap_diagnosis.sh run
```

`run` 自带全部前置核验，所以实际运行可直接执行第三条。使用 CPU，默认 8 线程，
CUDA 对该进程隐藏。读取并核对大缓存的哈希需要磁盘 I/O；不重新提取特征。
脚本只对每组已选尾部的少量特征计算余弦和线性头 CE 梯度，不载入参考特征缓存。
保持原 Python/NumPy/PyTorch 版本，以便原研究的身份校验通过。

默认种子为 `0,1,2`。所有源研究须完整完成其各自登记的种子；诊断可以选取这些种子的交集子集。
参数可用 `--seeds 0,1,2 --threads 8` 覆盖。独立环境变量如下：

| 变量 | 默认值（相对 `Experiment/core_code`） |
|---|---|
| `PROTECTION_GAP_SEEDS` | `0,1,2` |
| `DIAG_THREADS` | `8` |
| `PROTECTION_GAP_OUTPUT_DIR` | `outputs/kbs_protection_gap_diagnosis_v1` |
| `LEARNED_PROTECTION_OUTPUT_DIR` | `outputs/kbs_learned_protection_v1` |
| `PROTECTION_BUDGET_OUTPUT_DIR` | `outputs/kbs_protection_budget_v1` |
| `ORACLE_OUTPUT_DIR` | `outputs/kbs_oracle_protection_v1` |
| `REPAIR_OUTPUT_DIR` | `outputs/kbs_repair_aware_v1` |

也接受 `STUDY_DIR`、`CACHE_DIR`、`CHECKPOINT`、`PYTHON`；不继承旧实验的种子、GPU 或通用输出变量。
输出目录必须独立于所有源目录。重复运行只验证已有文件，不重算；源文件损坏、缺少阶段或设置改变会停止。

## 共同评估

展示八组：static、BADGE、random50、flip_uniform50、flip_confidence50、flip_risk50、
oracle_probe_damage_p05、oracle_source_correct_p05。

每个 seed 的评估排除列表是 learned-protection 和 protection-budget **两个历史排除列表的并集**。
保留未展示历史组的排除项，不缩减回八组查询。先在各自原评估集上重算并核对这些组的旧指标，再统一重算。
原先不同掩码的汇总表不能直接拼接。oracle 仍然使用了完整目标真值来选择样本，是特权机制参照，
不是同标注成本的可部署方案，也不被称为严格上界。

每组核对真实训练查询/标签。前 95% 是否与原 BADGE 查询顺序完全一致另外报告：
oracle 的补采项如与原 BADGE 重合，原补齐规则可能改变前缀。该情况不隐藏，也不归因于尾部单一差异。

## 三组诊断

1. **覆盖与多样性**：全尾部及其中预演误伤子集分别报告样本数、真实类别数、
   有效类别数 `1/sum(p_c²)`、最大类别占比、特征范数、两两余弦及最近邻余弦。
   零向量不参与余弦；不足两个有效向量记为空。有效类别数不代表特征区域覆盖。
   另报尾部预演误伤样本所涉及的真实类别，占共同测试集非崩溃类 BADGE 损伤的多少。
   这是粗粒度类别覆盖，不是近邻覆盖。
2. **预演误伤的持续性**：查询可能进入 BADGE 最终模型的训练，因此首先排除尾部中
   与 **全部 BADGE 训练查询** 重合的样本，报告剩余样本的最终误伤率，以及预演误伤持续到最终 BADGE 的比例。
   BADGE 自身尾部全部重合，对应条件率为空，不是零。它们仍是查询组成的事后统计，不能当作独立测试效果。
3. **保护样本的 CE 梯度**：只在三个共同保存的头（source、probe、BADGE final）上计算，
   不把不同方法自身的最终头混在一起。若 `r=p−onehot(y)`，含偏置的单样本梯度是
   `r [z;1]^T`，其范数为 `||r|| sqrt(||z||²+1)`。
   报告真类概率、CE、单样本梯度范数均值/中位数、平均梯度范数与方向一致性比。
   方向一致性比为 `||mean(g)|| / mean(||g||)`，受子集大小影响。
   全部使用 CPU float64 稳定 softmax；分类对错使用原保存预测，不由本轮重算 argmax 替代。
   这是未加权 CE 的局部量，不包含 KD、AdamW 状态或训练轨迹，不能直接解释为实际参数更新或保护贡献。

## 输出

目录：`Experiment/core_code/outputs/kbs_protection_gap_diagnosis_v1/`

- `report.md`：共同结果、配对差异、选样组成、多样性、梯度和解释边界。
- `by_seed.csv` / `summary.csv`：共同测试指标。
- `paired_by_seed.csv` / `paired_summary.csv`：实际选样和 oracle 的配对差值。
- `selection_audit.csv` / `selection_summary.csv`：前缀差异、训练重合、持续性、类别覆盖。
- `cohorts.csv` / `cohort_summary.csv`：尾部及预演误伤子集的覆盖、多样性。
- `gradients.csv` / `gradient_summary.csv`：全部方法、两个子集、三个共同头的梯度。
- `per_class.csv`：全部类别的共同测试指标及查询计数。
- `excluded_ids.json`：每个 seed 的共同排除样本 ID。
- 身份和完成文件：源身份、源报告/运行哈希、新脚本哈希及输出文件哈希。

所有率的零分母都记为空，汇总报告每个指标的有效种子数及样本标准差。
不新增种子，不根据诊断自动生成下一轮策略、不作显著性、创新性或机制成功判定。
M12 是开发数据；这些结果不能充当独立确认。

查看结果：

```bash
cat Experiment/core_code/outputs/kbs_protection_gap_diagnosis_v1/report.md
cat Experiment/core_code/outputs/kbs_protection_gap_diagnosis_v1/paired_by_seed.csv
```
