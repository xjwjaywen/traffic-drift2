# 理想保护样本机制对照

## 要回答的问题

已完成的预演诊断显示翻转区域能覆盖较多最终误伤，但区域也包含大量被纠正的样本。本轮检查：在当前修复器下，专门补充预演会误伤的正确样本，是否比一般性补充正确样本更能减少最终损伤，同时保留崩溃类恢复？

这是一轮使用完整目标真值的机制诊断。两个新对照都不是实际可部署的选样方法，也不是严格性能上界。每组用 1000 个目标训练标签，并不代表只消耗了 1000 个标签的信息；筛选过程访问了整个目标池的真值。与原 BADGE 等方法的比较只能提供背景，不能宣称同标注预算公平胜出。

## 固定设计

默认使用原实验中的种子 **0、1、2**，每个种子只增加两次训练，共 **6 次**。选样比例按原协议训练样本数计算，真实实验是 800＋200：

| 对照 | 共同前 800 个查询 | 后 200 个查询 |
|---|---|---|
| `oracle_probe_damage` | 保存的 BADGE 前缀 | 源模型正确且保存的预演模型错误的样本中均匀无放回采样 |
| `oracle_source_correct` | 相同 BADGE 前缀 | 所有源模型正确的样本中均匀无放回采样 |

两组都先从候选池排除共同前缀。普通正确样本组允许抽到预演误伤区域，不能把它改成“排除误伤区域”而不改变研究问题。后 200 个索引排序，再接到原前缀后；每组使用固定的 `default_rng(seed + 52021)`。若合格候选不足 200 个，则停止，不补齐、不重复抽样。

候选筛选仅使用源预测、保存的预演预测、目标真值以及旧查询记录；不读取任何最终修复预测、最终误伤集合或测试 F1。不会按修复结果挑种子或样本。

预演模型直接复用，不重训。最终修复继承原来的目标 CE＋参考 KD、源分类头初始化、参考样本索引、学习率、权重衰减、损失权重、温度、优化步数、训练 batch 和抽样位置流。新对照不引入加权损失、冻结类别或其他训练改变。设备、线程数和推断 batch 也从原 manifest 继承；启动器只允许用 `GPU_ID` 指定可见 GPU，不受旧 `DEVICE` 或 `STEPS` 环境变量影响。

## 服务器运行

在原 `traffic-ncde` 环境和仓库根目录运行：

```bash
bash Experiment/core_code/scripts/run_kbs_oracle_protection.sh preflight
bash Experiment/core_code/scripts/run_kbs_oracle_protection.sh run
```

也可以直接运行 `run`：每个阶段都会先验证全部原结果。`preflight` 核验来源及设备；候选池大小在 `select` 阶段核验。

若需要退出终端后继续运行：

```bash
nohup bash Experiment/core_code/scripts/run_kbs_oracle_protection.sh run > kbs-oracle-protection.log 2>&1 &
tail -f kbs-oracle-protection.log
```

已经启动后不要同时再启动第二个实例；输出目录使用非阻塞文件锁。中断后重新执行同一命令，已核验的选择和训练会跳过。不要同时运行前台与后台命令。

完成后查看：

```bash
cat Experiment/core_code/outputs/kbs_oracle_protection_v1/evaluation/report.md
cat Experiment/core_code/outputs/kbs_oracle_protection_v1/evaluation/paired_by_seed.csv
```

入口支持 `plan`、`preflight`、`select`、`train`、`evaluate`；`run` 按顺序启动选择、训练、评价三个独立进程。原 5 个种子的完整修复实验及评价必须都已完成。新选择全部冻结后才允许训练；新请求种子的全部训练完成后才允许评价。

路径默认相对于 `Experiment/core_code`：

| 环境变量 | 默认值 | 用途 |
|---|---|---|
| `REPAIR_OUTPUT_DIR` | `outputs/kbs_repair_aware_v1` | 原预演实验输入 |
| `STUDY_DIR` | `outputs/kbs_supplement_v1` | 原协议和 BADGE 选择 |
| `CACHE_DIR` | `$STUDY_DIR/cache` | 原缓存 |
| `CHECKPOINT` | `outputs/tls22_cnn/best_model.pt` | 原 checkpoint，核对哈希 |
| `ORACLE_OUTPUT_DIR` | `outputs/kbs_oracle_protection_v1` | 新输出，必须与原目录分开 |
| `ORACLE_SEEDS` | `0,1,2` | 必须是原完整实验中已有种子 |

旧 `SEEDS`、`REPAIR_SEEDS`、`OUTPUT_DIR` 不控制新实验。选择更多种子会增加训练量，且需要新的输出目录；本轮默认保持三个种子。没有新增训练超参数选项。

## 评价与输出

每个种子重新构造六组查询的并集：原 `badge`、`random`、`flip_uniform`、`flip_relation`，以及两个新 oracle 对照。全部方法和 static 基线都在排除该并集后的同一评估集上计算指标。原四组只读取保存的预测，不训练；先在原四组评估集上核对旧指标，再在新共同评估集上重算。不能把新结果直接拼到旧五种子汇总表。

输出保存在独立目录：

- `study_manifest.json`：固定设置、来源哈希、种子及原运行设置。
- `selections/seed_N/`：新查询、完整真值查看数量、各候选池大小、训练标签、共同排除集合。
- `runs/METHOD/seed_N/`：新分类头、全池预测、查询行、训练流和时间记录。
- `evaluation/report.md`：中文报告及结论边界。
- `evaluation/by_seed.csv` / `summary.csv`：三个种子、七个模型（包含 static）的重新计算指标及均值/样本标准差。
- `evaluation/paired_by_seed.csv` / `paired_summary.csv`：主要对照与背景比较的配对增量。
- `evaluation/per_class.csv`：全部类别的支持数、前后 F1/召回率、正负翻转、查询及补充查询分布。
- `evaluation/selection_audit.csv`：各组补充查询中源正确、预演误伤、崩溃类数量与类别覆盖。

来源代码、缓存、选择、训练和评价均校验哈希。新结果另外记录完整性清单。任何身份、完整性或重算指标不一致都会拒绝继续，原结果不会被覆盖。

## 如何判读

主要比较 `oracle_probe_damage − oracle_source_correct`，同时看崩溃类 F1、非崩溃类负翻转、正翻转、稳定类 F1、新崩溃和逐类分布。损伤下降但恢复下降仍是取舍；不从整体均值推断所有类别都改善。

本轮不设事后“过关阈值”，不自动宣称显著性、创新性或成功。三个种子共用同一源模型和 M12，不是独立跨环境验证。若定向保护有收益，下一阶段才研究如何用实际可用信号找到这些样本；若无优势，应先分析采样覆盖和当前修复器如何利用标签，不能据此断言所有保护策略无效。
