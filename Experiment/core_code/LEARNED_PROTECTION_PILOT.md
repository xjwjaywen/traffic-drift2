# 固定 50 配额的实际保护选样预实验

## 问题和范围

理想保护配额实验在 M12 上提示，50 个定向保护样本可能保留崩溃类恢复并减少误伤。本轮检查仅用实际可取得的标签和模型输出，能否找到有用的保护样本。默认种子 **0、1、2**，四组各训练一次，共 **12 次新的最终修复**；另外每个种子拟合一个四特征的小型逻辑回归，复用原预演模型和 BADGE 最终模型，不训练新编码器。

本轮设计是在观察 M12 oracle 结果之后提出的开发性预实验，不是独立确认或已成立的新方法。50 配额为总预算的 5%；真实协议 B=1000。所有样本的计费单位、参考集和修复设置与原实验一致。

## 先付费、再选样

1. 原 BADGE 前 800 个标签用于保存的预演模型；本轮直接复用该模型。
2. 购买 BADGE 第 801–950 个标签。这 150 个样本没有参加预演训练；四组共用相同的前 950 个查询。
3. 使用这 150 个样本中发生源→预演预测翻转的样本拟合风险排序。条件是已经发生翻转，因此“源预测正确”正好等价于“源正确、预演错误”。其余类别信息不参与排序。
4. 各组补采 50 个此前未查询的样本，全部计入预算并用于最终修复。即使查询后发现它们是纠错样本或两模型都错，也不丢弃、不追加查询。

完整目标真值、旧 oracle 选择、最终修复预测、真实崩溃类列表和源数据的标签都不会进入选样。旧缓存将标签与特征打包，本轮选择进程只索引这 150 个付费标签，训练只索引冻结的 1000 个查询。完整真值只在所有种子和组训练结束后的独立评价进程中用于离线评估。历史参考标签另计，所有组一致。

## 四组及固定规则

| 方法 | 后 50 个样本的来源 |
|---|---|
| `random50` | 全部未查询样本中均匀无放回选取 |
| `flip_uniform50` | 源模型与预演模型预测不一致的未查询样本中均匀选取 |
| `flip_confidence50` | 同一翻转池内按源模型最大 softmax 概率降序选取 |
| `flip_risk50` | 同一翻转池内按已查询标签训练的风险打分降序选取 |

源置信度不等于真实正确性；该组用于检查学习打分是否比简单置信度排序更有用。

风险模型输入四个有界数值：源最大概率、源 top-2 概率差、预演分配给源原预测类别的概率、预演 top-2 概率差。标准化参数只来自付费拟合样本中发生翻转的部分。使用不加类别权重的二元逻辑回归，目标为二元交叉熵**求和**加 `0.5 * ||w||²`，截距不惩罚。Newton 迭代最多 50 次，梯度容差 `1e-8`，固定回溯线搜索，不搜索特征、超参数或阈值。

若正例或反例任一不足 5 个，或求解未收敛，预先约定回退到源置信度排序，报告原因；此时风险组和置信度组选择及训练顺序完全相同，不能当作两个独立成功结果。

使用 `default_rng(seed+72021)` 生成共同随机优先序：均匀组取前 50 个，排序组以该顺序处理并列值。翻转池不足 50 时取全池，再从剩余未查询行均匀补足。前 950 行保持 BADGE 原顺序，后 50 行按 ID 排序；所有标签均保留。

最终修复沿用源头部初始化、参考行、目标 CE＋参考 KD、学习率、权重、温度、步数、batch 及训练抽样位置流。新增信号只需对保存的预演头做一次分批全池推断，核对重建预测与旧预测完全一致。不会修改原脚本或结果。

风险模型拟合数据来自 BADGE 选择分布，不能将输出当作全池经过校准的概率。排序也可能集中于少数类别；报告同时提供所选样本的真实类覆盖和预测类覆盖，供事后分析。

## 服务器运行

在原 `traffic-ncde` 环境、仓库根目录运行：

```bash
bash Experiment/core_code/scripts/run_kbs_learned_protection.sh preflight
bash Experiment/core_code/scripts/run_kbs_learned_protection.sh run
```

`run` 按选择→训练→评价启动三个独立进程，任一步失败即停止。旧 `kbs_repair_aware_v1` 的全部源种子、修复和评价必须完成。无需提供 oracle 结果；不把 oracle 查询当作已付费样本。

需要后台运行时，用下面命令代替前台 `run`，不要同时启动两个实例：

```bash
nohup bash Experiment/core_code/scripts/run_kbs_learned_protection.sh run > kbs-learned-protection.log 2>&1 &
tail -f kbs-learned-protection.log
```

中断后重跑相同命令，跳过已核验完成项。启动器还支持 `plan`、`select`、`train`、`evaluate`。

| 环境变量 | 默认值 | 用途 |
|---|---|---|
| `REPAIR_OUTPUT_DIR` | `outputs/kbs_repair_aware_v1` | 旧预演/修复输入 |
| `STUDY_DIR` | `outputs/kbs_supplement_v1` | 原协议和 BADGE 排序 |
| `CACHE_DIR` | `$STUDY_DIR/cache` | 原缓存 |
| `CHECKPOINT` | `outputs/tls22_cnn/best_model.pt` | 原 checkpoint |
| `LEARNED_PROTECTION_OUTPUT_DIR` | `outputs/kbs_learned_protection_v1` | 本轮独立输出 |
| `LEARNED_PROTECTION_SEEDS` | `0,1,2` | 已完成原预演实验中的种子 |

路径相对于 `Experiment/core_code`。设备、线程数和推断 batch 继承原实验；`GPU_ID` 可指定可见 GPU。旧 `SEEDS`、`ORACLE_SEEDS`、`PROTECTION_BUDGET_SEEDS`、`DEVICE`、`STEPS` 和旧输出变量不控制本轮的新设置。

## 评价和返回结果

新查询可能超出旧排除集合。因此共同评估集排除**旧四组查询与新四组查询的并集**；旧指标先在旧集合重算核对，再重算 BADGE/static 的新共同集合指标。不得将本轮数字直接拼入旧 oracle 配额表。原四组中除了 BADGE，其余只参与来源核验和历史查询排除，不作为相同 50 配额的方法展示。

完成后返回：

```bash
cat Experiment/core_code/outputs/kbs_learned_protection_v1/evaluation/report.md
cat Experiment/core_code/outputs/kbs_learned_protection_v1/evaluation/paired_by_seed.csv
```

`evaluation/` 还提供 `by_seed.csv`、`summary.csv`、`paired_summary.csv`、`per_class.csv`、`acquisition.csv`、`candidate_audit.csv`。`selections/seed_N/` 保存固定查询、拟合标签、信号、风险模型参数与增量耗时；`runs/` 保存最终头部、预测、查询标签与训练流。来源身份和完成文件均校验哈希，语义不一致会停止。

先比较风险组与翻转均匀组、源置信度组，再比较 BADGE 的误伤和崩溃类恢复。查询阶段命中率与最终修复结果需分开判断：预演误伤样本不一定是最终误伤样本，命中更多也不保证训练后改善。报告正/负翻转、新/残留崩溃及全部类别，不自动选赢家或声明显著性。三个种子共用源模型和月份，不能证明跨环境泛化或创新性。
