# KBS 第一批补充实验：服务器运行说明

本入口实现 E1（受控组件消融）、E2（全类别保护分析）以及可选 E3（单因素敏感性）。默认不执行检测触发的连续维护实验，也不重复已有跨期、QUIC、全模型微调或旧图 5 实验。

## 1. 拉取后先检查

在服务器仓库根目录执行，使用之前的环境：

```bash
conda activate traffic-ncde
git pull --ff-only origin main
bash Experiment/core_code/scripts/run_kbs_supplement.sh preflight
bash Experiment/core_code/scripts/run_kbs_supplement.sh smoke
```

`preflight` 检查 CUDA、数据目录、checkpoint 和主要依赖，并打印 Python 路径、版本与 GPU。它不安装包、不重新训练编码器。`smoke` 只在临时目录运行 CPU 合成数据测试，其数值不能作为论文结果。

默认数据路径是 `Experiment/core_code/data/tls22`，checkpoint 是 `Experiment/core_code/outputs/tls22_cnn/best_model.pt`。沿用仓库的 CESNET 加载逻辑；如果服务器已有另一处数据，先设置实际绝对路径：

```bash
export DATA_DIR=/你的实际路径/CESNET-TLS-Year22
# 仅当要指定某块卡时设置，编号先由 nvidia-smi 确认：
export GPU_ID=0
# 只有 checkpoint 放在其他位置时才需要：
# export CHECKPOINT=/你的实际路径/best_model.pt
bash Experiment/core_code/scripts/run_kbs_supplement.sh preflight
```

不要把示例路径照抄为真实路径。默认要求 CUDA；必要时可明确设 `DEVICE=cpu`，不会默默回落到 CPU。该入口不修改旧实验脚本、旧输出或环境中的依赖版本。

## 2. 启动第一批

查看任务矩阵：

```bash
bash Experiment/core_code/scripts/run_kbs_supplement.sh plan
```

后台运行并查看日志：

```bash
nohup bash Experiment/core_code/scripts/run_kbs_supplement.sh primary > kbs-primary.log 2>&1 &
tail -f kbs-primary.log
```

第一次先提取、校验并缓存 M-2022-4 与 M-2022-12 的冻结特征/源 logits。之后运行 5 配置 × 5 种子，共 25 次头部修复：

| 运行名 | 选样 | 目标 CE | 参考 CE | 参考 KD |
|---|---|---|---|---|
| margin_ft_only | Margin | 开 | 关 | 关 |
| margin_replay | Margin | 开 | 开 | 关 |
| margin_kd | Margin | 开 | 关 | 开 |
| margin_full | Margin | 开 | 开 | 开 |
| badge_full | BADGE | 开 | 开 | 开 |

默认目标标签预算 1,000，种子 0–4，参考集全部 178 类各取 5 个样本，KD 温度 2、权重 0.5。BADGE 沿用仓库当前实现，包括其梯度近似和降维方式，不声称实现了另一个完整 BADGE 版本。五个种子是修复/选样种子，源模型 checkpoint 固定。

同一条命令支持断点续跑：完整且哈希通过的结果会跳过；未完成的运行从源 head 重新执行。若参数、实现、数据、checkpoint 或已完成文件发生不匹配，脚本停止并要求新输出目录，避免把不同实验混到一个表中。不要在同一输出目录同时启动多个进程。

可以先做单种子服务器试运行，再用默认命令补齐剩余种子：

```bash
SEEDS=0 bash Experiment/core_code/scripts/run_kbs_supplement.sh primary
bash Experiment/core_code/scripts/run_kbs_supplement.sh primary
```

如需改变学习率、步数等协议参数，请使用新的 `OUTPUT_DIR`。修改协议后不能再把旧汇总行直接并入新表。

## 3. 本次消融的训练协议

它是新增的受控消融，**不能将其结果直接当成旧主表的同一次实验**。

每次优化使用固定大小的目标与参考 minibatch，分别从固定的独立随机序列中有放回抽样。目标序列不因开关 Replay/KD 而改变；四种 Margin 配置共用 query ID、reference ID、源 head、学习率和实际更新步数。

```text
loss = w_target * CE(target)
     + replay_enabled * w_reference * CE(reference)
     + lambda * T² * KL(teacher(reference)/T || student(reference)/T)

w_target    = (2 × budget) / (2 × budget + 178 × 5)
w_reference = (178 × 5) / (2 × budget + 178 × 5)
```

默认预算 1,000 时两者为 2,000/2,890 与 890/2,890。关闭一项时不重新归一化其他项。敏感性改变 k 时，上述权重仍固定于基准 k=5。

每种配置固定 1,380 次 AdamW 更新（基准 `30 × ceil(2890/64)`），每步目标/参考各 64 个样本，lr=0.001，weight_decay=0.0001。这里只用基准 epoch 推导默认步数；新协议使用固定步数，不再按拼接数据集长度执行不同次数的更新，也没有在代码中重复拼接目标样本。

FT+KD 与完整方法使用同一组参考样本，但参考真实标签不进入 FT+KD 的损失。蒸馏仅发生在参考样本上。源编码器与源 head 不被更新，保存的是其拷贝经过修复后的 head。

## 4. E2：全类别评估自动随 E1 生成

每次输出两个评估集合，并在同一集合上计算修复前与修复后指标：

- `strict`：排除本方法查询的所有样本，用于对应方法的配对变化。
- `common`：排除同一种子下 Margin 与 BADGE 查询样本的并集，用于跨选样方法比较。

两种集合均由无标签选样决定。最终稿的 12 个坍塌类别列表只用于事后分组评估，不用于触发、选样或训练。原 20 个 Stable 类指标与其余全部 166 个非坍塌类指标分开报告。

保存 178 类的 support、precision、recall、F1、F1 差值、混淆矩阵；汇总残余坍塌数、新增非坍塌组坍塌数、非坍塌类退化数量/比例、下降超过 0.05 的数量以及最差 10 类。

宏平均使用固定类别全集，零支持类别记 F1=0，并另报各组有支持的类别数；新增坍塌与退化计数只针对评估集中有支持的类别。新坍塌定义为修复前 recall≥0.1、修复后 recall<0.1，阈值固定。

## 5. 结果位置与完整备份

默认目录：

```text
Experiment/core_code/outputs/kbs_supplement_v1/
  cache/
    manifest.json             # checkpoint / 输入流 / 数据文件清单 / 特征缓存哈希
    head.pt
    reference.pt
    target.pt
  study_manifest.json         # 固定协议与初始运行环境
  selections/{margin,badge}/seed_*/...
  runs/<配置名>/seed_*/
    resolved_config.json
    environment.json
    repaired_head.pt
    query_ids.csv
    replay_ids.csv
    predictions.npz
    per_class_metrics.csv
    worst_noncollapse_classes.csv
    confusion_matrices.npz
    training_trace.json
    metrics.json
    run.log
    complete.json             # 最后写入，包含以上结果文件校验和
  primary_results_by_seed.csv
  primary_summary.csv
  sensitivity_results_by_seed.csv
  sensitivity_summary.csv
```

汇总保留每个种子和样本标准差；只有一个种子时标准差留空，不写成 0。汇总会列出已完成种子数及是否达到建议数量；生成表格不代表五种子已经全部完成。

`predictions.npz` 包含 `row_id/y_true/static_pred/repaired_pred/queried/strict_eval/common_eval`。ID 定义为 `缓存 fingerprint:reference或target:row_id`，绑定当次输入流顺序；这是可还原的快照行 ID，不是数据集原始 flow ID。原始 PPI/统计量/标签输入流及样本顺序的哈希保存在缓存 manifest 中。后续只靠预测与 manifest 即可重新统计全部类别；保存完整 cache 还能复用特征。

`repaired_head.pt` 中的 `cls_head_state_dict` 可直接装回同一源模型的 `model.cls_head`；文件内还保存对应协议。编码器从 manifest 指定的源 checkpoint 恢复。

耗时分开记录特征缓存、选样、头部训练及推理。RSS 是整个 Python 进程到该时刻的峰值，不能当作每配置独立进程峰值；GPU 分配峰值单列。不要与旧论文的完整独立运行耗时直接混成一张成本比较表。

实验输出受现有 .gitignore 保护，脚本不会自动提交大文件。回传分析至少保留 manifest、summary、逐种子 CSV、各运行的全部小型 CSV/JSON；完整复现还需保留 cache、预测文件及修复后的模型。

需要重新汇总时：

```bash
bash Experiment/core_code/scripts/run_kbs_supplement.sh summarize
SUITE=sensitivity bash Experiment/core_code/scripts/run_kbs_supplement.sh summarize
```

## 6. 第二批：可选敏感性

```bash
SUITE=sensitivity bash Experiment/core_code/scripts/run_kbs_supplement.sh plan
nohup bash Experiment/core_code/scripts/run_kbs_supplement.sh sensitivity > kbs-sensitivity.log 2>&1 &
```

默认种子 0–2，扫描 lambda={0,0.1,0.5,1}、T={1,2,4}、k={1,5,10}，一次只改一个参数；去重为 8 个配置，共 24 次。若第一批已经完成，协议一致的 `margin_replay` 和 `margin_full` 前三个种子会直接复用，因此新增 18 次。k 不同的参考样本按每类嵌套选取；若某类不足 k 个样本，脚本报错，不悄悄改变实际重放量。

先记录整个扫描的收益与退化，不根据最终测试月成绩重新挑选“默认参数”。需要选参时，另用更早的独立时期。

## 7. 验证范围

代码通过 CPU 合成数据单元测试和 shell 启动器测试，覆盖独立损失开关、固定更新数/抽样流、冻结源模型、样本排除、全类别混淆统计、特征缓存、断点续跑与参数/文件不匹配检查。真实 CESNET 全量运行与 CUDA 执行需在服务器验证；测试数值不是新增实验结论。

## 8. 第一批之后：BADGE 的参考 CE 对照

这一补跑要求第一批的 `badge_full` 种子 0–4 已完成，并保留原 feature cache、selections 和 runs。新配置 `badge_kd` 仍用 BADGE 查询目标标签、全部类别参考样本与参考 KD，只关闭参考样本的监督 CE。预算 1,000、参考每类 5 个、KD 权重 0.5、温度 2、更新 1,380 步均沿用第一批。

运行矩阵有两个配置、五个种子：校验并复用已有五次 `badge_full`，只新增五次 `badge_kd`。缺少基线或选样文件时会停止，不会默默重跑基线或重新选样。

```bash
cd /data/xjw/traffic-drift2
conda activate traffic-ncde
git pull --ff-only origin main
bash Experiment/core_code/scripts/run_kbs_supplement.sh badge-kd-plan
nohup bash Experiment/core_code/scripts/run_kbs_supplement.sh badge-kd > kbs-badge-kd.log 2>&1 &
tail -f kbs-badge-kd.log
```

若该服务器再次出现 `127.0.0.1:7897` 代理拒绝连接，可将上面的拉取命令替换为以下临时绕过代理的命令；它不修改 Git 配置或当前 shell 的代理设置：

```bash
env -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    git -c http.proxy= -c http.https://github.com.proxy= -c remote.origin.proxy= \
    pull --ff-only origin main
```

默认复用 `outputs/kbs_supplement_v1`；如果第一批用了自定义目录，`OUTPUT_DIR` 和 `CACHE_DIR` 也必须指向当时的目录。校验阶段会读取缓存和已完成文件的哈希，日志可能暂时没有训练步数输出。同一命令支持安全续跑，`Ctrl+C` 仅退出 `tail -f`。

补跑入口是 `scripts/kbs_badge_kd_followup.py`。原 `kbs_supplement.py` 和选样实现保持原内容，因此第一批记录的实现哈希仍可验证。补跑只注册新配置和独立汇总回调，额外记录 `badge_kd_extension_manifest.json` 中的入口哈希与运行环境；不绕过原协议、缓存或完成文件检查。

每对结果还核对查询 ID、参考 ID 及顺序、目标/参考训练抽样流、更新数、预测行 ID、标签、源预测和 strict/common 评估掩码。新汇总仅包含两种配置均完成且配对检查通过的种子。五种子未齐时会明确显示未满足建议数量；单种子标准差留空。

输出都位于原输出目录，原 `primary_*.csv` 和 25 次已完成运行保持不变：

- `runs/badge_kd/seed_0` 至 `seed_4`：与第一批相同格式的预测、模型、指标及训练轨迹。
- `badge_kd_summary.csv`：两种配置的均值、样本标准差、已配对种子数。
- `badge_kd_results_by_seed.csv`：两种配置的逐种子结果，五种子齐全时为 10 行数据。
- `badge_kd_paired_by_seed.csv`：逐种子差值，方向固定为 **badge_kd 减 badge_full**。F1 差值保留 0–1 单位；退化数/新增崩溃数的负差值表示更少类别受损。

完成后可直接复制这两份小表用于分析：

```bash
cat Experiment/core_code/outputs/kbs_supplement_v1/badge_kd_summary.csv
cat Experiment/core_code/outputs/kbs_supplement_v1/badge_kd_paired_by_seed.csv
```

如需重新校验并汇总：

```bash
bash Experiment/core_code/scripts/run_kbs_supplement.sh badge-kd-summarize
```

这是看到 Margin 消融结果之后提出的探索性对照，应报告全部五种子及有利/不利指标。它不增加独立源模型、时间段或数据集的重复数，也不自动证明新配置优于完整方法。

## 9. 逐类分析与敏感性实验顺序执行

完成 primary 的 25 次运行和 BADGE 补跑的 5 次运行后，可以顺序执行逐类分析和已有的 Margin 敏感性实验：

```bash
cd /data/xjw/traffic-drift2
conda activate traffic-ncde
# 先按第 8 节说明拉取 main，再启动：
nohup bash Experiment/core_code/scripts/run_kbs_supplement.sh stage2 > kbs-stage2.log 2>&1 &
tail -f kbs-stage2.log
```

`stage2` 依次执行：

1. 读取六种配置、种子 0–4 的现有逐类指标、选样记录和运行元数据，生成 `class_audit/`。不加载模型、预测矩阵或原始数据，不进行训练。分析输入先通过小文件哈希、配置一致性、共同评估集的修复前逐类指标/支持数、BADGE 查询/参考 ID 及逐类指标与原汇总的一致性检查。分析失败时停止，不继续启动训练。
2. 验证并复用特征缓存，执行第 6 节的 Margin 单因素敏感性：8 个唯一配置 × 种子 0–2。复用 `margin_replay` 和 `margin_full` 的 6 次已有运行，默认新增 18 次头部修复；已完成的敏感性运行也会跳过。
3. 生成 `sensitivity_matched_*.csv`，严格选取每种配置的同一组训练种子（默认 0–2）。原引擎的通用汇总会收进两个基线已保存的种子 3–4，默认形成 28 行，而其他敏感性配置只有三个种子；统一比较应使用新的 matched 表，默认恰好 24 行。

继续使用第一批的 `OUTPUT_DIR`、`CACHE_DIR`、checkpoint、数据路径与环境；原训练引擎、BADGE 补跑入口及其哈希保持不变。重复执行 `stage2` 可重新生成派生报告并续跑敏感性。不要同时对同一输出目录启动其他写入作业。

只分析已有结果时执行：

```bash
bash Experiment/core_code/scripts/run_kbs_supplement.sh class-audit
```

逐类分析固定使用 `common` 口径，新增崩溃、残余崩溃和严重退化沿用 `study_manifest.json` 的定义；零支持类别不计入新增崩溃或退化。报告显示支持数，不能为了改变结论临时过滤低支持类别。默认要求六种配置的五个种子齐全；检查部分种子时可显式设置 `AUDIT_SEEDS=0,1`，报告会标明实际种子数和未满足五种子建议。

`SEEDS` 仍只控制训练阶段；`AUDIT_SEEDS` 控制逐类分析。默认训练三个种子、分析已有五个种子。`stage2` 的额外 Python CLI 参数只传给敏感性训练；更改训练协议时需要单独的新实验安排，不能混入原目录。

输出文件：

| 文件（相对输出目录） | 内容 |
|---|---|
| `class_audit/report.md` | 中文摘要、六配置比较、BADGE 新增/残余崩溃类别，以及平均 F1 降低最多的类别 |
| `class_audit/summary.csv` | 各配置均值、样本标准差、分析种子数 |
| `class_audit/all_classes_by_seed.csv` | 全部 178 类 × 6 配置 × 5 种子的 F1/recall/支持数、查询量、参考样本量及退化标记 |
| `class_audit/cases_by_seed.csv` | 新增/残余崩溃、严重退化、每次最差十个非崩溃类的明细 |
| `class_audit/class_recurrence.csv` | 各类别受损次数、对应种子、支持数范围及最差变化 |
| `class_audit/badge_class_comparison.csv` | 两个 BADGE 配置逐类配对比较，含 KD 减 Full 的平均 F1 |
| `class_audit/provenance.json` | 分析脚本和读取小文件的哈希、协议、种子及验证范围 |
| `sensitivity_matched_summary.csv` / `sensitivity_matched_results_by_seed.csv` | 相同种子上的敏感性比较，默认正常完成为 8 行配置汇总和 24 行逐种子结果 |

原 `runs/`、`selections/`、预测/模型和 primary/BADGE 汇总保持原样；重新分析只更新派生的 `class_audit/` 文件。该分析不重新核验大型预测/模型文件，也不推断类别名称或退化因果。

分析报告先于训练结果生成，可直接复制回传：

```bash
cat Experiment/core_code/outputs/kbs_supplement_v1/class_audit/report.md
```

敏感性全部完成后再回传：

```bash
cat Experiment/core_code/outputs/kbs_supplement_v1/sensitivity_matched_summary.csv
cat Experiment/core_code/outputs/kbs_supplement_v1/sensitivity_matched_results_by_seed.csv
```

`sensitivity` 和 `stage2` 都会自动生成 matched 表。已有敏感性结果时，执行 `bash Experiment/core_code/scripts/run_kbs_supplement.sh sensitivity-report` 即可单独生成；缺少任一请求配置/种子时会停止，不把不齐的种子混入比较。

本阶段不自动更换默认方法或按最终测试月选择最优超参数。跨时段的新旧 BADGE 对照仍需在决定调整最终方法后单独安排。
