# CARE 新选样种子复验

本轮验证特征覆盖相对于类别均衡的收益是否依赖上一轮的单个查询集。沿用普通线性 CARE、源模型、M4 参考、M12 目标和所有拟合参数；不增加硬约束、不调预算或候选池参数。下一步跨月份目标预先定为 M11，本入口先列出现有缓存元数据，不提取新月份特征。

## 预先固定的比较

只比较 `mixed_balanced` 与 `mixed_coverage`。每臂 1000 个目标标签，共享 800 个原始 logit Margin 查询，另选 200 个补充查询。每个选样种子内，两臂的高置信候选池和源预测类别配额完全相同；区别只在层内随机选样与特征中心/最远点覆盖。

- **新选样种子**：`6174,6175,6176,6177,6178`，全部纳入验证。
- **训练种子**：沿用已完成实验的 `[0,1,2]`，每个选样种子内严格配对参考行、初始化和抽样索引流。
- **开发对照**：原选样种子 `6173` 的两臂分类头直接复用，不计入新种子主汇总。
- **预算基线**：原纯 Margin 三个分类头直接复用，用于评价相同 1000 标签预算的净收益。

新选样种子控制各预测类别候选 reservoir、轮转配额顺序和层内随机抽样。800 个修复查询固定，因此这轮验证的是补充查询的抽样变化，不是修复核心或月份变化。

默认新增 **5 个选样种子 × 2 臂 × 3 个训练种子＝30 次 CPU 拟合**，复用 9 个分类头，最终重评 39 个头。目标标签预算是每臂、每次选样重复的预算；多个重复的标签并集不受 1000 限制。

主统计单位是**选样种子**：先算每个新选样种子内三个训练种子的配对差异均值，再对五个选样种子汇总均值和样本标准差。不能将 15 个训练重复当成 15 个独立查询重复。开发种子完整显示在逐种子表中，但从验证汇总排除；不选择收益最大的种子。

## 服务器当前终端前台运行

前置条件是上一轮 `care_mixed_query_ablation_v1` 已完整完成，其依赖的保护试验、完整复评、小诊断和缓存仍在原目录。使用现有 `traffic-ncde` 环境，无新增依赖；CPU 8 线程，CUDA 隐藏。命令只恢复新文件，不改旧数值引擎或结果。

```bash
cd /data/xjw/traffic-drift2 &&
conda activate traffic-ncde &&
env -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    git -c http.proxy= -c http.https://github.com.proxy= \
        -c remote.origin.proxy= fetch origin codex/small-repair-diagnosis &&
git restore --source=FETCH_HEAD --worktree -- \
    Experiment/core_code/scripts/care_query_seed_robustness.py \
    Experiment/core_code/scripts/run_care_query_seed_robustness.sh \
    Experiment/core_code/CARE_QUERY_SEED_ROBUSTNESS.md &&
bash Experiment/core_code/scripts/run_care_query_seed_robustness.sh run
```

`run` 串行执行四个前台进程：月份缓存清单、冻结所有查询、拟合/复用全部头、独立评价。训练期间显示选样种子、训练种子、查询臂和步数；失败立即停止。重复同一命令可在校验后继续，完成项不重训。

出现下面完成标记后显示报告：

```text
CARE_QUERY_SEED_ROBUSTNESS_COMPLETE: .../care_query_seed_robustness_v1/evaluation/report.md
```

```bash
sed -n '1,36p' Experiment/core_code/outputs/care_query_seed_robustness_v1/evaluation/report.md
```

单独查询计划、校验输入或显示缓存元数据：

```bash
bash Experiment/core_code/scripts/run_care_query_seed_robustness.sh plan
bash Experiment/core_code/scripts/run_care_query_seed_robustness.sh preflight
bash Experiment/core_code/scripts/run_care_query_seed_robustness.sh inventory
```

`MONTH_CACHE_INVENTORY` 默认扫描仓库 `outputs/` 内小型缓存 manifest，打印月份、样本数和三个张量文件的存在性。它**不校验张量哈希**，不证明原始年度文件含有可加载的 M11，也不能把 M12 缓存直接用于 M11。若现有缓存存在其他目录，可用 `CARE_QUERY_SCAN_ROOT` 改扫描根目录；扫描有深度、条目数和时间上限，截断会明示。

路径覆盖使用 `CARE_QUERY_SOURCE_DIR`、`CARE_QUERY_AUDIT_DIR`、`CARE_QUERY_PROTECTION_DIR`、`CARE_QUERY_ABLATION_DIR`、`CARE_QUERY_CACHE_DIR`、`CARE_QUERY_CHECKPOINT`、`CARE_QUERY_OUTPUT_DIR`。线程必须与已完成来源一致。新入口不继承旧 `SEEDS`、`QUERY_SEEDS`、`STEPS`、`OUTPUT_DIR` 或 `CARE_MIXED_*` 的数值设置。`--query-seeds` 可用于新的预先声明协议，但改配置必须使用新的输出目录，且不能把开发种子放入新种子集合。

## 结果与判读

独立输出 `outputs/care_query_seed_robustness_v1/`，包含：

- `manifest.json`：新旧协议、代码、已完成来源和复用头哈希。
- `selection/`：所有查询、修复/补充分区、各选样种子的候选池/配额、共同评价行和完成标记。
- `runs/query_seed_<n>/<selector>/seed_<n>/care/`：分类头、付费标签、参考行、抽样流和拟合/复用记录。
- `evaluation/by_seed.csv`、`per_query_summary.csv`、`paired_by_seed.csv`、`paired_by_query.csv`：训练重复和选样重复分层结果。
- `evaluation/fresh_arm_summary.csv`、`fresh_paired_summary.csv`：仅新选样种子的主汇总和正负收益种子数量。
- `evaluation/per_class.csv`、`query_coverage.csv`、`predictions.npz`、`report.md`：逐类精度/召回/F1/翻转、付费覆盖和完整预测。

全部查询先冻结，训练只访问对应付费目标标签。先核验所有新分类头及复用来源完成，才在独立进程读取未查询目标真值。每个头使用同一完整 M12 评价集，排除**所有新旧查询的并集**。复用的开发头和 Margin 头重新推理，旧报告的不同评价集数值不能直接拿来作差。

主问题是 `coverage−balanced` 在新选样种子上的均值、离散程度和正收益种子数量。总体 Macro-F1、崩溃组 F1 和负翻转率分别判断，不将恢复收益解释成保护效果。均值只由预先指定的新种子组成；`±` 是标准差，不是置信区间。

若新种子大多保持优势，说明值得进入预先定好的 M11 复验；若收益经常反号，就应收缩“特征覆盖稳定有效”的主张。这一轮仍使用已分析过的 M12，不能证明跨月份、新颖性或目标全池安全。合成测试只验证实现和统计汇总。
