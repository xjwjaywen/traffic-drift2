# M11 特征覆盖：恢复与误伤诊断

本轮复用已完成 M11 的 33 个分类头、全部冻结查询、完整共同评价预测及 M11 特征缓存，解释 `coverage−balanced` 的逐类收益和损伤。不增加训练、不提取骨干特征、不新增标签，不修改旧数值脚本或完成结果。全量未查询真值仅用于这次事后诊断；诊断不是新查询策略或因果证明。

## 检查什么

1. **原结果对账**：校验完整 M11/M12 依赖链与缓存/代码/结果哈希，重新推理全部 33 个头，要求和原完整评价预测逐行一致。对账全部逐头指标、逐类结果、付费覆盖、配对差异及五查询种子的分层主汇总。
2. **逐类贡献**：对每个配对头计算 F1、Precision、Recall、TP、FP、FN 差异，以及该类对总体 Macro-F1 的贡献。原先 F1<0.1 且有支持的 M11 类与其余类贡献必须加总到原 Macro-F1 差异。
3. **恢复和误伤分开**：统计额外救回源错分、恢复均衡头的误伤、新增源正确误伤、丢失均衡头已修复、错到另一错等互斥事件。真实类的负翻转与错误流入某类造成的精度下降分别报告。
4. **错分去向**：保存每个真实类/源预测/均衡预测/覆盖预测四元组计数，定位谁被错分到谁。没有出现的训练/查询种子计数补零，避免只平均出现过的边。
5. **付费查询差异**：比较补充查询覆盖的真实类别、源正确/错误标签、低 F1 类标签、新增错分关系，以及在相同预测类别候选池上的单位特征覆盖距离。付费类别/关系信息均在原实验完成后审计。
6. **边界与近邻**：对全部 M11 低 F1 类、事后 F1 损伤与负翻转最多的各前 10 类，按每类/配对事件/拟合均匀抽样，默认最多 64 行。对冻结线性头的真实类与竞争类 logit gap 差精确拆分为特征权重项和偏置项；比较样本到同真类付费查询/补充查询的单位特征最近距离。

最近距离只能在确有同真类邻居时定义，缺邻居记 NA，不填零；保存完整抽样行号、母体/样本数。候选池距离只描述原高间隔 capped reservoir，不代表全目标分布。事后聚焦和事件条件抽样会产生选择偏差，几何与关系支持只作线索，不能声称某个查询因果导致收益或误伤。

主统计先平均同一查询种子的三个训练重复，再汇总全部五个查询种子，± 为 SD。连续探针量无事件时为 NA，输出有效重复数量；稀疏错分计数未出现时才补零。类别以已绑定整数 ID 报告，目前没有经核验的应用名称映射。

## 当前终端前台运行

使用已有 `traffic-ncde`，要求 `care_m11_replication_v1` 已完整完成。只恢复这轮新文件：

```bash
cd /data/xjw/traffic-drift2 &&
conda activate traffic-ncde &&
env -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    git -c http.proxy= -c http.https://github.com.proxy= \
        -c remote.origin.proxy= fetch origin codex/small-repair-diagnosis &&
git restore --source=FETCH_HEAD --worktree -- \
    Experiment/core_code/scripts/care_m11_mechanism_diagnostic.py \
    Experiment/core_code/scripts/run_care_m11_mechanism_diagnostic.sh \
    Experiment/core_code/CARE_M11_MECHANISM_DIAGNOSTIC.md &&
bash Experiment/core_code/scripts/run_care_m11_mechanism_diagnostic.sh run
```

串行三个独立前台进程：`audit → geometry → report`。全程 CPU 8 线程、CUDA 隐藏；先逐头显示预测对账进度，再显示查询几何/探针进度。主要开销是缓存哈希、33 个小头的完整池推理与诊断计算，不加载 M4 参考张量，不运行训练或安装依赖。失败立即停止，重复同一命令可在校验后跳过完成阶段；配置/代码改变必须使用新诊断输出目录。

出现完成标记后显示前半报告：

```text
CARE_M11_MECHANISM_DIAGNOSTIC_COMPLETE: .../care_m11_mechanism_diagnostic_v1/report/report.md
```

```bash
sed -n '1,65p' Experiment/core_code/outputs/care_m11_mechanism_diagnostic_v1/report/report.md
```

后半报告包括新增负翻转类、错分去向、查询几何和边界探针，后续可显示：

```bash
sed -n '66,155p' Experiment/core_code/outputs/care_m11_mechanism_diagnostic_v1/report/report.md
```

独立运行阶段或只读校验：

```bash
bash Experiment/core_code/scripts/run_care_m11_mechanism_diagnostic.sh plan
bash Experiment/core_code/scripts/run_care_m11_mechanism_diagnostic.sh preflight
# 也可分别运行 audit、geometry、report
```

路径环境变量均以 `CARE_M11_DIAG_` 开头：`SOURCE_DIR`、`AUDIT_DIR`、`PROTECTION_DIR`、`ABLATION_DIR`、`REPLICATION_DIR`、`SOURCE_CACHE_DIR`、`CACHE_DIR`、`CHECKPOINT`、`MONTH_DIR`、`OUTPUT_DIR`、`DATA_DIR`。`CARE_M11_DIAG_THREADS` 控制诊断线程，原实验来源用其已保存的运行设置校验。入口不继承 `SEEDS`、`QUERY_SEEDS`、`STEPS`、`DEVICE`、`OUTPUT_DIR` 或旧 `CARE_M11_*` / `CARE_QUERY_*` 设置。CLI `--probe-cap`、`--probe-seed`、`--top-k` 改变诊断抽样/展示范围，不改变原实验，需新输出目录。

## 输出

独立目录 `outputs/care_m11_mechanism_diagnostic_v1/`：

- `manifest.json`：输入、来源完成状态、诊断代码、运行与抽样设置。
- `audit/reconciliation.json`：所有头预测/指标对账和类别贡献/翻转加总检查。
- `audit/paired_classes.csv`、`class_by_query.csv`、`class_summary.csv`：逐类配对变化、查询均值与主汇总。
- `audit/paired_cohorts.csv`、`cohort_summary.csv`、`paired_edges.csv`、`edge_summary.csv`：互斥事件与错分四元组。
- `audit/group_contributions.csv`、`group_summary.csv`：低 F1 /其余类的 Macro-F1 贡献和翻转。
- `audit/query_classes.csv`、`query_relations.csv`、`relation_cohorts.csv`、`relation_summary.csv`：付费覆盖/错误关系及其与额外恢复/误伤事件的关联。
- `geometry/query_geometry.csv`、`geometry_summary.csv`、`pool_geometry.csv`：每次查询与各候选类的覆盖距离/标签组成。
- `geometry/focus.json`、`probe_rows.csv`、`probe_classes.csv`、`probe_summary.csv`：事后聚焦范围、可回查行号、边界拆分和条件近邻统计。
- `report/report.md`：结果表和判读范围，完成标记最后写入。

优先读低 F1 恢复表与 F1 损伤表，分清 ΔRecall、ΔPrecision、ΔTP、ΔFP；再读负翻转真实类与错分目的类，检查同一修复方向是否吸入了源正确样本。查询几何/付费错误关系/边界移动可形成后续干预假设，但不据此倒推全池安全、检测成功或新方法创新性。合成测试仅验证实现。
