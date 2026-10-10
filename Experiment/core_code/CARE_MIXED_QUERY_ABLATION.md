# CARE 混合查询消融

本轮检查上一轮 mixed 普通 CARE 的收益来自随机替换、候选池、预测类别均衡还是特征几何。只使用普通线性 CARE：目标 CE＋历史参考 CE＋KD。没有逐步硬约束或插值控制。原诊断、完整复评、目标保护试验及其文件哈希均保持不变。

## 固定协议

默认 M4 参考、M12 目标、178 类、256 维缓存特征。每臂目标标签预算 1000，训练种子沿用 `[0, 1, 2]`。四个 mixed 臂共享完全相同的 800 个最小原始 logit margin 查询，另选 200 个不重叠补充查询。

| 查询臂 | 200 个补充查询 | 拟合来源 |
|---|---|---|
| `margin` | 原 1000 个 Margin 查询，无独立补充 | 复用上一轮已核验的普通 CARE 头 |
| `mixed_random` | 从共享 800 行之外的全部目标行均匀随机选择 | 新拟合 |
| `mixed_pool_random` | 从特征覆盖的相同高置信候选池均匀随机选择 | 新拟合 |
| `mixed_balanced` | 在同一候选池内按源预测类别随机选择，配额与特征覆盖完全相同 | 新拟合 |
| `mixed_coverage` | 原预测类别分层、特征中心首点及最远点覆盖 | 复用上一轮查询和普通 CARE 头 |

候选池完全沿用上一轮：排除 800 个修复查询后，每个源预测类别保留 raw margin 上半区；超过 256 行时使用固定随机 reservoir 截取。预测类按固定随机顺序轮转分配 200 个名额，候选不足的类不强行补齐。特征覆盖与类别均衡的类配额完全一致，后者只把特征几何选择替换为层内随机选择。

查询不使用目标真值。候选池/配额继承上一轮协议，选择种子为 6173；新增全池随机、候选池随机和层内随机分别使用固定派生种子。每臂查询集在三个训练种子之间相同，因此三个种子只重复参考行和优化过程，不代表三个独立选样重复。

目标/参考 CE 权重、KD、初始化、AdamW、1380 步、batch 64 沿用已完成诊断。每个训练种子的历史参考行、目标和参考抽样**索引**流与原 CARE 配对；各臂目标行内容随查询策略变化。默认只新增 **9 次 CPU 拟合**，另复用 6 个分类头。复用头逐字节复制，并核验原查询、付费标签、参考行、抽样流和来源哈希。

## 服务器前台运行

前置条件：`small_repair_diagnostic_v1`、`small_repair_full_audit_v1`、刚完成的 `care_target_protection_v1` 及缓存均在原目录。环境使用 `traffic-ncde`。无需安装依赖，CPU 默认 8 线程，隐藏 CUDA。

在 `/data/xjw/traffic-drift2` 当前终端执行下面整段命令。代理覆盖只作用于本次 fetch；只取新文件，保留服务器已有数值引擎和完成结果。

```bash
cd /data/xjw/traffic-drift2 &&
conda activate traffic-ncde &&
env -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    git -c http.proxy= -c http.https://github.com.proxy= \
        -c remote.origin.proxy= fetch origin codex/small-repair-diagnosis &&
git restore --source=FETCH_HEAD --worktree -- \
    Experiment/core_code/scripts/care_mixed_query_ablation.py \
    Experiment/core_code/scripts/run_care_mixed_query_ablation.sh \
    Experiment/core_code/CARE_MIXED_QUERY_ABLATION.md &&
bash Experiment/core_code/scripts/run_care_mixed_query_ablation.sh run
```

`run` 串行启动三个前台进程：`select` 冻结全部查询、`train` 拟合或复用全部分类头、`evaluate` 独立评价。终端会显示各臂/种子、训练步数及评价进度。任何阶段失败，后续阶段不运行。重复同一命令可以校验后继续，完成项不重训。

可单独检查输入与计划：

```bash
bash Experiment/core_code/scripts/run_care_mixed_query_ablation.sh preflight
bash Experiment/core_code/scripts/run_care_mixed_query_ablation.sh plan
```

完成后应出现：

```text
CARE_MIXED_QUERY_ABLATION_COMPLETE: .../care_mixed_query_ablation_v1/evaluation/report.md
```

显示主结果、配对消融和补充查询覆盖：

```bash
sed -n '1,45p' Experiment/core_code/outputs/care_mixed_query_ablation_v1/evaluation/report.md
```

如需改输出位置，用 `CARE_MIXED_OUTPUT_DIR`；其他路径可用 `CARE_MIXED_SOURCE_DIR`、`CARE_MIXED_AUDIT_DIR`、`CARE_MIXED_PROTECTION_DIR`、`CARE_MIXED_CACHE_DIR`、`CARE_MIXED_CHECKPOINT` 指定。线程数必须与已完成诊断一致。`PYTHON` 可指定环境中的 Python 可执行文件。新入口不继承旧实验的 `SEEDS`、`STEPS`、`OUTPUT_DIR` 或 `CARE_PROTECT_*` 变量。

## 评价与文件

独立目录 `outputs/care_mixed_query_ablation_v1/`：

- `manifest.json`：新协议、原协议、已完成来源、代码和运行环境。
- `selection/`：查询行、修复/补充划分、候选池、配额、共同评价行及完成标记。
- `runs/<selector>/seed_<n>/care/`：分类头、付费查询标签、参考行、拟合/复用记录及完成标记。
- `evaluation/`：主报告、逐种子和汇总指标、配对差异、逐类精度/召回/F1/翻转、付费类别覆盖、逐行预测及完成标记。

先核验所有分类头和新旧来源哈希，才读取未查询目标真值进行评价。默认评价完整 M12，排除原诊断所有 random/Margin 查询、上一轮全部查询和本轮全部查询的**并集**。每臂严格使用同一批评价行。复用基线重新推理，不能直接与上一轮不同评价掩码的报告数字相减。

所有指标沿用原诊断定义；负翻转率是源正确评价行中修复后出错的比例，正翻转率是源错误评价行中修复后正确的比例。报告的付费源正确覆盖按缓存源 logits 判定，是事后诊断，不参与选择。它与上一轮 float64/float32 共同判定且剔除极弱间隔的约束锚点定义不同，不能直接当成同一计数。

重点对照：

1. `mixed_pool_random − mixed_random`：候选池构造的整体效应，包含置信度筛选和每类候选上限，不能归因于纯置信度。
2. `mixed_balanced − mixed_pool_random`：在同一池内固定预测类别配额的效应。
3. `mixed_coverage − mixed_balanced`：在相同候选池和配额下的特征几何效应。
4. 四个 mixed 臂各自减 `margin`：相同 1000 标签预算下的净收益。

这轮仍是同一月份上的开发消融；种子标准差不是置信区间。固定随机选样的一次比较不能证明选样稳健性，也不能证明新颖性或跨月份收益。若几何对照不优于类别均衡，应如实收缩方法主张；合成测试只验证程序正确性。
