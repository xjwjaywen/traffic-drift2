# 选样失败定位与下一轮创新探索

## 当前证据

用户于 2026-09-14 回传的五种子选样报告显示：完整预算 1000 下，BADGE 的崩溃类查询数/覆盖数为 44.4/11.2，随机为 23.2/7.6，风险加权分歧为 16.0/7.0。风险组专项原型推断正确率约 5%，普通分歧约 7.2%。风险组和打乱风险组的高置信错误数均值接近（52.0 vs 51.4）。这些是用户服务器报告的开发性结果，不是本地重新运行的实验。

这不支持直接扩大当前原型风险选样的训练规模，也不能证明任何原型方法或目标域局部结构都无效。诊断先回答错误产生在哪一环：原型、跨期迁移、候选评分/截断，还是多样性选择。

## 运行

原缓存和完成的 `kbs_acquisition_pilot_v1` 均须保留。在原 conda 环境中：

```bash
cd /data/xjw/traffic-drift2
conda activate traffic-ncde
env -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    git -c http.proxy= -c http.https://github.com.proxy= -c remote.origin.proxy= \
    pull --ff-only origin main
```

拉取成功后：

```bash
nohup bash Experiment/core_code/scripts/run_kbs_acquisition_diagnosis.sh run \
  > kbs-acquisition-diagnosis.log 2>&1 &
tail -f kbs-acquisition-diagnosis.log
```

完成标志为 `Results: .../report.md`。回传：

```bash
cat Experiment/core_code/outputs/kbs_acquisition_diagnosis_v1/report.md
cat Experiment/core_code/outputs/kbs_acquisition_diagnosis_v1/scout_summary.csv
```

入口支持 `preflight` 验证输入。环境变量：`PYTHON`、`GPU_ID`、`DEVICE`、`STUDY_DIR`、`CACHE_DIR`、`CHECKPOINT`、`PILOT_DIR`、`DIAG_OUTPUT_DIR`、`DIAG_SEEDS`。种子默认原 pilot 全部种子，可诊断其子集；改变选项时使用新的诊断输出目录。

读取原先完成的源缓存、pilot 选样和评价；**不生成新选样、不更新模型**。会重新计算各个五样本原型在参考/目标池上的预测，因此需要读取原缓存并做分批矩阵计算。CPU/CUDA 均可，默认 CUDA。没有新增依赖或数据下载。

## 四项诊断

1. **参考集几何是否可靠**：每个种子排除构建原型的 890 个参考行后，比较原分类头和原型准确率、支持类宏召回。避免使用原型自身训练行夸大其可靠性。没有修改原型数量/温度。
2. **到目标时期后是否失效**：报告目标全池、分歧池、高置信分歧池。每组分别拆解“两个都对、仅头对、仅原型对、两个都错”，不同条件池不能直接作为同一分布下的性能比较。
3. **选样流水线在哪里丢掉受害类**：重建三个方法的 capped candidate pool；分开报告共享随机探索、专项选择和随机补足。重建原型决策、分数、参考行及候选池必须匹配已保存结果；全选样统计须与原 `evaluation/by_seed.csv` 对上。
4. **是否有原型吸引了过多样本**：所有队列均输出每个类别的真实支持数、头预测数、原型预测数和召回率，保留零支持行。对照 `proto_predicted` 与 `true_support` 检查类别集中；这也是描述性诊断，不能自动推出因果。

每种子的 `cohorts.csv`、`per_class.csv` 和 `scout_pairs.json` 保存在 `seed_N/`。根目录 `summary.csv` 报均值/样本标准差，`report.md` 为中文表格。完成标记和文件 SHA256 支持恢复与复核；原 pilot/引擎文件和所有实验结果均不修改。

## 下一方向：从已确认的目标错误建立查询线索

原方案以源原型推断受害者身份，而选择出的候选中这一推断大多错误。值得检验的替代假设是：**先花预算确认少量目标域错分，再围绕已确认的受害者→吸收者关系分配剩余查询**。

这一轮只检查起点：按原 BADGE 的保存顺序取前 20% 查询（当前为 200 个），统计其中已确认的错分数量、不同错分对和崩溃类覆盖。BADGE 原实现保存 k-means++ 选择顺序，该前缀不依赖未查询标签。这里只核查线索数量，没有邻域检索、新查询或训练。

如果后续发展成方法，必须遵守：

- 初始 200 个标签计入总 1000 个预算；未来第二阶段只能读取已查询标签，不能提前查看全目标池的真实混淆矩阵。
- 候选局部区域由已查询目标特征和标签形成；保留探索预算，避免永远遗漏初始阶段没发现的类别。
- 单独比较定向错分关系与普通邻居检索、随机分配预算，以及打乱错分关系的控制，说明增益来自哪里。
- 控制重复近邻带来的冗余，以受害类别覆盖、修复收益和新崩溃为共同评价；发现更多错误本身不够。
- M12 已用于开发。冻结机制/预算分配后，另选未用于选方案与调参的数据验证。

二阶段查询、区域主动学习和反馈驱动的失效发现已有相关基础，不能把流程名称或少量组合视为创新证明。可对照的原始论文：

- [Region-Based Active Learning, AISTATS 2019](https://proceedings.mlr.press/v89/cortes19a.html)：研究分区域学习和标签分配。
- [Cost-aware Discovery of Contextual Failures using Bayesian Active Learning, CoRL 2025](https://proceedings.mlr.press/v305/parashar25a.html)：研究借助专家反馈发现多样化失效，任务为机器人系统，与本文不同。

如果这一方向的初始错误线索或局部可识别性不足，应转向有现成证据支持的“修复与类别保护之间的权衡”，而不是继续在当前原型温度和权重上追求一个最优数值。这里没有给出已验证的新算法，也没有自动通过/淘汰阈值。
