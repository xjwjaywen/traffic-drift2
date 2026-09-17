# 修复预演驱动的选样：第一轮开发性预实验

本轮只回答：在同样的标签预算和修复引擎下，根据拟执行更新的影响选样，
是否减少最终修复损伤；已确认的错分方向是否带来额外收益。
这是 M12 上的新开发实验，不是独立验证，也不是已证实的新颖性或因果机制。
不实现动态预算、bandit、oracle、全模型微调或新数据集训练。

## 服务器运行

在已有 `traffic-ncde` 环境中，从仓库根目录运行：

```bash
cd /data/xjw/traffic-drift2
conda activate traffic-ncde
git pull --ff-only origin main
bash Experiment/core_code/scripts/run_kbs_repair_aware.sh preflight
```

预检查出现 `"status": "ready"` 后：

```bash
nohup bash Experiment/core_code/scripts/run_kbs_repair_aware.sh run > kbs-repair-aware.log 2>&1 &
tail -f kbs-repair-aware.log
```

`Ctrl+C` 退出 `tail` 不会停止 nohup 后台任务。最后出现 `Results: .../evaluation/report.md`
才表示选样、四组训练和评价均完成。运行中仅输出训练进度，不提前展示用于调方法的测试指标。
中断后重跑同一个 `run` 命令会验证文件哈希并跳过已完成阶段；不支持同一输出目录多进程写入。
若发现文件损坏或参数/代码变化会报错，不静默复用或覆盖完成结果。

```bash
cat Experiment/core_code/outputs/kbs_repair_aware_v1/evaluation/report.md
cat Experiment/core_code/outputs/kbs_repair_aware_v1/evaluation/paired_by_seed.csv
cat Experiment/core_code/outputs/kbs_repair_aware_v1/evaluation/acquisition.csv
```

默认输入：

- `outputs/kbs_supplement_v1/study_manifest.json`
- `outputs/kbs_supplement_v1/cache/{manifest.json,head.pt,reference.pt,target.pt}`
- `outputs/kbs_supplement_v1/selections/badge/seed_{0..4}/` 的完整选样文件
- `outputs/tls22_cnn/best_model.pt`

读取已存在的特征缓存和已保存的 BADGE 顺序，不重新提取特征或运行全池 BADGE。
预检查和每个阶段都校验缓存哈希，大文件检查期间日志可能暂时没有新行。
输入目录只读；输出单独写入 `outputs/kbs_repair_aware_v1`。
本轮不需要以前 prototype、pool-control 或 relation-audit 分支的输出。

可设置 `GPU_ID`、`STUDY_DIR`、`CACHE_DIR`、`CHECKPOINT`、`PYTHON`、`DEVICE`。
新输出和种子只读专用变量 `REPAIR_OUTPUT_DIR`、`REPAIR_SEEDS`，不继承旧 `OUTPUT_DIR`、
`SEEDS`、`STEPS` 或 `PILOT_SEEDS`。训练步数、batch、学习率、权重衰减和目标 CE 权重
继承受控补充实验的 manifest；`--batch-size` 只控制全池推断。

只有做独立调试时才改种子；例如下例会建立另一个输出，不能拿来筛选最好种子：

```bash
REPAIR_SEEDS=0 REPAIR_OUTPUT_DIR=outputs/kbs_repair_aware_debug \
  bash Experiment/core_code/scripts/run_kbs_repair_aware.sh run
```

## 固定协议

生产输入是 M3 源模型、M4 参考缓存、M12 目标缓存，178 类；目标预算 B=1000，
历史参考每类 5 个，共 890 个，按原受控实验种子抽样。历史参考成本单列，各组相同。
脚本为合成测试允许预算为 5 的倍数，实际预算直接读取原 study，不新设可调预算。
默认种子 0,1,2,3,4，共 **5 次共享预演＋20 次最终修复**。

1. 每种子取原已保存 BADGE 顺序的前 800 个样本作为共同侦察标签。
   原 BADGE 实现的 PCA 和顺序 k-means++ 在确定随机种子后按同一顺序选择；
   使用既有 1000 点序列的前缀，避免另外重算近似实现。
2. 仅用这 800 个目标标签，从原源头开始做一次临时修复。
   CE 权重、KD 权重、训练步数均与最终修复一致；典型为 1380 步。
   用完整临时修复而非任意 5 步近似，先检验这一具体拟更新的选样价值。
3. 未查询池中，定义 `source_argmax != probe_argmax` 为翻转候选。
   这是**预测变化**，不是已知负翻转；查询之前不知道旧/新预测谁正确。
   本版不把任意 margin 变化当风险，也不基于旧原型或目标真实类别截断。
4. 由侦察标签构造 `count[v,a]`：真实类为 v、源预测为 a 的错分次数，排除正确预测。
   当某候选的源预测为 a、临时预测为 v 时，其关系权重为 `1 + count[v,a]`。
   这检验反向边界移动可能损伤真实 a 的假设，并不假定该候选真的属于 a。
   未确认关系权重仍为 1，不硬门控，所有翻转候选都保留被选择机会。

剩余 200 个标签的四组：

| 方法 | 后 200 个查询 | 与前一关键对照的差异 |
|---|---|---|
| badge | 原保存序列的后 200 个 | 原源模型 BADGE 1000 基线，不用预演重算 |
| random | 剩余全池均匀无放回 | 检查普通补采 |
| flip_uniform | 翻转候选内均匀无放回 | 仅加入拟更新的影响区域 |
| flip_relation | 同一翻转池内按上述关系权重无放回 | 仅加入已查询标签构造的错分方向 |

C/D 使用相同随机种子和抽样实现；当关系权重完全相同，两组选择完全相同。
翻转池不足 200 时取全部候选，再从未选剩余全池均匀补齐；记录回填数。
C/D 后 200 行按行号排序；若查询集合相同，训练顺序也相同。
整个翻转池能放进预算时不做无意义的加权排列，确保两组回填和训练也一致。
所有查询唯一，四组均保留相同前 800 个标签。选到不符合保护假设的样本也计费、也保留训练。
这是固定 80/20 的探索，不能由单一比例失败推断全部保护选样思路不可行。

## 最终修复与标签边界

全部请求种子的查询先冻结，随后另一个进程训练。每个最终修复从**同一个源分类头**重新开始，
不从预演头继续训练；使用该组全部 1000 个查询的普通目标 CE＋参考 KD。
关闭参考 CE；KD 权重 0.5、温度 2，与原 `badge_kd` 设置相同。
不引入额外保护损失，也不按查询标签是否正确重加权，避免同时改变选样和训练。
四组目标采样位置流、参考行及采样流、优化器步数完全配对。

旧缓存 `.pt` 同时存有特征和标签，因此载入时标签物理上会进入内存；这不是进程级信息隔离。
代码的标签访问契约是：

- `select` 仅索引共同前缀的 800 个标签；纯选样函数不接受完整标签或真实崩溃类组。
- `train` 仅索引该组冻结后的 1000 个标签。
- `evaluate` 必须等所有请求种子、所有四组完成才解释完整目标标签。

测试使用一个只允许指定查询索引的标签对象，禁止完整转换、遍历或其他索引；
另外替换所有未侦察样本的标签，验证选样保持不变。
离线评价标签与方法实际可用标签严格区分；方法若未来使用审计标签，必须计入 B。

## 评价和边界

每种子四组均排除四组查询的并集，并在这个**共同评估集**上重新计算 static。
不拼接旧主表指标，不必排除没有参与本轮方法的旧探索查询。
真实崩溃组、稳定组只用于事后评价；保留原 recall<0.1 定义和所有有支持的稀有类。

输出整体/崩溃类/稳定类/全部非崩溃类 F1、新崩溃数、原崩溃残留、正负翻转、逐类支持数和损伤。
负翻转率的分母是同组评估样本中源模型原本正确的数量。
额外报告后 200 个标签中实际属于“源正确→预演错误”的数量，以及正向纠错数量、回填数。
这些选样诊断不能替代最终修复指标。

配对报告：random−badge、flip_uniform−badge、flip_uniform−random、
flip_relation−flip_uniform、flip_relation−badge。均值和样本标准差；没有自动显著性检验、
线性插值前沿、最佳种子挑选或通过/失败阈值。
优先判断 C 是否优于 A/B，以及 D 是否比 C 多提供价值；若降低损伤却牺牲恢复，报告取舍。

5 个种子共用一个源模型和目标月份，不是 5 个独立环境。
M12 已用于方法开发，M11 已在原稿中使用；本试验不重新声明它们为未见数据。
有效性与相对 RoSE 等方法的新颖性需要分别论证。本轮是可行性探索，尚不构成完整竞争方法比较。

预演、推断、最终训练时间单独保存在 trace 中；预演在组间共享，用于科学配对，
BADGE/random 实际部署不需要预演，不能隐藏这种成本差别。
旧缓存特征提取和旧 BADGE 选样耗时未计入，不声称端到端成本优势。

## 校验

```bash
python -m unittest discover -s Experiment/core_code/scripts/tests -p 'test_kbs_repair_aware.py'
bash Experiment/core_code/scripts/tests/test_run_kbs_repair_aware.sh
```

这些是 CPU 合成输入校验，不产生论文实验结果。真实 CESNET/CUDA 结果由服务器运行产生。
