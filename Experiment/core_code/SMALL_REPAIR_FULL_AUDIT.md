# CARE 完整 M12 修复复评

复用已完成 `small_repair_diagnostic_v1` 的全部分类头，在排除原查询并集后的完整 M12 上评价。
默认 CPU、8 线程、18 个已有头；新增训练 0 次、新增查询 0 条，无需空闲显卡或新依赖。
这一轮检查修复收益和误伤是否在完整目标月份上保持，并为后续保护约束提供诊断。

## 服务器获取与前台运行

在学校服务器现有终端执行，保持 `traffic-ncde` 环境：

```bash
cd /data/xjw/traffic-drift2
conda activate traffic-ncde
env -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    git -c http.proxy= -c http.https://github.com.proxy= -c remote.origin.proxy= \
    fetch origin codex/small-repair-diagnosis &&
git restore --source=FETCH_HEAD --worktree -- \
    Experiment/core_code/scripts/small_repair_full_audit.py \
    Experiment/core_code/scripts/run_small_repair_full_audit.sh \
    Experiment/core_code/SMALL_REPAIR_FULL_AUDIT.md
```

只安装这三个新文件。已完成实验登记了原脚本、launcher、引擎、加载代码和缓存的哈希，
所以不要用这一轮更新原 `small_repair_diagnostic.py`、`run_small_repair_diagnostic.sh` 或旧引擎。
运行以下命令会先自动校验输入，再开始复评；日志直接显示在当前终端：

```bash
bash Experiment/core_code/scripts/run_small_repair_full_audit.sh run
```

开始时会读盘校验约 6.5 GB 缓存，因此可能暂时没有新日志。然后先逐个复核原抽样预测/指标，
显示 `Original sampled predictions and metrics reconciled` 后开始完整推理。
完整 M12 比此前 10 万条评价多，实际耗时待服务器测量；本轮不加载历史参考张量，仅校验它的文件哈希。
推理按批索引特征，避免复制整个排除集的特征。服务器原有内存足以容纳这一规模的缓存。

出现以下标记才表示报告全部生成、完成文件已写入：

```text
SMALL_REPAIR_FULL_AUDIT_COMPLETE: .../small_repair_full_audit_v1/report.md
```

随后显示报告：

```bash
cat Experiment/core_code/outputs/small_repair_full_audit_v1/report.md
```

可选的 `preflight` 仅校验文件哈希、协议和全部来源阶段完成状态，不加载张量或推理：

```bash
bash Experiment/core_code/scripts/run_small_repair_full_audit.sh preflight
```

中断后可重跑相同命令；未完成的复评会重新推理，已完成复评经哈希核验后跳过张量加载和推理。
来源实验、结果或代码不一致会报错。若原预测不能复现，先核对环境、原 CPU 线程数和推理 batch，
不要绕过检查。完成文件损坏会报错，不自动覆盖。改变配置时使用新的输出目录。

## 评价口径与输出

全部种子、random/Margin 查询的**共同并集**排除后，static 和全部分类头使用相同的完整目标行。
查询集合、训练权重和种子全部继承原实验；没有新的选样或训练参数。崩溃/稳定组只用于事后评价。
原抽样的行号、真值、static 预测、各头预测、逐种子/逐类指标必须对账。完整推理的批次边界不同，
若在原抽样行上产生预测差异，差异数会另外记录；不会悄悄替换成原抽样预测。

独立输出位于 `Experiment/core_code/outputs/small_repair_full_audit_v1/`：

- `report.md`：总体和配对结果、CARE 损伤最大的非崩溃类及主要负翻转去向。
- `by_seed.csv`、`summary.csv`、`paired_*.csv`：完整评价指标及原训练统计；均值和样本标准差按种子计算。
- `per_class.csv`、`per_class_summary.csv`：每类支持、精度、召回、F1、源正确/源错误数量、正负翻转及错→另一个错。
- `transitions.csv`、`transitions_summary.csv`：真实类/源预测/修复预测改变的三元组、类型和计数。
  某种子未出现的边在汇总中计为零；每条边的实际出现种子数同时给出。
- `query_coverage.csv`、`query_coverage_summary.csv`：已查询真值的类别覆盖及其中源模型原本正确的候选保护锚点数量。
- `predictions.npz`：完整共同排除集的行号、真值、static 和全部修复预测，可独立复算。
- `resolved_config.json`、`provenance.json`、最后写入的 `complete.json`：来源、代码、环境与文件哈希。

负翻转率的分母是源模型正确样本；正翻转率的分母是源模型错误样本。
逐类空分母记为 NA/空值，均值只使用有定义的种子，并记录 `*_valid_seeds`。
Macro-F1 延续全部 178 类口径，包含零支持类的零值；低支持类仍需要结合支持数解读。
均值/SD 是共享月份与源模型的训练随机性描述，不能作为独立环境的置信区间。

损伤、错分去向和候选锚点覆盖是诊断证据。后续“保留 CARE，加目标保护锚点和 margin 约束”
仍需单独训练与验证；本轮不证明该约束有效。M12 已是方法开发数据，不能当作独立确认集。

## 参数与本地校验

只继承 `SMALL_FULL_SOURCE_DIR`、`SMALL_FULL_CACHE_DIR`、`SMALL_FULL_CHECKPOINT`、
`SMALL_FULL_OUTPUT_DIR`、`SMALL_FULL_THREADS` 和 `PYTHON`。旧训练变量不影响本轮。
线程数必须与原协议相同（默认 8）；默认推理 batch 4096，可显式传 `--inference-batch-size`。
CUDA 在 shell 和 Python 入口均隐藏。

```bash
python -m unittest discover -s Experiment/core_code/scripts/tests -p 'test_small_repair_full_audit.py'
bash Experiment/core_code/scripts/tests/test_run_small_repair_full_audit.sh
```

合成测试验证完成门控、禁止训练/选样/参考张量加载、共同查询排除、手工 F1 和逐类翻转、
三元组计数、缺失边的零补齐、空分母、旧结果对账、输入哈希/修改时间保留及完成后续跑。
合成分数不是论文实验结果，真实结果需上述学校服务器运行产生。
