# CARE 小规模修复诊断

这一轮先定位可继续改进的环节：随机/不确定性选样的差别、参考保护的恢复与损伤取舍、
固定特征上增加分类头容量是否有帮助。它是 M12 上的方法开发预实验，不是论文主表复现或新方法证明。
默认 **CPU、8 线程、3 个种子、18 次小模型拟合**；无需等待空闲显卡。

## 获取脚本

在学校服务器现有终端中执行以下命令。仅从新分支取五个新增文件，服务器当前分支不需要切换；
旧 `kbs_supplement.py`、模型、缓存和已有实验结果保持原样。重复取脚本会更新这五个文件。
这里保留了此前可用的单次 Git 代理绕过，不修改全局代理设置。

```bash
cd /data/xjw/traffic-drift2
conda activate traffic-ncde
env -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    git -c http.proxy= -c http.https://github.com.proxy= -c remote.origin.proxy= \
    fetch origin codex/small-repair-diagnosis &&
git restore --source=FETCH_HEAD --worktree -- \
    Experiment/core_code/scripts/small_repair_diagnostic.py \
    Experiment/core_code/scripts/run_small_repair_diagnostic.sh \
    Experiment/core_code/scripts/tests/test_small_repair_diagnostic.py \
    Experiment/core_code/scripts/tests/test_run_small_repair_diagnostic.sh \
    Experiment/core_code/SMALL_REPAIR_DIAGNOSTIC.md
```

脚本使用已有 NumPy/PyTorch 和标准库，没有新增包安装。

## 预检查与运行

以下命令都从 `/data/xjw/traffic-drift2` 仓库根目录运行：

```bash
bash Experiment/core_code/scripts/run_small_repair_diagnostic.sh plan
bash Experiment/core_code/scripts/run_small_repair_diagnostic.sh preflight
```

`preflight` 校验三份缓存、源模型 SHA256、加载代码和月份，实际载入张量、检查维度/有限值，
并检查历史参考能否提供每类五条。读取约 6.5 GB 缓存期间可能暂时没有新日志。
数值引擎固定为仓库 main 中已测试的 `kbs_supplement.py`（SHA256 `8f391a4a…1189c`）；
若服务器有本地改动会明确报错，请先比较改动，不覆盖旧引擎来绕过检查。
出现 `"status": "ready"` 后启动：

```bash
nohup bash Experiment/core_code/scripts/run_small_repair_diagnostic.sh run \
    > small-repair-diagnostic.log 2>&1 &
tail -f small-repair-diagnostic.log
```

`Ctrl+C` 只退出 `tail`。看到 `SMALL_REPAIR_COMPLETE: .../evaluation/report.md` 才表示整个诊断完成。
若出现 `ERROR` 或 traceback，请先读错误；末尾的 shell 提示符本身不代表运行成功。
没有测量这台服务器上的实际耗时，不给出运行时长保证。

中断后用同一 `run` 命令续跑。已完成的选样、拟合、报告会核对哈希后跳过；
不支持同一输出目录并发写入。改变代码、环境版本或协议参数需换新的输出目录。
完成文件损坏会报错，不自动覆盖已有完成结果。

```bash
cat Experiment/core_code/outputs/small_repair_diagnostic_v1/evaluation/report.md
```

## 固定对照

默认预算：每组 **1000 个目标标签＋890 个历史参考标签**；种子 `0,1,2`。
仅使用已有 `kbs_supplement_v1/cache` 和 `tls22_cnn/best_model.pt`，不要求以前 pilot 的结果，
不重新提取特征、不重算全池 BADGE、不训练骨干。

| 对照维度 | 设置 | 用途 |
|---|---|---|
| 选样 | random、margin | 比较均匀采样与源模型 top-two logit margin；只用预测选样 |
| 修复 | linear_ce | 同一源线性头，只有目标 CE |
| 修复 | linear_care | 同一源线性头，目标 CE＋参考 CE＋KD |
| 修复 | residual_mlp_care | 源线性头＋128 隐单元 ReLU 残差，两分支都训练，损失同 linear_care |

残差输出层初始权重/偏置为零，初始化预测严格等于源线性头。
默认线性头 45,746 个参数，残差头 101,604 个参数；相同优化步数不意味着相同计算量。
它只测试一个固定的头部容量设置。失败不能证明特征完全不可修复，也不能证明必须改骨干。

三组修复都直接复用原受控实验的 `fit_controlled` 数值引擎：1380 步、batch 64、
AdamW lr=0.001、weight decay=0.0001。CARE 组 KD 权重 0.5、温度 2；
目标 CE 权重 `2000/2890`、参考 CE 权重 `890/2890`，在线性 CE 组也保留同一目标 CE 权重。
这一对照一次加入整个参考 CE＋KD 保护，不能把差值单独归因于 KD。
同一种子/选样下，查询、历史参考、目标/参考采样位置流及优化步数完全相同。
不同选样也使用相同的训练位置随机流。查询都按行号排序，因此不声称数值复现旧论文/旧 study 表格。
Margin 对种子不敏感，三种子共享同一 Margin 查询集合。

## 查询与评估边界

`run` 按三个独立 Python 进程执行 `select → train → evaluate`，前一阶段失败会停止。

- `select` 丢弃缓存中捆绑的目标标签，只向选样 API 传入 logits，先冻结全部种子/选样的查询。
- `train` 只索引已冻结的目标查询标签，历史参考按类均衡采样。
- `evaluate` 要求全部 18 次拟合完成并通过哈希核验，再使用完整目标标签做离线评价。

旧 `.pt` 文件捆绑了标签，所以载入时标签在物理上会进入内存；上述是代码访问契约，
不是进程或存储层的信息隔离。崩溃/稳定类名单只用于事后评价，不用于选样或训练。

共同评估集默认从排除**全部种子、全部选样查询并集**的剩余 M12 样本中，
以独立固定种子 1729 均匀抽取 100,000 条（抽样不读标签）。每个模型及 static 都用这些相同行。
不做类别均衡测试采样；逐类输出样本支持数，支持少于 20 条的类会标记。
总体/指定组 Macro-F1 仍包含零支持类的零值，不能与全 M12 主表直接比较；
小样本的稀有类 F1/召回和新增崩溃数只作诊断线索。

若需要完整评估，另建输出目录运行：

```bash
SMALL_REPAIR_OUTPUT_DIR=outputs/small_repair_diagnostic_fullM12_v1 \
    bash Experiment/core_code/scripts/run_small_repair_diagnostic.sh run --eval-size 0
```

不要用测试指标挑最好种子或扫描超参数。M12 已用于方法开发；这些种子共享月份和源模型，
不构成三个独立环境。训练准确率/CE只能提示拟合状况，不能独立证明瓶颈或因果机制。

## 输出与参数

默认输出 `Experiment/core_code/outputs/small_repair_diagnostic_v1/`：

- `diagnostic_manifest.json`：缓存、源模型、代码、环境版本和完整实验协议。
- `selection/queries.json`、`evaluation_rows.npy`：全部冻结查询和共同评估行。
- `runs/<selector>/seed_<n>/<engine>/{head.pt,trace.json}`：小模型、已付费标签/历史参考、采样流、参数数和训练耗时。
- `evaluation/report.md`：中文主结果和配对诊断表。
- `evaluation/by_seed.csv`、`summary.csv`、`paired_*.csv`：整体/崩溃/稳定/全部非崩溃 F1、正负翻转、训练准确率和耗时。
- `evaluation/per_class.csv`、`predictions.npz`：全部类别支持和损伤，以及共同评估行/标签/预测，便于复核。
- 各阶段 `complete.json`：登记结果文件哈希，完成标记最后写入。

输入读取/哈希、选样、拟合、评价的阶段总耗时见 `small-repair-diagnostic.log`；
每次拟合和评价耗时另外写入 trace/CSV。旧特征提取成本不计入。

只继承专用变量 `SMALL_REPAIR_CACHE_DIR`、`SMALL_REPAIR_CHECKPOINT`、`SMALL_REPAIR_OUTPUT_DIR`、
`SMALL_REPAIR_SEEDS`、`SMALL_REPAIR_THREADS` 和 `PYTHON`。旧 `SEEDS`、`OUTPUT_DIR`、`DEVICE`、
`GPU_ID` 等不影响本轮。CUDA 在 shell 和 Python 入口中均隐藏。
其他参数用 `--budget`、`--steps`、`--eval-size` 等显式传入，修改时使用新的输出目录。
默认会要求 178 类、256 维、M4/M12；合成测试用显式参数指定小尺寸。

## 本地校验

```bash
python -m unittest discover -s Experiment/core_code/scripts/tests -p 'test_small_repair_diagnostic.py'
bash Experiment/core_code/scripts/tests/test_run_small_repair_diagnostic.sh
```

测试覆盖标签访问边界、残差初始化、匹配采样流、手工重算 F1/翻转、共同查询排除、
完成门控、文件完整性、断点续跑、独立 CLI 进程及 launcher 失败传播。
合成测试不产生论文实验结论；真实分数由学校服务器运行产生。
