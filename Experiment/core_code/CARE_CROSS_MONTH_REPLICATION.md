# CARE M11 跨月复验

M12 上五个新查询种子的复验后，本轮使用预先指定的 **M11**，检验特征覆盖对相同候选池、相同预测类别配额的类别均衡随机是否仍有额外收益。沿用普通线性 CARE、源 M3 模型、M4 参考和已完成 M12 的全部拟合/候选池参数，不按 M11 结果调参或选择种子。

## 固定协议

- 五个查询种子 `6174,6175,6176,6177,6178`，各交叉原三个训练种子 `[0,1,2]`。
- `mixed_balanced` 和 `mixed_coverage` 均为 800 Margin＋200 补充；同种子内候选 reservoir 与预测类别配额相同。
- 纯 Margin 1000 标签作为基线，在 M11 重新训练三个头。混合两臂共 30 个头，合计 **33 次新的 CPU 拟合**；不复用 M12 已修复分类头。
- 每个头都从原源分类头初始化，沿用 1380 AdamW 步、batch 64、lr 0.001、weight decay 0.0001、每参考类 5 行、KD 权重 0.5、温度 2 和原 CE 权重。
- 全部 M11 查询先冻结，训练只读该头付费目标标签；另进程在全部头完成后评价完整 M11 查询并集排除集。M12 的行号不进入 M11 的查询或评价掩码。
- 主统计单位是五个查询种子：先平均同一查询种子内三个配对训练种子的差异，再统计均值、样本 SD 与正负收益数量。不能当作 15 个独立查询重复。Margin 的 ± 仅为三个训练种子的 SD。

预算是每次选样、每臂的 1000 个标签；重复实验的标签并集会超过 1000。现有缓存将标签和特征打包，标签隔离是程序 API 访问约定，不是文件或进程安全隔离。

## M11 特征缓存

当前 ToDesk 清单在仓库 outputs 内只显示 M4→M12 缓存，扫描未截断，年度原始 H5 存在；这不保证原始数据能成功加载 M11。新入口 `prepare` 会：

1. 校验完成的 M12 复验及完整依赖链、旧缓存、源 checkpoint 和原数值文件哈希。
2. 在独立 `outputs/care_m11_cache_v1/` 按字节复制已校验的 M4 `reference.pt` 与源 `head.pt`。
3. 使用源缓存记录的数据配置、原加载器、数据种子 0，**只提取 M11** 的源模型特征/logits。骨干冻结，CPU 8 线程，CUDA 隐藏；不重新提取 M4、不训练骨干。
4. 每 100 batch 显示进度；校验类别数、维度、有限值、原始输入流与所有张量哈希，最后写缓存 manifest。

该阶段会比小分类头训练慢，耗时需依据服务器实际进度判断。缓存准备会保存原始标签，但不会依据这些标签选样、拟合、打分或调参。若 M11 不可加载、类别数不符或原始数据在提取过程中改变，立即失败，不改用 M12，不继续训练。不会主动安装依赖、配置下载或改学校服务器环境。

重复同一命令可继续：完成缓存只读哈希，不再提取；完成分类头只校验，不重训。中断的特征提取会重新提取 M11，已完成旧缓存不变。协议或代码改变必须使用新缓存/结果目录。

## 服务器当前终端前台运行

前置环境为已有 `traffic-ncde`，前置结果为完整 `care_query_seed_robustness_v1`。只恢复新文件，不修改旧脚本或切换服务器分支：

```bash
cd /data/xjw/traffic-drift2 &&
conda activate traffic-ncde &&
env -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    git -c http.proxy= -c http.https://github.com.proxy= \
        -c remote.origin.proxy= fetch origin codex/small-repair-diagnosis &&
git restore --source=FETCH_HEAD --worktree -- \
    Experiment/core_code/scripts/care_cross_month_replication.py \
    Experiment/core_code/scripts/run_care_cross_month_replication.sh \
    Experiment/core_code/CARE_CROSS_MONTH_REPLICATION.md &&
bash Experiment/core_code/scripts/run_care_cross_month_replication.sh run
```

`run` 串行执行四个前台进程：`prepare → select → train → evaluate`，失败立即停止。不要加 `nohup` 或 `&`。运行时保持当前终端，等待：

```text
M11_CACHE_COMPLETE: ...
CARE_M11_REPLICATION_COMPLETE: .../care_m11_replication_v1/evaluation/report.md
```

只有第二个标记代表修复比较与评价已完成。随后显示报告：

```bash
sed -n '1,40p' Experiment/core_code/outputs/care_m11_replication_v1/evaluation/report.md
```

单独显示计划或校验已有来源：

```bash
bash Experiment/core_code/scripts/run_care_cross_month_replication.sh plan
bash Experiment/core_code/scripts/run_care_cross_month_replication.sh preflight
```

`preflight` 在尚无 M11 缓存时仅说明待提取，不能报月份数据已验证；真正的加载/特征校验由 `prepare` 完成。也可分别运行 `prepare`、`select`、`train`、`evaluate`。

路径环境变量为 `CARE_M11_SOURCE_DIR`、`CARE_M11_AUDIT_DIR`、`CARE_M11_PROTECTION_DIR`、`CARE_M11_ABLATION_DIR`、`CARE_M11_REPLICATION_DIR`、`CARE_M11_SOURCE_CACHE_DIR`、`CARE_M11_CHECKPOINT`、`CARE_M11_CACHE_DIR`、`CARE_M11_OUTPUT_DIR`。`CARE_M11_DATA_DIR` 可指定现有原始 TLS 数据目录，默认沿用源缓存记录的目录。`CARE_M11_THREADS` 默认 8，必须与已完成来源一致。入口不继承旧 `SEEDS`、`QUERY_SEEDS`、`STEPS`、`DEVICE`、`OUTPUT_DIR` 或 `CARE_QUERY_*` 参数；没有可根据 M11 结果调查询种子/预算/训练参数的选项。

## 报告与判断

结果独立写入 `outputs/care_m11_replication_v1/`。`manifest.json` 绑定 M11 缓存、代码和已完成 M12 来源；`selection/` 保存全部查询/池/配额/共同评价行；`runs/` 保存 33 个分类头、付费标签、参考行和抽样流；`evaluation/` 保存报告、完整预测、逐类结果、付费覆盖、训练重复表、逐查询种子表及分层主汇总。完成标记最后写入。

主对照 `coverage−balanced` 同时报告总体 Macro-F1 和负翻转率差。原 M12 问题/稳定类是**迁移分组**，不直接称 M11 崩溃/稳定类。另按预定规则，在全部拟合后从共同 M11 评价集定义“有支持且源模型 F1<0.1”的低 F1 组，写入 `evaluation_groups.json` 并报告其改善；它只用于事后描述，不参与查询、训练或挑参数，也不是 CARE 检测成功的证据。

不将 M11、M12 的绝对 F1 直接作差。看两个月各自 `coverage−balanced` 的方向、幅度、离散程度与误伤：M11 优势消失则收缩特征覆盖的跨月主张；同方向则值得继续其他月份和更强机制比较。一个新月份仍不能证明普遍有效或方法新颖性。合成测试只验证代码和协议。
