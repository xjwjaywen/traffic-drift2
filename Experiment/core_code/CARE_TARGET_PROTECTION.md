# CARE 目标保护机制试验

我们已有可靠 CARE 基线和完整 M12 误伤诊断，但没有已验证的新方法。
这轮检验：固定 1000 个目标标签预算，加入保护样本覆盖和逐步间隔约束，能否在恢复相近时减少误伤。
保留骨干、缓存、CARE 基础损失和历史参考预算；默认 CPU、8 线程、原种子 0/1/2。
800/200 分配、候选上限 256 和间隔参数先固定，不用 M12 测试结果搜索最优值。

## 服务器获取和前台运行

来源必须已经完成 `small_repair_diagnostic_v1` 和 `small_repair_full_audit_v1`。
只取三个新文件，保留原诊断、复评和引擎文件及结果哈希：

```bash
cd /data/xjw/traffic-drift2 &&
conda activate traffic-ncde &&
env -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    git -c http.proxy= -c http.https://github.com.proxy= -c remote.origin.proxy= \
    fetch origin codex/small-repair-diagnosis &&
git restore --source=FETCH_HEAD --worktree -- \
    Experiment/core_code/scripts/care_target_protection.py \
    Experiment/core_code/scripts/run_care_target_protection.sh \
    Experiment/core_code/CARE_TARGET_PROTECTION.md &&
bash Experiment/core_code/scripts/run_care_target_protection.sh run
```

日志直接显示在当前终端。先冻结所有查询，再另进程拟合，最后另进程评价。
开始各阶段会核验原完成文件、源模型和约 6.5 GB 缓存，读盘期间可能暂时没有新日志。
约束检查增加 CPU 计算量，尚未测量服务器上的实际耗时；不需要显卡、新包或重新提取特征。

看到 `CARE_TARGET_PROTECTION_COMPLETE: .../evaluation/report.md` 才表示整个试验完成。
随后先显示主结果和配对表：

```bash
sed -n '1,30p' Experiment/core_code/outputs/care_target_protection_v1/evaluation/report.md
```

可先只检查来源文件完整性及协议（不加载张量）：

```bash
bash Experiment/core_code/scripts/run_care_target_protection.sh preflight
```

中断后用相同命令续跑，已完成选样/拟合/评价核验后跳过。完成文件损坏会报错，不自动覆盖。
改代码、环境或参数需换新输出目录；同一目录不支持并发写入。

## 四格对照及简单插值控制

| 查询 | 修复 | 新训练次数（默认三种子） | 用途 |
|---|---|---:|---|
| 原 Margin 1000 | 原线性 CARE | 0，复用已有结果 | 基线 |
| 原 Margin 1000 | 逐步受限 CARE | 3 | 原查询下约束是否有效 |
| Margin 800＋保护候选 200 | 原线性 CARE | 3 | 仅改变查询覆盖的作用 |
| Margin 800＋保护候选 200 | 逐步受限 CARE | 3 | 查询与约束的共同作用 |
| 各查询组 | 源头→对应 CARE 结果的最大可行插值 | 0，派生控制 | 是否仅因少更新而改善 |

共新增 9 次线性头拟合，另外复用 3 个基线、派生 6 个控制；全部 18 个头共同评价。
每组仍为 1000 个目标标签＋原每类 5 条历史参考（178 类时共 890 条）。
同一种子的参考行、目标/参考位置抽样流、1380 个 optimizer 步、batch 64 和基础损失保持一致。
插值没有新的优化步；其 trace 的基础训练统计来自对应 CARE 父运行。

## 无标签选样与确认锚点

`select` 不读取目标真值：

1. 修复部分使用源 top-two raw logit margin 最小的 800 条，稳定行号处理并排序。
2. 排除这 800 条后，按**源预测类别**分层，每层保留 margin 不低于该层中位数的候选。
3. 大候选层以固定独立种子取最多 256 条；在单位化特征中先选近候选中心的点，再用最远点扩展覆盖。
4. 以固定打乱的类别顺序轮转分配 200 个名额，不足容量会明确报错。最终恰好 1000 个互异行号。

原 Margin 1000 查询必须逐行复现已冻结的旧查询。两查询组都固定后才允许索引查询标签。
三训练种子共享查询集合；种子变化只影响原参考抽样和优化流。
高置信预测不是正确标签：只有付费查询真值与源预测相符、且间隔足够的点才能成为约束锚点。
不使用完整 M12 真值、受损类 ID、崩溃类名单来选择查询或构造约束。
旧缓存捆绑真值，这里是代码访问边界，并非存储/进程级物理隔离。

## 更新规则

对确认锚点 `(x,y)`，源线性头间隔为 `m0_j = z0_y - z0_j`。
对每个其他类别 `j`，要求新头的间隔满足：

```text
z_y - z_j >= 0.25 × min(m0_j, 1)
```

初始源头可行。每次 AdamW 按原 CARE 损失提出新参数，计算从上一可行头到提议头的最大可行线段比例。
将更新缩至该比例后，对实际 float32 参数重查；间隔违反须不超过 1e-6，锚点 float32 预测仍须正确。
不满足就减半回退；数值检查失败时恢复上一可行头，不能把失败结果登记为成功。
几何间隔用 float64 计算，源正确但间隔极小/平局的点排除，数量记入 trace。

这一规则是保守线段搜索，不是欧氏投影，也不声称求得最优约束 CARE 解。
AdamW 动量仍跟随原梯度历史；接近零的接受步会如实记录。相同优化步数不代表相同计算预算。
约束仅覆盖确认锚点，其他目标样本的安全性须由共同评估检验。

插值控制用相同锚点，计算从源头到对应无约束 CARE 最终头的最大可行比例，不用评估真值调比例。
如果受限更新不优于这类简单控制，就缺少更复杂更新机制的证据。

## 评价与成功判据

所有拟合和派生控制通过完成哈希核验后，`evaluate` 才读取完整目标真值。
共同排除**全部原查询（含旧 random 组）＋全部新查询**的并集，默认评价其余完整 M12。
因此原 Margin 基线也会在新共同排除集上重算，不能直接拼接上次 68.61% 的数值。
Macro-F1 保留全部类别（含零支持类），负翻转分母为源正确样本，空分母留空。

同时检查：锚点覆盖是否增加；恢复/正翻转是否保持；负翻转和逐类损伤是否下降；
间隔执行是否通过；接受步是否接近停滞；是否优于简单插值。
仅降低负翻转而放弃恢复，不算机制成功。先看这轮证据，再决定是否进入其他月份/数据集确认及创新性论证。
这些共享月份的种子不是独立环境；M12 是方法开发数据，这轮不证明新颖性。

## 创新性边界（2026-10-10 核查）

降低负翻转、保护源正确预测、保留分类间隔都已有相关研究，不能把这些目标本身当作创新：

- [Positive-Congruent Training，CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Yan_Positive-Congruent_Training_Towards_Regression-Free_Model_Updates_CVPR_2021_paper.html)：对源正确样本加强蒸馏来减少负翻转。
- [Margin Preserving Training，AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/37825)：在新增类别的更新中使用 logit 间隔校准及双源 focal distillation。
- [ROSE，2024](https://arxiv.org/abs/2411.10896)：使用小型标注验证集限制主动学习候选范围，针对负翻转改善查询。

根据这些来源，我们将当前脚本定位为机制筛选：固定类别漂移、有限目标标签下，
查询覆盖与逐步可行间隔能否协同改善修复/保护。800/200 分配和常规特征覆盖还不是成熟的联合选样算法，
点上的约束也不构成目标分布风险保证。若此轮优于普通 CARE 和简单插值，下一步仍需加入相关方法对照、
发展可论证的预算/覆盖机制，再做跨月份确认；不能仅靠换场景或组合已有部件证明创新。

## 输出与测试

输出 `Experiment/core_code/outputs/care_target_protection_v1/`：

- `manifest.json`、`selection/`：来源、协议、查询分区及共同评估行。
- `runs/<query>/seed_<n>/<repair>/{head.pt,trace.json}`：付费查询/参考、锚点覆盖、优化流、耗时、接受步和约束违反。
- `evaluation/report.md`、`by_seed.csv`、`summary.csv`、`paired_*.csv`：主对照、插值控制及约束执行统计。
- `evaluation/per_class.csv`、`query_coverage*.csv`、`predictions.npz`：完整逐类损伤、付费覆盖和共同排除预测。
- 每阶段最后写入的 `complete.json`：结果哈希和协议身份。

仅继承 `CARE_PROTECT_SOURCE_DIR`、`CARE_PROTECT_AUDIT_DIR`、`CARE_PROTECT_CACHE_DIR`、
`CARE_PROTECT_CHECKPOINT`、`CARE_PROTECT_OUTPUT_DIR`、`CARE_PROTECT_THREADS` 和 `PYTHON`。
其他设置需显式参数；默认继承原训练协议，线程数/环境必须与原实验一致。

```bash
python -m unittest discover -s Experiment/core_code/scripts/tests -p 'test_care_target_protection.py'
bash Experiment/core_code/scripts/tests/test_run_care_target_protection.sh
```

合成测试独立验证所有 rival 的线段边界、零锚点时与原 CARE 数值一致、无标签分层覆盖、
固定预算、付费标签访问 guard、匹配优化流、复用基线、新拟合次数、输入哈希/修改时间保留、
共同查询排除、手算 F1/负翻转、完成门控及无加载续跑。合成分数不作为实验结论。
