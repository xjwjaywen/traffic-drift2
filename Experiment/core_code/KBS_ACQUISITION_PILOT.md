# 受害者样本发现：选样探索 v1

本轮只检查能否在固定标注预算内找到高置信错分和更多受害类别样本，**不训练、不报告修复 F1**。M12 已参与提出此方案，结果只能作为开发性探索。旧论文主表、受控消融、BADGE KD 和敏感性实验均保留。

## 服务器运行

在已有 `traffic-ncde` 环境、原仓库根目录执行：

```bash
cd /data/xjw/traffic-drift2
conda activate traffic-ncde
env -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    git -c http.proxy= -c http.https://github.com.proxy= -c remote.origin.proxy= \
    pull --ff-only origin main
bash Experiment/core_code/scripts/run_kbs_acquisition_pilot.sh plan
nohup bash Experiment/core_code/scripts/run_kbs_acquisition_pilot.sh run > kbs-acquisition-pilot.log 2>&1 &
tail -f kbs-acquisition-pilot.log
```

拉取成功后再运行脚本。`Ctrl+C` 只退出 `tail`，不会终止 `nohup` 任务。
看到 `Results: .../evaluation/summary.csv` 表示选择和评估均已完成。

把以下输出发回分析：

```bash
cat Experiment/core_code/outputs/kbs_acquisition_pilot_v1/evaluation/report.md
cat Experiment/core_code/outputs/kbs_acquisition_pilot_v1/evaluation/summary.csv
cat Experiment/core_code/outputs/kbs_acquisition_pilot_v1/evaluation/paired_by_seed.csv
```

默认读取 `outputs/kbs_supplement_v1/cache` 和已有五种子 BADGE/Margin 选样，预算从原 study manifest 继承（当前 1000）；五个种子为 0–4。每种方法都查询相同数量的不同目标行。无需安装新包、下载数据或重新抽取特征。

入口也支持 `preflight`（验证旧缓存/选样和 CUDA）、`select`（只选样）、`evaluate`（只做事后标签评估）。`run` 依次启动两个独立 Python 进程，选样失败或任何种子未完成时不进入评估。可从仓库根目录或 core_code 目录调用。

环境变量：`PYTHON`、`GPU_ID`、`DEVICE`、`STUDY_DIR`、`CACHE_DIR`、`CHECKPOINT`、`PILOT_OUTPUT_DIR`、`PILOT_SEEDS`。输出默认 `outputs/kbs_acquisition_pilot_v1`，与原 study 分开，不使用旧 `OUTPUT_DIR`。调整 seed/device/batch-size/软件版本或代码时使用新输出目录，不覆盖完成结果。重跑相同命令校验并复用已完成种子。

## 预先固定的最小方法

1. **参考几何**：原受控协议的 `replay_indices`，每类 5 个、同一种子的同一批参考特征（178×5=890）。每个向量先 L2 归一化，按真实参考类别取均值，再归一化得到原型。不使用额外目标标签或针对失败类别手工建原型。
2. **目标分歧**：旧分类头预测 b；对目标特征与所有原型的余弦相似度做温度 0.1 的 softmax，得到 q，令 a=argmax(q)。只在 a≠b 且目标特征非零时定义正分数 `q[a] * p_head[b]`。分歧只表示候选，不视为真实错分或已证实的吸收关系。
3. **类别风险代理**：参考/目标预测频率分别使用 `(count[c]+0.5)/(N+0.5*C)`，`risk[c]=clip(1-target_freq/reference_freq,0,1)`。只使用模型预测计数，**不是原稿的五信号监测器**。真实应用流量减少、标签先验变化也能产生风险，不能把风险当作崩溃真值。
4. **风险加权**：基础分歧分数乘 `1+2*risk[a]`。控制组打乱 risk 与类别的对应关系后，保持同一公式。
5. **固定探索份额**：三个新方法使用同一个种子下相同的 `floor(B/2)` 个全池随机样本。剩余名额从正分数候选中选；按各自分数保留最多 `20*B` 个候选（分数相同按行 ID 排），用分数加权 k-means++ 距离采样保证特征多样性。候选不足则从未选行随机补足，日志和结果记录补足数量。几何距离针对专项集合计算，不以探索集合初始化。

没有参数扫描或自动选择“最优”方案。高置信阈值 0.9 仅用于评价；选样的头置信度是连续权重。

| 方法 | 定义 |
|---|---|
| badge | 原 study 已验证的 B 个 BADGE 行，原样复用 |
| margin | 原 study 已验证的 B 个 Margin 行，原样复用 |
| random | B 个全池随机行，检查简单随机是否已经足够 |
| disagreement | 半随机＋基础分歧专项选择 |
| risk_disagreement | 半随机＋类别风险加权分歧 |
| shuffled_risk | 半随机＋打乱类别风险的分歧对照 |

普通分歧组与风险组使用相同的原型、探索行、预算、候选上限、距离规则和随机种子；风险权重会改变候选池和专项选择。BADGE 是原始全预算基线，不含半随机混合。

## 防止标签泄漏与结果混用

- 原 .pt 缓存把特征、logits、标签存于同一文件；选择进程反序列化后立即删除目标标签项。所有选择函数只有参考标签、目标特征/预测、风险等输入，没有目标标签或预定义崩溃类别参数。
- 用标签事后判断是否选中高置信错误，属于离线评估；不是声称部署时能够免费知道全池真值。
- 选择先保存全部方法/种子的行 ID、推断类别、参考 ID、风险和打乱映射。逐种子的 complete.json 最后写入并记录 SHA256。评价进程只有校验全部指定种子后才读取目标标签。
- 缓存 manifest、缓存文件、源 checkpoint、原 study、原 BADGE/Margin 选择、代码与运行配置均绑定。修改、缺失、不匹配或重复 ID 会报错；不会静默重新生成旧缓存/旧选样。
- 每个 seed 保存所有六种方法查询并集 `common_excluded_ids.json`。本轮评价的是所选标签的构成，没有训练后的测试分数。若下一阶段修复，需要在共同剩余目标集上重新评估所有基线，不可直接拼接旧 `primary_summary.csv`。
- 原始崩溃/稳定组只在事后评价从原 study 中读取。逐类输出覆盖全部类别，包括零支持类别；不为 109、167 或其他已知失败类别添加选择规则。

## 输出与判读

- `pilot_manifest.json`：冻结方法参数、输入和软件/代码标识。
- `seed_N/selection.json`：六种方法的行 ID；三个新方法的推断类别、探索/专项/补足数量；参考行 ID 和风险映射。
- `evaluation/report.md`：中文摘要。
- `evaluation/summary.csv`：五种子均值与样本标准差；单种子或未定义比率的标准差留空。
- `evaluation/by_seed.csv`、`paired_by_seed.csv`：逐种子指标与 risk 减 BADGE/普通分歧/打乱风险的差值。
- `evaluation/per_class.csv`：每类全池支持度、选中数、错分数、高置信错分数，含零行。
- `evaluation/pool.json`：全池高置信错分数量、类别支持与评价分组。

重点同时看高置信错误数、崩溃类查询数/覆盖数、非崩溃类覆盖和专项推断准确率。高置信错误富集倍数的分母是同一目标池的高置信错误比例。没有这种错误时相关比例留空。专项推断准确率只评价专项选择，排除随机探索和补足行。

先回答风险是否超过普通分歧和打乱对照；再与 BADGE 和随机比较。更多错误标签可能只是选到了难以学习的样本，并不保证 F1 提高。标签发现、模型恢复、保护其他类别和跨时间泛化需要分别验证。不要根据一个种子、一个失败类别或一个指标选出“成功”。本轮不报告 p 值，也不自动判定成功。

Margin 原始选样可能在五种子完全一致，这不是五次独立验证；其他种子也不是独立源模型或月份。若继续开发，需冻结方案后在未用于选方案/调参的数据上验证。

首次运行会校验并读取数 GB 缓存，CPU 内存需求包括原缓存张量；GPU 按批计算原型分数，专项多样性仅使用最多 20,000 个候选特征。CPU 也可运行，但速度不同；未在本地运行真实 CESNET/CUDA，耗时由服务器确认。不要同时启动多个写入相同 pilot 目录的进程。
