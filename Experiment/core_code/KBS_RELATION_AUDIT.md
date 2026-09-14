# 吸收关系能否指导 CARE 修复：先做预检

本轮检查一个机制假设：预算内确认的错分关系能否限定有用的修复范围。
它是新机制开发前的诊断及固定后处理试验，没有训练新模型，也不证明创新性。
复用已完成的 `badge_full` 和 `badge_kd`，不依赖之前效果不佳的原型、风险池。

## 服务器执行

在已有 `traffic-ncde` 环境中，从仓库根目录运行：

```bash
cd /data/xjw/traffic-drift2
conda activate traffic-ncde
git pull --ff-only origin main
bash Experiment/core_code/scripts/run_kbs_relation_audit.sh preflight
nohup bash Experiment/core_code/scripts/run_kbs_relation_audit.sh run > kbs-relation-audit.log 2>&1 &
tail -f kbs-relation-audit.log
```

如 Git 再次连接失败于旧的 `127.0.0.1:7897` 代理，临时绕过代理拉取：

```bash
env -u http_proxy -u https_proxy -u all_proxy \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  git -c http.proxy= -c http.https://github.com.proxy= -c remote.origin.proxy= \
  pull --ff-only origin main
```

CPU/NumPy 分析，不占用 GPU，不需要 `.pt` 特征缓存、原始 CESNET 数据或重新训练。
需要原 study 下完整的两组运行、选样和完成清单。校验完成清单中列出的文件，所以
原运行内的 `repaired_head.pt` 等已登记文件也应保留；脚本仅校验其哈希，不加载模型。
默认使用种子 0–4。输入位置与自定义运行示例：

```bash
STUDY_DIR=outputs/kbs_supplement_v1 \
RELATION_OUTPUT_DIR=outputs/kbs_relation_audit_v1 \
RELATION_SEEDS=0,1,2,3,4 \
bash Experiment/core_code/scripts/run_kbs_relation_audit.sh run
```

路径按 `Experiment/core_code` 解析，也可给绝对路径。`RELATION_SEEDS` 独立于其他
训练/探索的种子变量。更改种子、脚本、源结果或 NumPy 版本，需要新的输出目录。
正常重复运行会校验并复用结果；不会修改完成的原实验。输出必须与 study 分离。
`plan` 无需结果文件；`preflight` 校验输入；`run` 顺序启动独立的 `freeze`、`evaluate`
两个进程，冻结失败即停止。不要并发写入同一输出目录。

日志出现 `Results: .../evaluation/report.md` 且命令正常退出，表示本轮完成。
查看以下文件即可回传：

```bash
cat Experiment/core_code/outputs/kbs_relation_audit_v1/evaluation/report.md
cat Experiment/core_code/outputs/kbs_relation_audit_v1/evaluation/relation_by_seed.csv
cat Experiment/core_code/outputs/kbs_relation_audit_v1/evaluation/paired_by_seed.csv
```

## 冻结规则与公平性

失败方向为“真实受害类 v → 旧预测吸收类 a”。仅用原 BADGE 的 1000 个已付费
目标标签确认错分，允许的纠正方向为 `a → v`。每类还允许维持旧预测：

`Gamma(a) = {a} ∪ {v: 存在已查询样本，真实标签 v，旧预测 a}`。

没有使用完整目标标签、修复后预测或事后指定的崩溃类构建 Gamma。冻结合约读取
原 `.npz` 的 `static_pred` 和 `row_id` 成员，查询真值只来自已存 `query_ids.csv`；
`.npz` 的目标真值和新预测虽然捆绑在同一文件内，但冻结进程不解压、不解释这些成员。
完成全部请求种子的关系后写入哈希清单，评价进程必须先验证该清单。

单次确认错分就纳入方向，不扫描最小次数、阈值或候选宽度。一条确认边不等于
该类别整体崩溃；报告单次出现边的数量及查询外覆盖。当前是对这种最简单硬限制
的检验，不将稀疏确认当作可靠的完整类别关系。

每个修复头比较三个版本：

1. 原始 BADGE full / KD 的保存预测。
2. `relation_gate`：新 top-1 在 Gamma(旧预测) 内时接受，否则退回旧预测。
3. `degree_control`：每个旧预测类别保留与 Gamma 相同数量的非自身候选，但从其他
   类均匀无放回抽取目的类；使用独立固定种子偏移 `seed+9187`。

控制保留出度及哪些旧预测类别可改变，不保留入度、目标类别流行度或真实关系，可能
偶然抽中真实边。它检验真实目的类是否有用，不是图拓扑完整消融，也不是多个控制重复。
全程复用同一查询，无新标签。评价用原 `common` 掩码，即排除当次 Margin∪BADGE
查询，并核验确切索引、snapshot ID、full/KD 查询及参考样本、抽样流和预测对应关系。
以前的原型/风险选样查询没有用于此次机制，不将其标签引入此试验，也不额外排除。

源模型与两个原修复的指标从逐样本预测重算，并与已保存的 common 总指标和所有类别
指标对账。不同种子目标样本/标签/源预测须一致，各种子的查询并集可以不同。
原数值引擎、BADGE follow-up、原型探索及固定池脚本保持原样。

## 输出与判读

`evaluation/` 下：

- `report.md`：中文摘要。显示原始/受限修复的 F1、正负翻转、新增崩溃，以及
  阻止负翻转和保留正翻转的比例。没有自动成功/失败或显著性判定。
- `by_seed.csv`、`summary.csv`：7 种预测版本（源模型＋两个修复各三个版本），
  默认 35 行逐种子结果；跨种子均值及样本标准差，缺失分母对应值记为空。
- `paired_by_seed.csv`、`paired_summary.csv`：每个修复中两个限制版本分别相对原修复，
  以及真实关系相对出度控制；默认 30 行。差值方向见 `comparison`。
- `relation_by_seed.csv`、`relation_summary.csv`：确认边、单次边、有效吸收类、允许
  纠正的旧错误比例，以及理想准确率/崩溃类微平均召回上界。控制行的查询证据统计
  仍描述共享的真实查询；候选目的类及其纠正覆盖来自随机控制。
- `relation_per_class.csv`：每种图的所有类别支持、查询真值计数、已确认错误数、
  正确查询数、可纠正错误及该类召回上界，包括零支持、零错误类别。
- `per_class.csv`：全部类别的原始/更新 F1、recall、正负翻转和新崩溃。新增崩溃
  总表沿用原非崩溃组定义，同时单列所有有支持类别的阈值跨越。
- `transitions.csv`：原修复的 `(真值, 旧预测, 新预测)` 全部已观察三元组及计数，
  包括旧错→另一错；标记是否被关系允许、源真值类是否新崩溃、新目的类是否恢复。
  流式写入，避免保留全部种子的转移表。
- `seed_N_gated_predictions.npz`：四个限制版本的逐样本预测、原行号、共同评价掩码。
  不将排除样本混入汇总。

首先检查新崩溃是否流向被恢复的类别，但这只是事后关联，不是因果证明。然后
看关系允许的纠正覆盖是否太低：旧错误只有真实标签落在 Gamma(旧预测) 内才可能
改对。理想上界假定所有允许旧错误全部改对、旧正确全部保持；它不是可实现算法，
也不是宏 F1 的理论上界。宏平均 F1 按原协议包含指定的全部类别；零支持由独立字段标示。

后处理只在接受新 top-1 和退回旧预测之间选择，正负翻转都只能是原翻转的子集。
因此应同时看避免的误伤和损失的恢复；接近不更新不能当作创新收益。
若真实关系相对随机目的类没有作用，或可纠正覆盖过低，应暂停这套硬限制，不能据此
否定所有关系感知方法。若确有相近恢复下的保护优势，再研究训练机制，并加入普通
保护/蒸馏及新旧头插值等对照。

M12 已用于方法选择，所有结果仍为开发性探索。五个查询/修复种子共享源模型和月份。
本轮没有统计总体安全保证、独立验证或原创性证明；未来机制冻结后需要未参与选择的
数据验证。代码测试使用合成数据，只验证算法算术、标签边界和文件协议。
