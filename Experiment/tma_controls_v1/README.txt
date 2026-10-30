TMA 第二轮：损失消融与固定混合采样（探索性实验）

背景：首轮发现 3 月设备 5/22/26/37、4 月设备 20 达到预设崩溃标准；
逐设备检查未发现这些案例主要由 raw16/transformed32 的跨类别精确重复造成。
这不证明它们必然可分或可恢复。第一轮未来结果已经被查看，因此本轮是探索，
不能把它当作新方法在独立测试集上的确认性结果，更不能隐去无收益的月份。

前提：pilot_v1/seed_0 已完整结束，保留缓存、源模型、参考记录、查询记录、各修复分类头及预测文件。
本脚本不下载数据，不扫描原始压缩包，不重新训练源编码器，不修改首轮目录和实现。
复用首轮完整历史参考集合（包括原来的训练集补足记录）、原始查询顺序和固定优化参数。
读完校验和后重载模型，要求源模型的未来预测与首轮完全一致；损失消融的训练批次哈希也必须一致。

服务器前台运行：
cd /data/xjw/traffic-drift2
conda activate traffic-ncde
git pull --ff-only origin main
set -o pipefail
python -u Experiment/tma_controls_v1/controls.py 2>&1 | tee tma-controls-v1.log

默认根目录 /data/xjw/tma-care-data/pilot_v1，seed=0，cuda，CPU 线程数 8。
可用 --root、--seed、--device、--threads 指定环境；其他种子必须先完成原始 pilot。
同一命令可复用校验通过的已完成第二轮结果；未完成第二轮会从头重跑，首轮不会重跑。
需保持原 Python 包版本。仅运行到 CONTROLS COMPLETE 才代表本轮完成。

五个新增条件，在全部四个月运行，维持相同标签预算、源头、参考集合和优化步数：
random_kd：原随机查询 + 目标 CE + 0.5 KD，取消参考 CE。
badge_replay：原 BADGE 查询 + 目标 CE + 参考 CE，取消 KD。
badge_kd：原 BADGE 查询 + 目标 CE + 0.5 KD，取消参考 CE。
hybrid_target：固定一半随机、一半 BADGE + 目标 CE。
hybrid_care：同一混合查询 + 目标 CE + 参考 CE + 0.5 KD。
混合采样保留原随机查询前半部分，然后在剩余无标签候选中选 BADGE，绝不重复计费。
比例预先固定为 1/2；不在未来测试上搜索比例，也不按真实类别分配预算。
“仅 KD”指额外历史约束只有 KD，目标 CE 始终存在；温度仍为 2，其他参数沿用首轮。
这些是控制实验与简单基线，不宣称构成新方法。

同时为原六个修复条件和新五个条件输出每类：
已查询样本上的训练拟合召回、未查询候选上的召回、未来测试上的召回。
所有未付费候选标签和未来标签仅用于训练结束后的诊断，不进入采样或优化。
前两者不能代替独立未来测试；查询池内部也可能存在时间/会话相关性。
原始 RF/MLP 静态对照保留在合并 summary.csv 中，不重新训练。

结果目录：/data/xjw/tma-care-data/pilot_v1/controls_v1/seed_0/
请返回 REPORT.txt（结束时会打印到终端）、summary.csv。
详细文件：paired_deltas.csv、query_generalization.csv、query_dates.csv、per_class.csv，及每月的新分类头/查询/预测/优化记录。
query_dates.csv 保留每类逐日的候选数和实际查询数，便于检查查询是否集中在月初旧模式。

解释方向：固定查询的消融用于区分参考 CE 与 KD 的影响；混合采样用于检查覆盖保障是否有帮助。
若查询样本仍拟合不好，再检查优化和冻结表示；若查询拟合好但未查询/未来差，检查覆盖与时序泛化。
上述现象提供定位线索，不单独证明因果，不能据此保证新模型或增加标签必定有效。

合成测试：python -m unittest discover -s Experiment/tma_controls_v1 -p 'test_*.py'
