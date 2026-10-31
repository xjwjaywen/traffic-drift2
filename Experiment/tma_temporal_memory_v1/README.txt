TMA 跨月已付费标签可用性审计（探索性，只读输入）

问题：前几个月已经付费取得的真实设备标签，能否为后续月份的少数设备
提供更多样本？这些旧样本在冻结编码器的特征空间里是否仍靠近本月同类？

运行（服务器 traffic-ncde 环境，先拉取 codex/tma-selective-repair）：
  python -u Experiment/tma_temporal_memory_v1/audit.py

只快速检查 seed 0：
  python -u Experiment/tma_temporal_memory_v1/audit.py --repair-seeds 0

默认使用 0--4 五个修复种子；每个种子复用已完成的原稿和 hybrid 查询，
要求每月 1000 条付费查询且前 500 条与原稿随机查询前缀一致。
仅从当月查询池按已付费 ID 取真标签。上一月/更早月份的已付费标签
可用于下一月的记忆；本月前 500 条随机付费标签只作为诊断锚点。
不读取任何 *_eval.npz 或未来测试预测，不训练新分类头，不改变旧输出。

输出：
  /data/xjw/tma-care-data/pilot_v1/temporal_memory_audit_v1/source_0/aggregate_0_1_2_3_4/
  REPORT.txt           重点设备和总体摘要
  paid_class_counts.csv 每 seed/月/设备的当月及旧月付费标签数
  anchor_transfer.csv  旧付费标签的最近邻/类别中心在本月随机锚点上的诊断
  overview.csv         所有类别的概况

请把 REPORT.txt 发回来。若旧标签对困难设备既无数量也无跨月特征相似性，
就不应直接上跨月记忆模型。若有信号，再固定协议比较每月独立 CARE、
简单累积付费标签、容量受限的时间记忆。TMA 所有月份已被看过，后续
任何提升只能算探索，正式结论需要独立数据或未看过的时间段。
最近邻命中率同时列出按旧类比例随机抽记忆的期望值，避免把大类
样本多造成的自然高命中率误认成跨月特征迁移。

审计后固定的一轮训练对照（不按重点设备选样或调参）：
  python -u Experiment/tma_temporal_memory_v1/experiment.py --repair-seeds 0
  python -u Experiment/tma_temporal_memory_v1/experiment.py

2020-02/03/04 每月只使用本月同一批 1000 个 hybrid 付费标签，以及
严格更早月份已付费的查询。新方法每类最多保留 10 条旧标签，优先
最近月份，月份内随机抽取；对照方法重复相同数量的本月付费样本。
两个新分类头都从固定源头重置、使用原 CARE 历史参考/KD 和相同更新步数。
报告同时列出原 hybrid CARE、重复当前样本、跨月付费记忆三组。
每个月两组新分类头保存后才打开该月测试集，之后才继续下一月。
这只是“旧标签能否带来额外收益”的探索，不是新方法有效性结论。
输出在 temporal_memory_trial_v1/source_0/aggregate_0_1_2_3_4/REPORT.txt。
