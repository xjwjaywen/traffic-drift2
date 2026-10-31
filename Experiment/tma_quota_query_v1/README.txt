TMA 限额定向查询 + BADGE 探索 v1

这一步接续 victim_retrieval_v1 的发现：邻近检索可以找到设备 5，
但每类固定分 100 次查询损害其他类别覆盖。本试验改成“找到够用就停”。

服务器运行（已激活 traffic-ncde）
  git pull --ff-only origin codex/tma-selective-repair
  python -u Experiment/tma_quota_query_v1/experiment.py
  cat /data/xjw/tma-care-data/pilot_v1/quota_query_v1/source_0/repair_0/REPORT.txt

输入：已有 source_0/repair_0 的 paper_protocol_v1、hybrid_adaptive_v1、
TMA pilot_v1 缓存。输出写到新的 quota_query_v1 目录，按月可续跑。

查询规则事先写死：
  每月 1000 标签；前 500 与混合 BADGE 原运行完全相同。
  从真实随机标签估计各类出现率，用未标注池的源预测估计预测率。
  只考虑随机样本少于 15、至少 2 条真类且已有源错分的欠预测类；
  按欠预测量最多取 10 类。只在它们被错分进去的源预测区域内找邻居。
  顺序查询时，新标签必须先付费取得；某类累计 15 条真标签即停止，
  每类最多试 25 次，每月定向查询最多 150 次。剩余预算由 BADGE 选择。
  未标注候选的真类、后半月评估标签均不参与查询决定。

对照：相同 1000 标签预算下的 hybrid_target / hybrid_care。
  用新查询分别训练 target-only 和原 CARE 分类头；原模型和训练协议一致。
  REPORT.txt 同时列出目标设备的真实付费标签数、宏 F1、崩溃类 F1、
  负翻转，以及设备 5、22、26、37、20 的逐类结果。

解释边界：我们已看过 2020-01 至 2020-04，且这版限额受上一轮结果启发；
因此结果只用于探索。即使这次表现好，也需冻结规则后做独立时间验证。
