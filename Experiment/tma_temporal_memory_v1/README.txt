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
