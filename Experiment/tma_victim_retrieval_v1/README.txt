TMA 被吞并设备的定向查询覆盖试验 v1

目的
  不训练新模型，只检查同样 1000 标签预算下，能否比随机或混合 BADGE
  更快找到被源模型错分的真实设备样本。

运行（服务器 traffic-ncde 环境）
  git pull --ff-only origin codex/tma-selective-repair
  python -u Experiment/tma_victim_retrieval_v1/retrieve.py
  cat /data/xjw/tma-care-data/pilot_v1/victim_retrieval_v1/source_0/repair_0/REPORT.txt

需要已有 seed_0、paper_protocol_v1/source_0/repair_0 和
hybrid_adaptive_v1/source_0/repair_0。默认只跑源种子/修复种子 0。
不需要额外下载数据，也不使用后半月测试集。

协议
  复用 paper_protocol_v1 原随机查询的前 500 条作为相同付费起点。
  从这些真标签估计每类出现率；从未标注候选池的源模型预测估计每类预测率。
  在至少 2 条随机真类标签、至少 1 条源错分的类中，优先选择真类出现率
  高于预测率的前 5 类。每类把随机样本实际错分到的前 2 个类别当候选区域。
  三种探索策略：在候选区域里找最像受害类种子的流量；同时排斥真吞并类
  种子的对比检索；同区域随机抽样对照。每策略再选 500 条；若区域不足，
  缺口在其余池中均匀抽样。每种策略的总预算都是 1000 条。
  完整随机 1000 与已有混合 BADGE 1000 是固定基线，前 500 条相同。
  selection.json 在读取尚未付费的候选标签统计覆盖率之前写出。

看什么
  报告给出 2020-03 设备 5、22、26、37 与 2020-04 设备 20 的
  总付费真类条数，并列出各月达到 10 条真类标签的设备类数。
  也要看 focus_classes.csv：如果目标设备根本没有被规则选中，覆盖再高
  也不能当作可部署算法证据。coverage.csv 给出所有类和策略的计数。
  本试验只比较查询覆盖，不代表后续 CARE 性能会提升。
  所有月份已被反复查看；结果只能筛选方向，不能作为独立验证。
