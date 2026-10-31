付费纠错梯度约束的 KD 探索试验

根据 2020-04 设备 20 的事后消融发现提出；不是已验证的新方法。
保留 CARE 原有付费目标监督、历史参考回放监督和 KD 权重。
每一步用当月已付费且被固定源模型错分的样本计算纠错梯度；
仅当参考 KD 梯度与纠错梯度相反时，投影去掉冲突分量。
查询身份、标签预算、源模型、参考样本和优化步数均与完整 CARE 一致。

服务器运行（已激活 traffic-ncde，先完成 component_ablation_v1）：
  git pull --ff-only origin codex/tma-selective-repair
  python -u Experiment/tma_component_ablation_v1/projection.py

汇总：
  /data/xjw/tma-care-data/pilot_v1/gradient_safe_kd_v1/source_0/aggregate_0_1_2_3_4/2020-04/device_20/REPORT.txt
  by_seed.csv   逐 seed 对照
  mean.csv      五 seed 均值

这是在已看过月份上提出并测试的候选方法。即使改善，也不能作为独立
有效性结论。原 KD 与此投影方法采用相同回放行数和优化步数，便于
隔离 KD 梯度约束的作用；与 target-only 的对照仍含回放步数差异。
