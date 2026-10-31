TMA 设备 20 的 CARE 组件消融（事后诊断）

在已完成的 hybrid / quota 查询上，固定每个 seed 的 1000 条付费标签、
源模型、参考样本和训练参数。复用已保存的 target-only 和 full CARE
预测；只新训练两个头：
  replay_only = 目标监督 + 历史参考监督，不做蒸馏
  kd_only     = 目标监督 + 历史参考蒸馏，不做参考监督

服务器运行（已激活 traffic-ncde，已完成五个 quota seed）：
  git pull --ff-only origin codex/tma-selective-repair
  python -u Experiment/tma_component_ablation_v1/experiment.py

汇总：
  /data/xjw/tma-care-data/pilot_v1/component_ablation_v1/source_0/aggregate_0_1_2_3_4/2020-04/device_20/REPORT.txt
  by_seed.csv     各 seed、查询方式、训练组合的指标
  mean.csv        五 seed 平均

每 seed 的两个新头均保存之后，才读取未来月份的测试预测和标签。
这是在已看过月份上的机制诊断，不能作为独立有效性验证。
回放会增加训练行数和优化步数，因此四组差异也包含训练步数变化；
若有稳定信号，再做固定步数的损失对照。
