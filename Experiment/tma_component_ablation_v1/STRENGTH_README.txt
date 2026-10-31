KD 梯度投影的简单强度对照

2020-04 的投影法几乎每个优化步都触发投影，因此需要排除“只是把
KD 整体变弱”这一解释。此对照保留 CARE 的付费标签、回放行、优化
步数和源模型，只把 KD 权重从 0.5 固定减半到 0.25。

服务器运行（已激活 traffic-ncde，先完成 component ablation 和 projection）：
  git pull --ff-only origin codex/tma-selective-repair
  python -u Experiment/tma_component_ablation_v1/strength.py

报告：
  /data/xjw/tma-care-data/pilot_v1/kd_strength_control_v1/source_0/aggregate_0_1_2_3_4/2020-04/device_20/REPORT.txt

这是看过四月结果后选择的对照，仅用于探索。若投影与半强度 KD
效果相近，就不能把收益归功于梯度冲突定位，更不能据此宣称创新。
