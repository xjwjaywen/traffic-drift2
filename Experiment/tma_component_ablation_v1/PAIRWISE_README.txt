付费混淆关系的局部 KD 探索试验

背景：TMA 四月设备 20 常被源头判成 30；完整 KD 压住一部分修复。
梯度投影在几乎全部优化步触发，与简单降低 KD 权重呈取舍关系。
本试验不手写设备 20/30，也不使用未来标签选择边界。

方法：从当月 1000 条已付费标签构建“真类 c、源预测 s”的错分计数。
将 count(c->s)/(paid_count(c)+1) 与反方向的比率对称化，得类别图权重。
对历史 KD 梯度施加 (I+L)^-1，其中 L 是此图拉普拉斯矩阵；因此
仅在有付费错分证据的类别边界上减弱 KD，原有历史回放 CE 保留。
没有已付费源错分时，算子为 I，训练退化为原 CARE。参数和步数固定。

服务器运行（先完成 strength.py 的 5 个 seed）：
  git pull --ff-only origin codex/tma-selective-repair
  python -u Experiment/tma_component_ablation_v1/pairwise.py

报告：
  /data/xjw/tma-care-data/pilot_v1/pairwise_kd_v1/source_0/aggregate_0_1_2_3_4/2020-04/device_20/REPORT.txt
每 seed 的 paid_confusion_edges.csv 只含当月已付费证据；头与边界决策
在读取未来测试文件之前保存。与 CARE、half KD、projected KD、
replay-only 作相同标签、源模型和种子的描述性比较。

本轮先固定继续研究的门槛：在 hybrid 与 quota 两种查询下，五 seed
平均的整体 macro-F1 和设备 20 F1 均不低于 half KD，同时平均负翻转
率不高于 half KD。任一条件不满足，就不能说局部 KD 优于简单减半，
应停止沿此规则调参。逐 seed 结果须完整报告，不只报平均值。

限制：本规则是在看过四月结果后提出的，不能称为独立验证或已证明
的论文创新。类别图平滑与选择性蒸馏也有相关先例；需与简单 KD 权重
控制、强基线及未看过的数据对比，才可评估额外价值。
