TMA 选择性 CARE 探索试验 v1

目的：检验“哪些原模型预测类别需要启用 CARE”是否能减少负翻转，同时保留崩溃类恢复。
这是原稿协议之后的一轮探索；只有目标月份结果支持，才考虑写入 IoTJ 稿件。

前置：同一服务器、同一 traffic-ncde 环境，已有完整的
  /data/xjw/tma-care-data/pilot_v1/paper_protocol_v1/source_0/repair_0/complete.json
如果没有，先运行 Experiment/tma_paper_v1/run.py 完成原稿协议。

服务器运行（从当前仓库分支拉取后）：
  cd /data/xjw/traffic-drift2
  conda activate traffic-ncde
  python -u Experiment/tma_selective_repair_v1/trial.py
可选：--root /其他路径/pilot_v1 --source-seed 0 --repair-seed 0 --device cuda --threads 8
全程前台执行，每月打印进度。中断后可原命令续跑已完成月份；同一源种子不能并发运行。

每月使用 paper_protocol_v1 保存的 BADGE 查询，付费标签预算不变。
五折交叉拟合 CARE：每个折外查询的修复预测来自未见过该查询标签的头。
按源模型预测类别统计“源错 CARE 对”和“源对 CARE 错”；已知付费样本至少 10、
两头预测正确性不同的样本至少 3、Beta(1,1) 后验的净增益概率 > .9 才启用 CARE。
部署时用源模型预测类别选择原头或完整预算训练的 CARE 头；未知类标签扣预算但不参与门控。
门控在读未来测试预测/标签前写入 gate.json。折外头仅用约 80% 标签训练，
因此门控估计可能偏保守；折外训练增加算力但不增加标注成本。
BADGE 查询分布不等于实际测试分布，类内净收益估计也可能有选择偏差。

输出：
  pilot_v1/selective_repair_v1/source_0/repair_0/REPORT.txt
  同目录 summary.csv、per_class.csv、gate_classes.csv
  月目录 gate.json、oof_predictions.npz、oof_training.json、predictions.npz、混淆矩阵
  所有输出有输入清单与校验标记；不修改已有 pilot 或 paper_protocol_v1 结果。

服务器跑完请给我以下内容，不必传原始流量或模型：
  cat /data/xjw/tma-care-data/pilot_v1/selective_repair_v1/source_0/repair_0/REPORT.txt
  cat /data/xjw/tma-care-data/pilot_v1/selective_repair_v1/source_0/repair_0/gate_classes.csv

判断：逐月比较 static_mlp、badge_care、gated_care 的 macro-F1、崩溃组 F1、
剩余崩溃数、高召回组 F1 与负翻转率。不要只按某一个月或单个分数选方法。
这些月份先前已经看过，属于探索结果；后续需要新的时间段或独立数据验证。

本地合成检查：
  python -m unittest discover -s Experiment/tma_selective_repair_v1 -p 'test_*.py' -v
