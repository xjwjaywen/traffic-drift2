TMA 类别自适应历史保护探索 v1

假设
同训练步数消融显示：历史参考 CE 与 KD 在某些目标类上都会压低召回，
但它们有时改善整体 F1 或减少负翻转。因此尝试只放松已显示明显退化类别的
历史保护项，其余类别保留 CARE 的参考训练与蒸馏。
这是方法探索，不宣称已经证明新方法有效或具备论文创新性。

判别规则（仅用查询期已付费标签）
对训练期支持 >=50 且召回 >=.8 的已学会类别，查询期至少有5个已付费真类样本，
并且 Beta(1,1) 后验 P(源模型在该类错误率 >.5) > .9 时，标记为放松保护。
真类未知或未付费的候选不参与判别；随机/BADGE 均复用原稿保存的1000条查询。
规则与代码写入输出配置；未来测试标签和预测只在判别与头训练保存后读取。

训练
沿用原稿 CARE 的30 epoch、目标重复、历史参考、多重集、随机流、优化步数、
目标 CE、AdamW 和温度。被标记类别的历史参考行不再计参考 CE；
KD 只约束未标记类别之间的相对 logits。目标 CE 对所有已付费已知类别保持不变。
若无类别被标记，训练路径逐参数等同原稿 CARE。
训练后的头仍是一个全局分类头，不能假设影响严格局限于被标记的类别。

前置与运行
同一 Linux 服务器、同一 traffic-ncde 环境，已完成 paper_protocol_v1/source_0/repair_0。
在已拉取 codex/tma-selective-repair 分支的仓库运行：
  cd /data/xjw/traffic-drift2
  conda activate traffic-ncde
  python -u Experiment/tma_adaptive_v1/experiment.py
默认 2020-01 至 2020-04、random 与 BADGE，source_seed=repair_seed=0，
device=cuda，threads=8。其他路径可用 --root；若原稿使用不同设备/线程数，
这里须传相同 --device/--threads。每月/选择器独立完成标记，可原命令续跑。

输出
  /data/xjw/tma-care-data/pilot_v1/adaptive_preservation_v1/source_0/repair_0/REPORT.txt
  同目录 summary.csv、per_class.csv、decision_classes.csv；
  月/选择器目录含 decision.json、头、预测、混淆矩阵与输入哈希。
请回传 REPORT.txt 内容即可初步判断，不必传流量数据或模型。

评价边界
在每个选择器内部，与同一查询、同一预算的 paper_care 对比才是主要方法对照；
paper_target 的优化步数不同，仅作参考。两个选择器之间查询标签不同。
本规则是在看过目标月份后提出的，四个月仍只能算探索，不能作为独立盲测。
若结果值得继续，须固定规则后用新的时间段或独立数据与多修复种子验证。

本地合成检查
  python -m unittest discover -s Experiment/tma_adaptive_v1 -p 'test_*.py' -v
