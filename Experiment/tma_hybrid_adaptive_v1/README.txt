TMA 半随机、半 BADGE 查询与类别自适应修复 v1

问题
2020-03 设备5：随机查询有14条真类样本，BADGE仅1条；
2020-04 设备20：随机查询7条中源模型错3条，BADGE 12条中错10条。
这提示覆盖和困难样本发现之间有取舍，但不能仅凭这些计数断定抽样偏差或月内漂移。
本轮固定半随机、半 BADGE 的查询策略，比较同查询下的 CARE 与自适应修复。

方法
每月从原稿已保存的均匀随机查询顺序取前500条，再从剩余候选中用原仓库
BADGE 近似选500条；两步都不看标签。总共付费1000条，不从其他实验借标签。
同一批标签训练 hybrid_target、hybrid_care、hybrid_adaptive 三个头。
adaptive 判别沿用 adaptive_preservation_v1：训练期学好的类，在至少5条已付费真类
查询上有高置信错误率时，减少该类历史参考 CE 并从 KD 类别集合中移除。
三组从同一源模型启动；hybrid_care 与 hybrid_adaptive 使用相同 CARE 步数。
查询选择、判别和三个头保存后，才打开未来测试标签与预测。

前置和运行
服务器使用 traffic-ncde 环境，已有 paper_protocol_v1/source_0/repair_0 完成结果。
在已拉取 codex/tma-selective-repair 分支的仓库运行：
  cd /data/xjw/traffic-drift2
  conda activate traffic-ncde
  python -u Experiment/tma_hybrid_adaptive_v1/experiment.py
默认四个月、source_seed=repair_seed=0、device=cuda、threads=8。
其他数据位置传 --root；设备/线程参数须与原稿协议运行一致。
每月单独完成标记，中断后原命令续跑。

输出
  /data/xjw/tma-care-data/pilot_v1/hybrid_adaptive_v1/source_0/repair_0/REPORT.txt
  同目录 summary.csv、per_class.csv、decision_classes.csv、query_coverage.csv。
  每月目录有查询清单、判别、三个头、预测、混淆矩阵和输入校验。
  query_coverage.csv 的候选池真类计数只用于事后诊断，不参与选择或训练。
请返回 REPORT.txt 内容即可，不必上传原始数据或模型。

解释边界
hybrid_adaptive 对 hybrid_care 是同查询、同标注预算和相同 CARE 优化步数的
训练方法比较。与旧 random/BADGE 方法比较时，查询身份也改变，不能单独归因训练。
查询混合本身已有先例，不能单凭此试验宣称方法新颖；所有目标月份此前均被观察，
本轮只能筛选研究方向，不能当独立验证。若结果有前景，应固定规则后测新时间段或
独立 IoT 数据，并重复修复种子。

本地合成测试
  python -m unittest discover -s Experiment/tma_hybrid_adaptive_v1 -p 'test_*.py' -v
