TMA target-only 与 CARE 的逐样本配对诊断

目的：检查设备 20 的目标头正确预测有多少被 CARE 改错，同时统计
CARE 纠正了多少目标头错误，以及错误流向哪些设备。只读已经完成的
quota_query_v1、hybrid_adaptive_v1 和源模型产物；不训练、不调参。

服务器运行（需先完成 quota repair seeds 0--4）：
  git pull --ff-only origin codex/tma-selective-repair
  python -u Experiment/tma_interference_audit_v1/audit.py

默认输出：
  /data/xjw/tma-care-data/pilot_v1/interference_audit_v1/source_0/repair_0_1_2_3_4/2020-04/device_20/REPORT.txt
  paired_summary.csv          每个 seed、查询方法的配对变化
  per_class_interference.csv  各真实设备的得失与源模型正确性
  device_routes.csv           设备 20 的源模型→目标头→CARE 预测去向

可选参数：--repair-seeds 0,1,2,3,4 --month 2020-03 --device-id 5。
这是对已看过月份的事后诊断；逐样本预测差异本身不能证明 CARE 的
蒸馏项或回放项是原因，需要后续同标签、同随机种子的损失消融。
