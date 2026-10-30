TMA 同步数损失消融 v1（探索性）

背景
逐类结果显示：同查询的 target-only 与 CARE 在设备 5、20、26 等类别上差异很大。
原稿 target-only 和 CARE 的多重集大小、训练步数不同，所以不能把差异直接归因于 KD。
本包先做机制诊断，再决定是否有必要设计类别自适应修复。

前置
同一 Linux 服务器、同一 traffic-ncde 环境；已完成 paper_protocol_v1/source_0/repair_0。
只读取完成标记内的 pilot/cache/paper 文件；输出到新的 mechanism_v1 目录。
不覆盖现有 TMA 或 CESNET 结果。

运行
  cd /data/xjw/traffic-drift2
  conda activate traffic-ncde
  python -u Experiment/tma_mechanism_v1/ablate.py
默认 source_seed=0、repair_seed=0、device=cuda、threads=8。
默认仅诊断 2020-03/04 的 random 和 badge，共四组查询，每组训练四个分类头。
训练前检查源模型、缓存、原稿修复头与代码环境。每组完成后可以原命令续跑。
若原稿协议使用非默认 device/threads，需用相同的 --device/--threads。

四个同步数方法
  matched_target：目标 CE；参考行 CE 权重 0，无 KD。
  replay_only：目标 CE + 历史参考 CE，无 KD。
  kd_only：目标 CE + 历史参考 KD，参考行 CE 权重 0。
  matched_care：目标 CE + 历史参考 CE + KD；权重与原稿一致。
四者均使用 CARE 的查询重复、参考行、30 epoch、相同乱序、参考抽样与优化步数。
关闭参考 CE 时只将对应行 CE 置零，目标 CE 贡献保持 CARE 原有量级；
即使某个 batch 只有参考行，也执行优化器步，以维持相同的步数和权重衰减。
matched_care 的分类头参数与未来预测必须和已完成 paper_care 完全一致，否则停止。
原稿 paper_target/paper_care 与静态源模型一同列出；机制归因主要看四个同步数方法。

输出
  /data/xjw/tma-care-data/pilot_v1/mechanism_v1/source_0/repair_0/REPORT.txt
  同目录 summary.csv、per_class.csv；每月/选择器下有训练轨迹、头、预测、混淆矩阵。
REPORT.txt 包含全部方法的宏指标与设备 5、22、26、37、20 的逐类结果。
把 REPORT.txt 内容发回来即可初步判断；不必传原始流量或分类头。

解释边界
四个方法训练步数相同，可拆分历史参考 CE 与 KD 的影响；
但目标月份和重点设备是在看过先前结果后选的，只能做探索性机制分析。
BADGE 查询有选择偏差；不同查询策略的比较不能单独归因于训练损失。
仅调这些目标月份的参数再报告同月份成绩，不构成独立验证。

本地合成测试
  python -m unittest discover -s Experiment/tma_mechanism_v1 -p 'test_*.py' -v
