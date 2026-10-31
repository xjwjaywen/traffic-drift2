TMA 预算内非线性分类器对照 v1

为什么做
  事后多标签诊断显示，冻结特征上的非线性二分类探针能识别重点崩溃类，
  但该探针用了远超 1000 条的标签。这里严格回到每月 1000 个已付费标签，
  检查只换一个现成非线性分类器是否有帮助。它是简单基线，不是新方法。

服务器运行
  cd /data/xjw/traffic-drift2
  git pull --ff-only origin codex/tma-selective-repair
  python -u Experiment/tma_nonlinear_head_control_v1/experiment.py
  cat /data/xjw/tma-care-data/pilot_v1/nonlinear_head_control_v1/source_0/REPORT.txt

需要已完成的 paper_protocol_v1 和 hybrid_adaptive_v1 修复种子 0--4。
默认固定源种子 0，覆盖 2020-01 至 2020-04 与修复种子 0--4。
如先试一次，可加 --repair-seeds 0；输出签名固定种子集合，之后要跑
五种子须先移走原 nonlinear_head_control_v1/source_0 输出，故建议直接默认运行。

对照
  raw_rf：源期预处理后的 32 维原始特征上训练随机森林。
  frozen_rf：同样的模型训练于冻结编码器的 128 维表示上。
  两者使用同一个 hybrid 付费查询集合和同一批每类五个历史参考；
  200 树、最小叶子 2、balanced_subsample，其余为 sklearn 默认值。
  直接比较原 hybrid_target、hybrid_care 和静态 MLP 的保存预测。
  汇总宏 F1、负翻转率与重点设备 F1。每对模型先保存预测，再读取未来标签。

解释边界
  RF 与 CARE 的模型结构和训练目标均不同，差异不能只归因于“非线性”。
  只要 RF 在完全相同预算下不稳定改善，就不值得沿简单非线性头继续包装。
  即使改善，随机森林本身是现成基线，仍须提出独立机制并做外部验证。
  所有目标月份已经多次检查，结果只能用于探索，不是独立盲测。

零训练的错误互补性审计
  python Experiment/tma_nonlinear_head_control_v1/rf_complementarity.py
  cat /data/xjw/tma-care-data/pilot_v1/rf_complementarity_v1/source_0/REPORT.txt
  它统计 RF 与 CARE 各自独有的纠错和新增的源正确样本误判。
  另算一个使用未来真值选择 RF 预测类的事后路由；它按分组正确数作选择，
  不最大化宏 F1，也不是宏 F1 的严格上界。它存在标签泄漏，绝不能当
  实际方法的性能。输出目录必须是空的新目录。
