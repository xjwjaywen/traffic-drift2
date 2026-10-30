TMA 修复可行性审计 v1（探索性诊断，不是新方法的效果实验）

目的
  检查少量、代表性的付费标签能否预测哪个完整模型对某个真实设备类有帮助。
  候选完整模型：固定源模型、只校正输出偏置、只用目标标注修分类头、CARE。
  只校正输出偏置是有监督的正则化校准，不等于验证纯 label shift。
  此版没有测试编码器适配；只有前面这些选择有可辨识性才值得增加该候选。

现有输入（不会覆盖）
  /data/xjw/tma-care-data/pilot_v1/seed_0
  /data/xjw/tma-care-data/pilot_v1/paper_protocol_v1/source_0/repair_0
  /data/xjw/tma-care-data/pilot_v1/hybrid_adaptive_v1/source_0/repair_0

服务器运行
  git pull
  python -u Experiment/tma_repairability_audit_v1/audit.py

只先看崩溃月份也可以
  python -u Experiment/tma_repairability_audit_v1/audit.py --months 2020-03,2020-04
  注意输出签名固定了月份，若先跑两个月，后来想跑四个月，需要把
  repairability_audit_v1/source_0/repair_0 的旧输出移到别处或使用新路径。
  因此建议直接运行默认命令。

查看结果
  cat /data/xjw/tma-care-data/pilot_v1/repairability_audit_v1/source_0/repair_0/REPORT.txt
  详细逐类表：同目录 choice_outcomes.csv
  每月的 oof_predictions.npz、decision.json、oof_global.csv 和 future_global.csv
  可用于核查决策是否只用了付费标签。

协议
  每月复用已有 1000 个 hybrid 查询：前 500 个原有随机查询作代表性审计，
  后 500 个 BADGE 查询仅供分类头训练。五折交叉拟合中，被审计的随机
  查询不会进入该折分类头或偏置校正的训练。所有参数和阈值写死在代码中。
  decision.json 在读取未来评估标签和已有完整模型预测之前写出。
  每个真实设备类要求随机样本至少 10 条；否则标为证据不足，回退 CARE。
  如果其他候选在付费折外 F1 高于 CARE 至少 0.05，则选择该候选。

特别注意
  类别选择按真实类统计，只用于研究可辨识性。真实类在部署时未知，
  不能把逐类选择拼成一个可部署分类器，也不能把未来最佳方法当选择器。
  后半月测试集仅用于事后检查选择是否正确和计算后悔值。
  2020-01 至 2020-04 已被多次检查，这些结果不是独立时间盲测。
  若已付费样本不足、选择结果不稳定，应停止该方向，而不是继续调阈值。
