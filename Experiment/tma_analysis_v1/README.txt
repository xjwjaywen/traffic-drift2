TMA 结果诊断：只读完成的 pilot_v1 输出，不训练、不推理、不读原始大压缩包。

服务器前台执行：
cd /data/xjw/traffic-drift2
conda activate traffic-ncde
git pull --ff-only origin main
python Experiment/tma_analysis_v1/analyze.py

默认 seed=0；可用 --seed 1 和 --root 指定其他已完成结果。
返回 /data/xjw/tma-care-data/pilot_v1/analysis_v1/seed_0/REPORT.txt 或粘贴终端输出。
依赖现有 numpy/pandas。读取小型缓存中的记录 ID/标签和保存的预测，不需要 GPU。
程序校验消费的文件哈希，重算混淆矩阵、主要指标、崩溃/恢复计数，并核对查询记录和预算。
不修改任何原始结果，也不改变 tma_pilot_v1 中的代码，保留其缓存/已完成种子的复用能力。
输出逐月成对差值、崩溃设备、各采样策略的真实查询覆盖和主要误分类去向。
报告使用未来标签，仅供离线诊断，不是在线监测器。它不能单独决定新方法或证明因果。
