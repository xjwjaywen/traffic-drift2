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

进一步检查崩溃设备的特征退化、跨类相同输入和发生日期：
python Experiment/tma_analysis_v1/feature_audit.py
返回 /data/xjw/tma-care-data/pilot_v1/feature_audit_v1/seed_0/REPORT.txt。
只读缓存特征、保存的预处理参数和查询记录，不重新扫描原始压缩包、不运行模型或训练。
报告区分原始 16 个统计特征重复与标准化/截断后重复；给出同一向量对应不同设备的计数。
同向量多数类上限使用测试真标签事后计算，不是可部署分类器的性能，也不证明完整网络流不可区分。
本脚本不排除任何样本或类别、不自动改动评估协议；新方法的正式验证需要独立数据。
