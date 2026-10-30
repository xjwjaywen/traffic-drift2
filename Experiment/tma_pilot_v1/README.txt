TMA 第一轮服务器试验：静态退化与 CARE 分类头修复

仅供第一轮可行性验证，默认一个种子，不是正式论文结果。
不下载数据；读取 stage0 已校验的 /data/xjw/tma-care-data/raw/features_nov-apr.tar.gz。
CLI 只允许 Linux。Mac 上只运行人工构造的测试夹具，不能在 Mac 下载或训练真实数据。

服务器：
cd /data/xjw/traffic-drift2
conda activate traffic-ncde
git pull --ff-only origin main
python -u Experiment/tma_pilot_v1/pilot.py all

依赖 numpy pandas torch scikit-learn scipy joblib，均来自现有环境；不重装 CUDA。
默认使用 cuda、8 个 CPU 线程；可通过 CUDA_VISIBLE_DEVICES 指定 GPU。
默认输出 /data/xjw/tma-care-data/pilot_v1；支持 --data-root 和 --output-root。
可分开执行 prepare / run。prepare 流式扫描压缩包一次，按哈希抽样，不解压大 CSV。
prepare 的进度以每 100 万条原始记录打印，首次仍需扫描六千万条，耗时取决于服务器。
中断后同一命令复用已完成且校验通过的缓存/种子；未完成的数据准备或种子会重跑。
并发运行会因输出目录锁失败。配置或代码改变须使用新的 --output-root。

预先固定的协议：
1. 保留 protocol=6/17 的 TCP/UDP，时间范围 [2019-11-01,2020-05-01)。
   1970 的异常记录和 5 月部分月份不进入主实验。
   此实验是一般 IoT 设备识别，未证实每条流都加密，不能直接当作加密流量实验。
2. 11 月 1--21 日训练，每类最多 2000 个唯一记录；11 月 23--30 日验证选源模型 epoch。
   12 月 1--14 日参考池每类最多 200；12 月 16--31 日作为独立历史性能检查。
   11 月 22 日、12 月 15 日及每个目标月 15 日为整日隔离窗口。
3. 1--4 月各自独立试验：1--14 日最多 50000 条无标签候选，16 日至月底最多 200000 条未来测试。
   验证、历史检查和未来测试均按标签无关的记录哈希作全局均匀抽样，保持自然频率近似。
   训练/参考按历史真类分层抽样。未来不按类别平衡、不筛选“容易崩溃”的类别。
   完整记录的 64 位哈希用于抽样和样本内去重；不是全库重复计数，不是会话独立性证明。
   数据缺少可靠全局会话 ID，时间隔离不能证明所有长会话都已隔离；正式实验需进一步审查。
4. 类别空间仅由训练期实际 TCP/UDP 设备确定。已查询的新类仍扣标签预算，不能用于固定分类头训练。
   未来新类单列样本数和预测，不并入已知类宏指标；这是闭集修复试验，不具备开放集检测能力。
5. 使用 16 个流量统计特征（data.py FEATURES），不使用设备 ID、时间、IP、域名/域名 ID、端口。
   带符号 log1p、缺失值中位数填补、标准化仅拟合训练集，另加 16 个缺失标记，标准化后截断至 +/-10。
   不因 bytes_in 等缺失而整行删除。源 MLP 256/128、LayerNorm、线性头，30 epoch，历史验证选择。
   强简单基线：200 棵 RandomForest，叶节点至少 2 条；两者使用同一历史训练样本和预处理。
6. 每个目标月重置为同一源模型，单次查询 1000 条，不是四个月总共 1000 条，也不是累计在线维护。
   比较 static_mlp、static_rf、random_target、random_replay、random_care、margin_care、badge_target、badge_care。
   相同选择器的损失消融使用完全相同的查询、参考样本、目标批次、优化步数和未来测试集。
   每类 5 个历史参考；若 12 月上半月缺席，用 11 月训练记录补足，reference_ids.csv 记录来源。
   冻结编码器，只更新头：500 步、batch=128、AdamW lr=.001、weight_decay=.0001。
   target=目标 CE；replay=目标 CE+参考 CE；care=目标 CE+参考 CE+0.5 KD，温度=2。
   修复函数和 top-3/PCA BADGE 近似从仓库原代码逐字提取，provenance.json 记录来源和校验和。
   该 BADGE 近似不是完整按类对齐的梯度嵌入；本次不修改它以便保持既有实现。
   本包适配 CARE 的采样与修复模块，不是原论文 CNN 或完整五信号监控流程的逐项复现。
7. 主崩溃定义：历史及目标测试支持均 >=50、12 月历史召回 >=.8、目标静态召回 <.1。
   同时报告历史 >=.5 的敏感性计数。阈值预先固定；未来标签仅事后诊断，不参与采样或调参。
   无样本类别召回记为缺失而非 0，不计崩溃；样本不足和原本未学好类别保留逐类结果。
   修复后分别统计恢复到 .1/.5、稳定类退化、新增崩溃、负翻转率。
   宏指标按当期出现类计算，跨月类别集合不同，必须同时看逐类表和覆盖；不能只比较一个总分。
8. conditional_shift.csv 给出相同设备的特征 KS 效应量和缺失比例变化，不把大样本 p 值当作漂移证据。
   即使特征分布改变，也不能单独推断 P(Y|X) 改变、固件更新或其他原因。

默认 seed=0。若第一轮值得继续，固定协议后运行 --seeds 0,1,2，可复用已完成 seed_0。
不要根据未来测试优劣挑种子或只保留有利月份。若无合格崩溃也应保留并报告该负结果。

结束后返回以下文件（没有 PILOT COMPLETE 或种子 complete.json 时结果可能不完整）：
/data/xjw/tma-care-data/pilot_v1/summary.csv
/data/xjw/tma-care-data/pilot_v1/seed_0/per_class.csv
/data/xjw/tma-care-data/pilot_v1/cache/sample_counts.csv
如需判断输入分布变化，同时返回 seed_0/conditional_shift.csv。
保留 source.pt、random_forest.joblib、preprocessor.npz、查询记录、参考记录和预测文件用于复查。

验证（仅合成数据，不连接或下载 TMA）：
python -m unittest discover -s Experiment/tma_pilot_v1 -p 'test_*.py'
