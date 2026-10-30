TMA 五个修复种子的配对差值、胜负与设备查询覆盖

本包只读取已经完成的 paper_protocol_v1 结果及 pilot_v1 缓存。
不导入训练代码或 PyTorch，不加载模型、不推理、不训练、不下载、不读取原始压缩包。
只需现有环境的 numpy/pandas；保留全部 CESNET、TMA 缓存、模型和原报告。

服务器前台运行
cd /data/xjw/traffic-drift2
conda activate traffic-ncde
git pull --ff-only origin main
python -u Experiment/tma_paired_analysis_v1/analyze_pairs.py

默认 source seed 0，repair seeds 0,1,2,3,4。所有五个完成记录必须存在，缺失时停止，
不会自动训练或静默丢掉缺失种子。屏幕显示每个种子的读取进度及最终 REPORT。
最后应看到 PAIRED ANALYSIS COMPLETE。
可用 --root /其他路径/pilot_v1 指定数据目录；--source-seed 和 --repair-seeds 可显式
指定已完成的其他源/修复种子。默认分析所有四个月，不选择有利月份。

新输出目录（独立于原训练结果）
/data/xjw/tma-care-data/pilot_v1/paired_analysis_v1/source_0/repairs_0_1_2_3_4/

先把这个报告贴回来：
cat /data/xjw/tma-care-data/pilot_v1/paired_analysis_v1/source_0/repairs_0_1_2_3_4/REPORT.txt

需要逐设备配对统计时，再返回：
cat /data/xjw/tma-care-data/pilot_v1/paired_analysis_v1/source_0/repairs_0_1_2_3_4/device_pair_wins.csv

输出文件
- paired_by_seed.csv：每种子、月份、选择器、指标的 CARE / target-only 值与差值。
- paired_wins.csv：配对差均值、样本标准差、最小/最大值、CARE 胜/平/负、缺失数。
- device_pairs_by_seed.csv、device_pair_wins.csv：关键设备逐种子 recall/F1 配对及胜负。
- query_coverage_by_seed.csv：三月设备5/22/26/37、四月设备20，每种子每选择器的
  候选数、查询数、均匀随机下期望查询数、查询率、日期覆盖、8--14日查询数、
  首末查询UTC时间、首次测试时间间隔、测试样本数和对应 target/CARE 未来召回/F1。
- query_dates.csv：上述设备每天的候选/查询数量；完整保留1--14日的零计数。
- eval_dates.csv：上述设备下半月测试缓存每日计数；跨种子/选择器不重复累加。
- device_recovery_counts.csv：逐设备退出崩溃区间（recall>=.1）、达到恢复阈值
  （recall>=.3）和失败（recall<.05）的运行次数，以及召回均值/标准差。
- REPORT.txt：主要配对表、恢复次数、每种子的设备查询日期与未来召回。
- inputs.json、config.json、complete.json：已读取输入的哈希、配置与输出完成记录。

统计口径
1. 差值始终是 CARE - target-only，F1/召回/NFR转换为百分点，类别计数用类。
   判断胜负时 F1/召回越高越好，NFR/剩余崩溃/新增崩溃越低越好。
   浮点平局容差为原始指标单位1e-12，独立于显示精度。
2. 先计算每个种子的配对差，再计算这些差的均值/样本标准差；不把两个标准差相减。
   无崩溃组月份的 collapse_f1 是缺失，不计作0、不计平局。
3. 每个选择器保存一份供 target-only/CARE 共用的查询记录。本包核验记录ID、标签、
   已知/未知类标记与两个方法的标签成本，不重新选择查询。Margin无target对照，
   只展示其覆盖和恢复；target列缺失，不解释为零召回。
4. 验证输入完成标记、文件哈希、相同源模型/协议/环境、固定的诊断类别组，
   从逐类结果核对总体/崩溃组/高召回组F1和剩余崩溃数，并对照测试缓存支持数。
   本包不重新计算模型预测；原训练结果中的预测核验流程仍是其来源。
5. 查询池真实标签仅用于离线诊断；日期覆盖不证明代表性，不用于重新调参。
   候选/测试计数来自抽样缓存，不是全量流量，零计数不能证明设备离线。
   三月同时变化的设备不应当作彼此独立的漂移事件。
6. 五个修复种子固定同一源模型和同一时间切分；胜负次数不是统计显著性检验，
   不能从五次重复推断五个独立部署环境。所有未来月份已被查看。
7. target-only/CARE 训练多重集不同，更新数可能不同；本分析不单独归因 CE 或 KD。

重复运行
完整报告会校验后复用，并再次验证之前消费的输入文件；不重写旧训练结果。
中断后可原命令重跑；只会重新整理这个新报告。配置/代码/输入改变时拒绝混写。
输出目录有锁，避免两个分析进程同时写入。文件损坏时停止，不自动修复原实验。

合成数据测试（不训练模型）
python -m unittest discover -s Experiment/tma_paired_analysis_v1 -p 'test_*.py' -v
