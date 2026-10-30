TMA 原稿 CARE 协议核对与补跑 v1

用途
保留 CESNET 和现有 TMA 结果，复用已完成 pilot_v1 的抽样缓存、预处理和源模型。
本包不下载数据、不读原始压缩包、不重新训练编码器。结果只写入新目录：
  /data/xjw/tma-care-data/pilot_v1/paper_protocol_v1/source_0/
运行依赖与首轮一致，使用 traffic-ncde；无需更新 PyTorch/CUDA 或安装新依赖。
真实数据仅允许在 Linux 服务器运行；Mac 测试只使用人工合成夹具。

第一次运行（全部前台，直接显示进度）
cd /data/xjw/traffic-drift2
conda activate traffic-ncde
git pull --ff-only origin main
python -u Experiment/tma_paper_v1/run.py

默认先重算旧结果的共同分组指标，再用现有 source seed 0、repair seed 0 补跑：
  Random-target-only、Random-CARE、BADGE-target-only、BADGE-CARE、Margin-CARE
  2020-01、02、03、04 共 20 个分类头训练。
静态 MLP/RF 复用已保存预测。所有方法均用静态 MLP 定义的同一崩溃/高召回组。
不启动后台任务。每个方法开始、每个月完成时有进度；分类头内部不逐 batch 打印。
最后应出现 PAPER REPAIR COMPLETE 和 AGGREGATE 路径。

只重算旧指标、不训练（可选，默认完整运行已包含此步骤）
python -u Experiment/tma_paper_v1/run.py --audit-only

seed 0 核对完成后重复修复种子（先不要为了看更好的成绩筛选种子）
python -u Experiment/tma_paper_v1/run.py --repair-seeds 0,1,2,3,4
固定 source seed 0，复用已经完成的 repair 0，补跑 repair 1--4。
repair 1--4 仅改变查询、历史参考抽样和头训练随机性，不生成 pilot_v1/seed_1 等源模型。
每个修复种子内部，相同选择器的 target-only 与 CARE 使用相同查询。
不同种子重新选取每类五个参考，跨月复用该种子的参考；每月独立重置原分类头。
--source-seed 可指定其他已完成 pilot 源模型；与其同号的 repair seed 复用该源轮次
原有查询/参考，其余 repair seeds 在同一固定源模型上重新选样。

数据位置不是默认路径时：
python -u Experiment/tma_paper_v1/run.py --root /你的路径/pilot_v1
--device cuda、--threads 8 为默认值；CUDA 不可用会停止，不会静默切换 CPU。

输出
source_0/audit/
  legacy_metrics_common_groups.csv：旧首轮/已完成第二轮的共同 MLP 分组重统计。
  legacy_per_class_common_groups.csv：逐类结果及单次恢复等级。
  reference_sources.csv：历史参考来源与回退数量。
  REPORT.txt、inputs.json、complete.json。
source_0/repair_0/
  REPORT.txt：本轮主要结果、口径与边界。
  summary.csv：四个月 *（五个修复方法＋两个静态基线）=28行。
  per_class.csv、paired_deltas.csv、reference_ids.csv。
  每月子目录含查询ID、分类头、训练次数追踪、预测、混淆矩阵与完成标记。
  paired_deltas.csv：CARE 与 target-only 差值，以及同查询的新旧协议差值；
  新旧对照仅在 repair_seed=source_seed 时输出，避免把不同查询当成配对实验。
source_0/aggregate_0/ 或 aggregate_0_1_2_3_4/
  summary_mean_sd.csv：固定源模型的修复种子均值、样本标准差和有效计数。
  collapse_recovery_from_mean_recall.csv：先平均逐类召回，再按 .05/.30 划分恢复等级。
  单种子标准差为空，无崩溃组月份的 collapse_f1 为空；静态行重复不代表重训了源模型。

先返回以下文件内容即可分析，不需要上传原始数据或模型：
cat /data/xjw/tma-care-data/pilot_v1/paper_protocol_v1/source_0/repair_0/REPORT.txt
cat /data/xjw/tma-care-data/pilot_v1/paper_protocol_v1/source_0/repair_0/paired_deltas.csv
cat /data/xjw/tma-care-data/pilot_v1/paper_protocol_v1/source_0/audit/reference_sources.csv

训练协议
- paper_engine.py 的 build_head_training_set / fit_head 逐字提取自原始主方法，
  provenance.json 记录来源提交及函数 SHA256；没有调用旧 fit_controlled。
- 实际已知类查询 Bk 份，每份重复 rho=2；CARE 再加入每类5个参考，各出现1次。
- 每轮打乱混合多重集，30 epochs，batch64，AdamW lr=.001、wd=.0001。
- CARE 在每个更新步独立有放回抽取参考 KD batch，大小等于该步实际 CE batch，
  KL 包含 T^2，T=2、权重.5，teacher 为原源头；编码器冻结。
- TMA 若 Bk=1000、38类，CARE 混合2190项、1050次更新；目标CE与参考CE
  样本混合权重约 .913/.087。未知查询扣预算但不训练，按实际 Bk 记录权重/步数。
- target-only 也是两份目标、30 epochs，但没有参考/KD，样本数不同导致更新数可能不同。
  这是简单方法对照，不是固定优化步数的 CE/KD 因果消融。
- 使用原仓库 top-3/PCA BADGE 近似，不切换完整标准 BADGE，不加入 hybrid。
- 保留原 TMA 时间切分、特征、参考期不足时的训练期回退，以及未来测试。
  这是设备识别修复适配，不宣称所有流量加密或完整五信号监测已验证。

指标修正
- initial_collapse_count：静态 MLP 崩溃组大小。
- remaining_collapse_count：该组修复后 recall<.1 的类数。
- high_recall_group_f1：历史和目标静态 recall 均>=.8的组，不等于原论文固定20类稳定组。
- negative_flip_rate_vs_static_mlp：静态 MLP 原本正确的已知类预测被当前方法改错的比例。
  对 RF 也采用同一基准，避免暗中切换分母。
- new_collapses_among_historically_learned：在历史学好且支持足够的集合内新增的崩溃。
- 原 TMA 支持>=50、历史recall>=.8的条件保留，缺席类不算崩溃，新类单列。
- recovery_status_per_run 是单次 .05/.30 等级，跨种子正式等级见 aggregate 文件。
- 已看过全部未来月份；本轮是恢复原稿协议，新种子不是新的独立时间盲测。

保护与恢复
- 不改 pilot_v1/*.py，避免旧缓存实现哈希失效；不覆盖 seed_0、controls_v1 或 CESNET。
- 每次校验原始输入的哈希、包版本、源模型类别/特征映射；重新推理的源预测必须一致。
- 月完成标记校验通过可跳过；Ctrl+C 后原命令前台重跑，最多重做未完成的那个月。
- 完成结果被改动或配置/代码改变时拒绝混写。保留原结果，按新的版本目录组织后续变更。
- 同一 source 输出有文件锁，不能并行启动两个本包进程。
- 第二轮 controls_v1 仅在其 complete.json 存在时读入审计；缺少它不阻挡原稿协议补跑。

本地验证（只使用合成数据，不读取真实数据）
python -m unittest discover -s Experiment/tma_paper_v1 -p 'test_*.py' -v
测试覆盖：原函数逐字/哈希一致、相同训练轨迹、混合比例/步数/末尾batch、
冻结源模型、无可用标签、剩余崩溃/共同组、未知类成本、旧结果逐文件不变、
完整运行、按月恢复、修复与源种子分离、配置改变/输入损坏时拒绝复用。
