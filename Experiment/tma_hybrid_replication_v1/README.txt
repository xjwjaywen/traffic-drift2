固定 hybrid CARE 的多修复种子复现

目标
固定 source seed 0、查询混合比例 500/500、自适应阈值和四个月的实验代码。
repair seeds 0--4 只重复查询、历史参考样本与头训练随机性。
这不是新的时间盲测，也不是五个独立数据集或五次源模型训练。

在服务器的 traffic-ncde 环境运行：
  cd /data/xjw/traffic-drift2
  git pull origin codex/tma-selective-repair
  python -u Experiment/tma_paper_v1/run.py --repair-seeds 0,1,2,3,4
  python -u Experiment/tma_hybrid_replication_v1/replicate.py --repair-seeds 0,1,2,3,4

第一条实验命令复用已有的 paper repair 0，生成缺失的 repair 1--4；
第二条依次运行 hybrid repair 0--4，已完成的 seed 会校验并复用，然后汇总。
每月仍只用当月 1000 条付费标签。脚本不修改旧输出或原始缓存。
中断后重复原命令即可续跑。device/threads 参数必须与 paper 输出一致。

汇总输出：
  /data/xjw/tma-care-data/pilot_v1/hybrid_replication_v1/source_0/aggregate_0_1_2_3_4/REPORT.txt
  summary_mean_sd.csv       各方法按月均值、样本标准差、有效 seed 数
  paired_by_seed.csv        每 seed、每月的配对差值
  paired_mean_sd.csv        配对差值的均值、标准差、macro-F1 正增益 seed 数
  focus_class_mean_sd.csv   设备 5/22/26/37/20 的 recall/F1 汇总
  relaxation_frequency.csv  每类被判定需要放松保护的 seed 次数

只重做已完成结果的汇总：
  python -u Experiment/tma_hybrid_replication_v1/replicate.py --repair-seeds 0,1,2,3,4 --aggregate-only

请返回 REPORT.txt；若要看具体分歧，再返回 paired_by_seed.csv 中
hybrid_adaptive_minus_hybrid_care 的行。

解释边界
hybrid_adaptive 与 hybrid_care 在同 seed 内共享标签，可比较训练差异。
hybrid_care 与 paper 的 badge_care/random_care 查询身份不同，反映查询策略差异。
所有四个月已被看过，seed 重复只能检查稳定性；投稿前仍须在冻结规则后
用未看过的时期或独立 IoT 数据验证。
