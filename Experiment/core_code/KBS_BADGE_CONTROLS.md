# BADGE 固定查询组件对照

本轮补齐受控 v1 协议下 BADGE 的四格组件对照。复用已完成的 `badge_full`、
`badge_kd` 和保存的查询；只增加两种配置，各五个种子，共 **10 次头部训练**。
不重新训练源模型、不重新选样、不重新提取特征。

| 配置 | 目标 CE | 参考 CE | 参考 KD | 本轮 |
|---|---|---|---|---|
| badge_ft_only | 开 | 关 | 关 | 新增 5 次 |
| badge_replay | 开 | 开 | 关 | 新增 5 次 |
| badge_kd | 开 | 关 | 开 | 校验并复用 |
| badge_full | 开 | 开 | 开 | 校验并复用 |

默认仍为 M4 参考、M12 目标、1,000 个目标标签、每类 5 个参考样本、
1,380 次 AdamW 更新、batch 64、学习率 0.001、weight decay 0.0001、8 个线程。
KD 开启时权重 0.5、温度 2。所有配置使用同一源头部、同一批目标/参考行及其顺序、
相同的目标与参考抽样随机流。冻结编码器，仅更新原线性头部。

目标 CE 和参考 CE 的权重沿用已完成 study 的名义混合权重，关闭损失不重新归一化。
这是固定优化设置的组件控制，不能解释为分别调参后各配置的最优成绩。
FT-only 仍经过原引擎的参考缓存加载，但参考标签、参考 KD 均不进入其损失；
因此该实现不能用于声称 FT-only 的最小内存占用或 I/O 成本。

## 服务器运行

先在仓库内更新代码并激活原环境。下面的 Git 命令只为这次拉取取消之前失效的代理。

```bash
cd /data/xjw/traffic-drift2
conda activate traffic-ncde
env -u http_proxy -u https_proxy -u all_proxy \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  git -c http.proxy= -c http.https://github.com.proxy= \
  -c remote.origin.proxy= pull --ff-only origin main
```

可先查看计划，随后预检已完成 full/KD、协议和现有缓存：

```bash
bash Experiment/core_code/scripts/run_kbs_badge_controls.sh plan
bash Experiment/core_code/scripts/run_kbs_badge_controls.sh preflight
```

通过后后台运行。`run` 本身也会核验，所以不必重复预检。

```bash
nohup bash Experiment/core_code/scripts/run_kbs_badge_controls.sh run \
  > kbs-badge-controls.log 2>&1 &
tail -f kbs-badge-controls.log
```

同一输出目录只允许一个训练写入进程。Ctrl+C 退出 `tail` 不停止后台任务。
中断后重新执行相同 `run` 命令：完整且通过哈希校验的运行会跳过；未完成的那次从头开始，
不从半程优化器状态恢复。所有四组均完成后重跑只核验并汇总，不再加载特征缓存。

默认 `OUTPUT_DIR=outputs/kbs_supplement_v1`，`CACHE_DIR=$OUTPUT_DIR/cache`，
路径相对于 `Experiment/core_code`。支持 `CONFIG`、`CHECKPOINT`、`DATA_DIR`、
`DEVICE`、`GPU_ID`、`PYTHON`。种子单独用 `BADGE_CONTROL_SEEDS`，默认 `0,1,2,3,4`；
不继承其他任务的 `SEEDS` 或 `STEPS`。额外 Python CLI 参数原样转发，但配置必须匹配已有 study。
不要为了跳过校验创建新的空 study 或修改已有配置/完成标记。

## 结果读取

最后应出现 `Verified BADGE controls: 5/5 seeds x 4 configurations`。
`badge_controls_status.json` 的 `all_requested_complete: true` 表示请求的全部种子完整；
检查 `requested_seeds` 确认为 0–4。报告文件存在本身不代表已跑完。

```bash
cat Experiment/core_code/outputs/kbs_supplement_v1/badge_controls_report.md
cat Experiment/core_code/outputs/kbs_supplement_v1/badge_controls_paired_by_seed.csv
```

需要重新汇总时执行（不读取特征/原始数据，不训练）：

```bash
bash Experiment/core_code/scripts/run_kbs_badge_controls.sh summarize
```

独立输出前缀 `badge_controls_`：

- `report.md`：中文报告，静态基线、四配置和配对差值。
- `summary.csv` / `results_by_seed.csv`：四组同一批完整种子的均值、样本标准差和逐种子值。
- `paired_summary.csv` / `paired_by_seed.csv`：KD−FT、重放−FT、full−重放、full−KD、full−FT。
- `per_class.csv`：所有类别、每个配对种子的支持数、召回/F1 和损伤，保留零支持类别。
- `status.json`：请求/完成/缺失种子及派生报告的文件哈希，报告写完后最后发布。
- `extension_manifest.json`：本扩展及原训练实现、协议的身份记录。

四组必须核验目标/参考 ID 顺序、目标真值、抽样流、损失开关及预测数组对齐。
严格评估排除 BADGE 查询；common 评估排除原 Margin 和 BADGE 查询并集，
并从逐样本预测重算整体与逐类指标。只汇总四组都完整的相同种子，不以五种子基线对比三种子新配置。
原 `primary_*`、`badge_kd_*`、敏感性结果及其他探索输出不会被改写；原数值引擎文件保持不变。

报告同时提供新增/残留崩溃、非崩溃类退化、最差类别变化及正/负翻转计数。
平均 F1 提升不能替代损伤检查。训练耗时不含缓存提取/选样，不是端到端维护成本。
共用一个源模型和 M12 的五个种子也不是五个独立时间/环境验证。

本轮回答参考 CE/KD 是否改善固定查询下的修复及保持能力；不自动选获胜配置，
也不证明新增方法创新性。M12 已用于开发，后续变更方法需要冻结方案后另行验证。
监测触发与定期维护的连续时间实验不包含在此次运行中。
