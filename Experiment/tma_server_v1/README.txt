TMA 服务器验证：第零阶段，下载与数据检查

目标仓库 /data/xjw/traffic-drift2，环境 traffic-ncde，GPU NVIDIA L20。
此阶段仅下载作者公开数据并检查完整性、CSV 字段、时间和设备覆盖。
尚未实现训练，不会给出漂移或崩溃的模型结论。
默认数据目录 /data/xjw/tma-care-data；不依赖旧 CESNET 缓存。
数据只在 Linux 服务器下载；本机验证只使用人工构造的小型测试夹具。

更新服务器仓库后运行：
cd /data/xjw/traffic-drift2
git pull --ff-only origin main
conda activate traffic-ncde
bash Experiment/tma_server_v1/run.sh preflight
nohup bash Experiment/tma_server_v1/run.sh all > tma-stage0.log 2>&1 &
tail -f tma-stage0.log

依赖 requests numpy pandas scikit-learn torch，预检会显示版本和 GPU。
缺包时先确认当前环境，再按报错安装缺少的包，不要重装 torch/CUDA。
退出 tail 不会停止后台任务。也可分别运行 download / audit。
首次下载约 2.06 GB 流特征压缩包和设备映射；校验 gzip 后保留原压缩包，
审计直接流式读取，无需解压生成更大的 CSV。需要至少 4 GiB 可用空间。
未完成传输重试最多三次，每次从头开始；完整文件凭本地 SHA256 复用。
SHA256 用于后续完整性验证，不能冒充发布者提供的校验和。
日志若出现登录/配额/HTML 错误会失败，不会把网页当成数据。

完成后返回以下输出：
cat /data/xjw/tma-care-data/audit_v1/REPORT.txt
cat /data/xjw/tma-care-data/audit_v1/data_audit.json
以及 monthly_class_counts.csv、weekly_class_counts.csv。
audit_complete.json 最后写入；失败或仅有报告文件不代表完整完成。

数据来源：
https://github.com/DADABox/revisiting-iot-device-identification
https://github.com/DADABox/revisiting-iot-device-identification/blob/master/data/README.md
https://arxiv.org/abs/2107.07818

后续根据审计固定时间划分、来源类别和重复会话处理，再实现模型实验。
拟比较静态源模型、随机采样、Margin/BADGE、目标微调、参考蒸馏、完整 CARE，
保持每个对照的标签预算、参考样本和未来评估集一致；源模型与修复种子分别记录。
保留强简单分类器，检查原始就学不好的类别，禁止把这些类别当成漂移崩溃。
模型训练与修复必须在服务器进行；本包未包含或宣称实验结果。
