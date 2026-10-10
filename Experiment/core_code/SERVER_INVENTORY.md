# 学校服务器上的脚本与数据定位

在学校服务器已激活的环境中运行。只使用 Python 标准库，默认仅在终端打印报告，
不导入 PyTorch、不读取模型张量、不启动实验、不安装依赖。
它读取目录、文件大小、缓存 manifest 和已安装包的版本；若有 `nvidia-smi`，
读取当前 GPU 状态，但不建立 CUDA 计算上下文。

从仓库根目录运行：

```bash
python Experiment/core_code/scripts/server_inventory.py --root /data/xjw
```

报告包含脚本、TLS 配置、HDF5 数据、检查点、特征缓存和符号链接目标，
并显示当前 Python、Conda 环境、包版本及服务器可用内存。
重点核对缓存目录的 `target.pt`、`reference.pt`、`head.pt` 和 `manifest.json`。
四个文件都在，只说明候选输入齐备；还须按实验协议核对哈希、时间段和标签空间。

默认不跟随目录符号链接，排除 Conda 环境、Git、Python 缓存等目录；
最多扫描八层目录、150,000 个条目和约 30 秒。每种匹配最多保存 250 条，
终端仅显示前 12 条；完整匹配数和扫描是否截断会单独显示。
若数据目录是符号链接，先查看报告中的目标路径，再用 `--root` 指向实际数据目录。

需要保存 JSON 时，指定一个新的报告文件名：

```bash
python Experiment/core_code/scripts/server_inventory.py --root /data/xjw \
  --output Experiment/core_code/outputs/server_inventory/check-1.json
```

报告目录受现有 `.gitignore` 保护，不上传服务器的目录清单或环境信息。
工具拒绝覆盖已有报告，重复检查时改用 `check-2.json`，或不指定 `--output`。

只获取检查脚本而不切换服务器的实验分支：

```bash
git fetch origin codex/server-inventory && \
  git show FETCH_HEAD:Experiment/core_code/scripts/server_inventory.py | \
  python - --root /data/xjw
```
