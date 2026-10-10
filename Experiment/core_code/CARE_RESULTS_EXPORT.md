# 将服务器上的 CARE 结果保存到 Git

本工具只读取七个已知实验目录：小规模修复、完整池审计、目标保护、混合查询消融、M12 新查询种子复验、M11 跨月复验和 M11 恢复/误伤诊断。默认导出其中已经完整完成的实验；未完成/不存在的目录会明确跳过。损坏的完成结果立即报错，不静默跳过。

保存原始报告、完成阶段的汇总/逐类 CSV、JSON 协议、冻结查询和训练记录，文件字节不改写。逐行边界探针 `geometry/probe_rows.csv` 留在服务器，其汇总和哈希仍导出。先沿最终阶段的本实验完成引用检查 `complete.json`、`resolved_config.json` 和全部登记文件的 SHA256，包括留在服务器上的预测数组/头参数。不读取原始数据、特征缓存或参考张量，不重新计算指标，不验证外部父实验或缓存的当前文件。哈希说明文件完整性，不代表研究结论正确。

Git 目录为 `Experiment/results/care_development/`，原始输出仍在 `Experiment/core_code/outputs/`。`.pt/.npy/.npz/.h5` 等二进制不导出，完成记录保留它们的哈希；文本子集不能用于断点续跑。导出清单保存来源路径、源工作区 Git HEAD/分支、导出脚本哈希、文件大小、跳过项和已检查文件。代码可能通过 `git restore` 单独更新，所以源 HEAD 不能代替原协议里的数值脚本哈希。

文本单文件限制 25 MiB，总计限制 100 MiB；超过会停止，可用 `--studies` 选取较少实验。完成文件未登记的文件不收集，符号链接/越界路径拒绝。每次快照独立保存在 `outputs/care_result_exports/<UTC时间-随机ID>/`，另生成可传输 ZIP。

## 当前服务器终端前台上传

只更新本次入口，不切换当前分支或覆盖已有数值脚本：

```bash
cd /data/xjw/traffic-drift2 &&
conda activate traffic-ncde &&
env -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    git -c http.proxy= -c http.https://github.com.proxy= \
        -c remote.origin.proxy= fetch origin codex/small-repair-diagnosis &&
git restore --source=FETCH_HEAD --worktree -- \
    Experiment/core_code/scripts/export_care_results.py \
    Experiment/core_code/scripts/run_export_care_results.sh &&
bash Experiment/core_code/scripts/run_export_care_results.sh publish
```

`publish` 先导出，再重新 fetch 指定代码分支，在快照目录建立独立 Git worktree 和新分支 `codex/server-results-<UTC时间-随机ID>`；只提交指定结果目录并普通 push。不会更改原工作区的 HEAD、索引、代码改动，不自动合并到 main，不 force push。服务器需已有 Git 提交身份和 GitHub 写权限；沿用用户自己的 Git 配置，不保存或索取凭据。网络操作临时移除之前失效的代理环境/配置，不改全局配置。

成功必须出现经过 `ls-remote` 对账的标记：

```text
CARE_RESULTS_PUSHED: codex/server-results-... <完整commit>
```

将这行分支名提供给本地分析者即可 fetch 原始表格。看到 `CARE_RESULTS_EXPORTED` 只代表服务器已生成快照，不能当作 GitHub 上传成功。

若提交身份或推送认证失败，ZIP/已创建 worktree/已完成提交会保留，显示 `CARE_RESULTS_UPLOAD_NOT_CONFIRMED`。先通过 Git 自己的认证方式配置身份/权限，然后在显示的 worktree 中重试普通推送；不要在聊天中发送密码或 token。也可以通过 ToDesk 文件传输将打印的 ZIP 传到 Mac，解压后由 Mac 上传。独立 worktree 可在确认远端提交后用 `git worktree remove <显示路径>` 清理；ZIP 和导出快照保留。

## 只导出或选择实验

只导出，完全不创建分支或进行网络操作：

```bash
bash Experiment/core_code/scripts/run_export_care_results.sh export
```

只上传最近两轮原始结果：

```bash
bash Experiment/core_code/scripts/run_export_care_results.sh publish \
    --studies care_query_seed_robustness_v1,care_m11_replication_v1
```

显式选择要求所有指定实验完成。`--outputs-dir` 可指定输出所在目录，`--repo-dir` 可指定已有 Git 仓库；目录名必须是工具列出的已知实验。使用 Python 标准库即可，不需要 GPU、torch 或新装包。

M11 机制诊断完成后再次运行 `publish`，新快照会包含它：

```bash
bash Experiment/core_code/scripts/run_care_m11_mechanism_diagnostic.sh run &&
bash Experiment/core_code/scripts/run_export_care_results.sh publish
```

这些实验仍是方法开发证据，需按各原始协议的评价排除集和统计单位解释，不能混用旧表的不同评价集合。
