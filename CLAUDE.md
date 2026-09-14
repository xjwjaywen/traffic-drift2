# traffic-drift — 项目指南

> **重要：Claude 必须自主维护本文件。** 架构或约定变化时更新，保持简洁。

## Git 信息

- Remote: git@github.com:xjwjaywen/traffic-drift2.git
- 默认分支: main

## 任务生命周期

你收到任务后，按以下 9 步流程自主完成：

1. **领取任务** — 你已被分配任务，阅读本文件和项目代码理解上下文
2. **创建工作区**:
   - `git fetch origin`（如有 remote）
   - `git worktree add -b task-<简短描述> .claude-manager/worktrees/task-<简短描述> origin/main`
   - 进入 worktree 目录工作（后续所有操作在 worktree 中）
   - 如果 worktree 创建失败，直接在当前分支工作
3. **实现功能** — 编写代码，确保可运行
4. **提交代码** — `git add` + `git commit`，commit message 简洁描述改动
5. **Merge + 测试**:
   - `git fetch origin && git merge origin/main`（集成最新代码，如有 remote）
   - 运行测试（如有测试命令）
6. **自动合并到 main**（如有 remote）:
   - `git fetch origin main`
   - `git rebase origin/main`，如果冲突则自行 resolve
   - 如果成功：`git checkout main && git merge <task-branch> && git push origin main`
   - 如果这一步有任何失败，退回到步骤 5 重试
   - （纯本地项目跳过本步）
7. **标记完成** — 更新文档（必须在清理之前，防止进程被杀时状态丢失）
8. **清理** — 回到项目根目录:
   - `git worktree remove .claude-manager/worktrees/<worktree名>`
   - `git branch -D <task-branch>`
   - 如有 remote: `git push origin --delete <task-branch>`
9. **经验沉淀** — 在 PROGRESS.md 记录经验教训（可选）

### 冲突处理

rebase 发生冲突时：
1. 查看冲突文件: `git diff --name-only --diff-filter=U`
2. 逐个解决冲突
3. `git add <resolved-files> && git rebase --continue`
4. 如果无法解决: `git rebase --abort`，退回步骤 5

### 状态判断

- 通过 `git remote -v` 判断是否有 remote
- 有 remote → 必须完成步骤 6（merge + push）
- 无 remote → 跳过步骤 5 的 fetch、步骤 6 和步骤 8 的远程分支删除

## 文件维护规则

> **以下文件都由 Claude Code 自主维护，每次功能变更后必须同步更新。**

- **CLAUDE.md**（本文件）：架构、约定、关键路径变化时更新，只改变化的部分，保持简洁
- **README.md**：面向用户的文档，功能、使用流程变化时同步更新，保持与实际代码一致
- **TEST.md**：测试指南，新增功能时同步添加测试用例和文档
- **PROGRESS.md**：见下方「经验教训沉淀」

## 测试规范

**开发时必须主动使用测试，不是事后补充！**

- **改代码前**：先跑测试，确认基线全绿
- **改代码后**：再跑一遍确认无回归
- **新增功能**：同步新增测试用例，更新 TEST.md
- **修 bug**：先写复现 bug 的测试（红），修复后确认变绿

## 经验教训沉淀

每次遇到问题或完成重要改动后，要在 PROGRESS.md 中记录：
- 遇到了什么问题
- 如何解决的
- 以后如何避免
- **必须附上 git commit ID**

**同样的问题不要犯两次！**

## KBS 补充实验约定

- 入口：`Experiment/core_code/scripts/run_kbs_supplement.sh`；协议与服务器命令见 `Experiment/core_code/KBS_SUPPLEMENT.md`。
- 新受控消融固定 optimizer 步数和目标抽样流，参考 CE 与参考 KD 独立开关；不得将新协议结果拼入旧主表。
- 缓存/运行必须记录数据与代码标识，完成标记最后写入；配置不一致时使用新输出目录，不覆盖已有完成结果。
- 合成测试仅验证代码，不能作为真实实验结果；测试命令见 `TEST.md`。
- `badge-kd` 补跑通过独立 `kbs_badge_kd_followup.py` 注册配置和汇总，保留 v1 数值引擎文件及其哈希；先校验已有 BADGE 基线/选样，再新增五次关闭参考 CE 的运行。其汇总仅计入通过 ID、抽样流和评估掩码配对检查的种子，独立输出 `badge_kd_*.csv`。
- `run_kbs_badge_controls.sh` / `kbs_badge_controls.py` 补齐 BADGE 四格组件对照：已完成 full/KD 是前置条件，只新增 FT-only 和参考 CE 两配置、默认五种子十次头部训练。复用原引擎与全部查询，保留固定 CE 权重和抽样流；独立 `badge_controls_*` 报告仅含四组完整配对种子，重算共同排除后的指标及逐类损伤。`BADGE_CONTROL_SEEDS` 不继承旧任务的 `SEEDS`/`STEPS`；完成后续跑不加载特征。协议/原运行损坏时失败，不绕过哈希或补造基线。详见 `KBS_BADGE_CONTROLS.md`。
- `class-audit` 用标准库核验现有六配置逐类指标/小文件并写入派生 `class_audit/`，不重跑训练；阈值来自原协议。`stage2` 先分析五种子，再执行已有三种子 Margin 敏感性；分析失败立即停止。`AUDIT_SEEDS` 与训练 `SEEDS` 独立。
- 敏感性比较使用 `sensitivity_matched_*.csv`，各配置严格使用同一组种子（默认 0–2）；旧引擎的通用汇总会额外纳入两个基线的种子 3–4。`sensitivity-report` 校验完成文件后单独生成配对种子汇总，不修改 v1 引擎。

## 工作区注意事项

- 选样探索独立入口：`run_kbs_acquisition_pilot.sh` / `kbs_acquisition_pilot.py`，协议见 `KBS_ACQUISITION_PILOT.md`。输出必须与旧 study/cache 分开；不修改 v1 引擎。原型复用每类五个参考行，风险仅用预测频率下降代理。先在不传入目标标签的 API 中锁定所有选择，再另进程事后评估；选样结果不等于修复收益。未来训练必须重算六方法查询并集排除后的共同评估。
- `run_kbs_acquisition_diagnosis.sh` 为 pilot 的独立事后诊断，保持 pilot 文件哈希不变。参考评估排除原型构建行，重建目标候选池并拆分探索/专项/补足，结果与原选样评价逐项对账。只统计已存 BADGE 前 20% 标签中的真实错分作为下一假设的可行性线索；没有第二阶段查询或训练。
- `run_kbs_pool_control.sh` / `kbs_pool_control.py` 在原风险候选池内做 uniform / score-only / distance-only 对照，冻结池、探索、预算及数值设置；先逐行复现原分数×距离专项选择再继续。保留 pilot/diagnosis/v1 引擎文件哈希，先完成全种子无目标标签选择再另进程评价。输出独立 `kbs_pool_control_v1`；未来修复使用原六方法加新三方法查询并集排除。没有训练或创新性结论。
- `run_kbs_relation_audit.sh` / `kbs_relation_audit.py` 复用 BADGE full/KD 逐样本结果进行 CPU/NumPy 预检。独立进程先只用已查询真值与旧预测冻结全部种子的纠正关系，再评价原修复、关系限制和出度匹配随机目的类限制；不读取特征或训练模型。原 common 查询并集排除及逐类指标必须对账，所有输入保持不变。输出独立 `kbs_relation_audit_v1`；同时报告保留正翻转、阻止负翻转、逐类可纠正覆盖及旧错→另一错。完整目标标签和崩溃类别清单只用于评价，不参与关系构建；本轮不是新方法或创新性证明。

- 在 worktree 中工作时，不要切换到其他分支
- 完成任务后确保代码可运行、测试通过
