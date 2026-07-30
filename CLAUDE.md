# CLAUDE.md — CrossPiezo / PULSE 项目规则

## 必读

每次开始工作前依次读取：

1. `CROSSPIEZO_TASKBOOK.md`
2. `PiezoProtocol_Draft_v0.1.tex`
3. `reports/open_questions.md`
4. 当前 Phase 的报告和配置

## 永久规则

- `E:/DATA` 是只读外部数据根目录。
- 不修改 PiezoJet、GaugeFlow、T2C-Flow、EviMem-RL 或其他外部项目。
- 不访问冻结测试标签进行开发。
- 不以化学式匹配代替结构匹配。
- 不静默转换单位、Voigt、符号、剪切或坐标基。
- 所有转换写入 transformation history。
- 不把 JARVIS/MP 平均张量称为真实张量。
- 不在 feasibility gate 前训练完整 PULSE。
- 不在结果生成前填写 LaTeX 的结果性 `TBD`。
- 不联网下载数据，除非用户在当前会话明确批准。
- 不复制大型数据到当前仓库。
- 不把第三方数据提交到 Git。
- 失败和 quarantine 记录必须保留。
- 所有 split、配置、manifest 和报告必须 hash 绑定。
- 每个 Phase 完成后运行测试并生成报告。
- Failed Gate 不得通过增加模型复杂度绕过。
- 完成 Phase 4 后停止，等待人工确认。

## 编码规则

- Python 3.11+。
- 使用 `pathlib`，兼容 Windows 路径。
- 公共 API 有类型注解和 docstring。
- 核心 schema 使用 Pydantic v2。
- 不允许 `except Exception: pass`。
- 不允许 silent fallback。
- 大文件使用分块或流式读取。
- 任意随机过程固定 seed。
- 图表和表格只能由版本化脚本生成。
- 不在代码中硬编码 `E:/DATA`；路径来自配置或 CLI。
- 默认操作必须安全且可 `--dry-run`。

## 科学声明规则

以下词语只有在证据通过后使用：

- `protocol uncertainty floor`
- `limits AI screening`
- `leaderboard inversion`
- `soft-mode amplification`
- `robust discovery`
- `calibrated`
- `generalizes`

没有第三协议或实验时，不使用：

- `true tensor`
- `ground truth consensus`
- `experimentally validated`
- `physically exact ranking`

## 当前执行边界

只执行任务书 Phase 0–4。不得自动进入：

- PULSE 全模型；
- SOTA 大规模复现；
- 新 DFT；
- 大规模下载；
- 论文结果改写。
