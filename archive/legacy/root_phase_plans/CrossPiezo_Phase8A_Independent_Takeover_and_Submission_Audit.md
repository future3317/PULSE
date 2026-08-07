# CrossPiezo Phase 8A：独立接管与投稿审计

> 基线：`paper/screening-resolution-q1-v1` @ `6e4ebf29c18189d115dc29007933edf74bc8c582`  
> 新分支：`paper/independent-submission-audit-v1`  
> 目标：由无历史记忆的新代理独立验证 v0.6，并给出投稿或第三协议决策。  
> 禁止：新 DFT、新模型、新 endpoint、修改 P0/P2。

## A. 接管验证

1. 核对 branch、commit、工作树。
2. 运行：
   - 全部 pytest；
   - Phase 7C 独立验证；
   - manifest/hash 检查；
   - LaTeX 编译。
3. 生成 `reports/phase8a/00_takeover_audit.md`。
4. 冻结数字不一致则停止。

## B. Portfolio 主张冻结

独立读取 CSV，区分：

- 最佳点估计策略；
- 跨 fold 最稳定策略；
- paired-difference CI 显著策略；
- budget=2 coverage upper bound。

冻结一个 equal-budget 主 claim，覆盖 P0/P2 与 F1/F3/F4，不得 cherry-pick。

若没有统一优胜策略，写成：

> Rank aggregation provides a decision trade-off rather than a universally superior method.

## C. 三类审稿人攻击测试

分别模拟：

1. 材料计算审稿人；
2. 统计/ML 审稿人；
3. 数据与 provenance 审稿人。

每类给出：

- 5 个 major concerns；
- 5 个 minor concerns；
- 现有证据能否解决；
- 精确修改位置。

输出：

```text
reports/phase8a/01_reviewer_attack_matrix.md
```

## D. 稿件 v0.7

另存：

```text
CrossPiezo_ScreeningResolution_Manuscript_v0.7.tex
```

检查：

- title 与 F1/F3/F4 一致；
- elite/intermediate/broad 不混写；
- high-response 无 collider 夸大；
- controls 仅支持 workflow sensitivity；
- portfolio 仅支持 two-source robustness；
- 所有数字来自 manifest；
- figures/tables/captions 可独立理解；
- Methods 可重现；
- references 真实且支持 claim；
- limitations 完整。

不得覆盖 v0.6。

## E. 投稿路线

生成：

```text
submission/phase8a_venue_decision.md
```

给出：

1. 现稿可投的 benchmark/material-informatics 路线；
2. 做 48-material third protocol 后的增强路线；
3. 数据论文路线。

外部检索只用官方期刊页面并记录来源。

## F. 第三协议 readiness

不执行，只审计：

```text
configs/third_protocol_phase7c.yaml
```

检查：

- 48 材料可确定性重建；
- 四组各 12；
- P2、晶系、原子数平衡；
- 指标、预算、失败规则冻结；
- 未用 Phase 7C 后结果挑样本；
- 资源和 wall-time 估计。

输出：

```text
reports/phase8a/02_third_protocol_readiness.md
```

## G. 最终结论

只允许：

### Submission Ready

### Submission Ready, Stronger Venue Requires Adjudication

### Not Ready

输出：

```text
reports/phase8a/03_final_decision.md
```

完成后停止，不执行第三协议。
