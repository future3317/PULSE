# CrossPiezo Phase 5B：Benchmark Consolidation 与论文可发表性判定

> 前置判定：Benchmark Go  
> 当前禁止：不开发完整 PULSE；不新增未经批准的 DFT/DFPT；不直接填写 LaTeX 结果段。  
> 核心任务：把“显著跨源差异”升级为经得起审稿的 benchmark 和 leaderboard stress test。

## 0. 必须纠正的 Phase 5A 结论

1. `structural_ridge` 在两个 in-source 任务上都差于 zero baseline：
   - JARVIS: 0.516 > 0.494
   - MP: 1.046 > 0.914  
   因此 Phase 5A 的 PMR=1.031 不能作为“protocol gap 与有效模型误差可比”的证据。
2. PMR 当前混用了 median protocol discrepancy 和 mean model MAE，统计口径不一致。
3. normalized symmetry residual 极高：
   - JARVIS: 0.92
   - MP: 0.43  
   必须恢复 source-native calculation frame，不能仅用 unified CIF 的点群判断张量有效性。
4. soft-mode grouped-CV 为负；机制主张暂时删除，除非完整 atom-resolved Λ 被恢复并重新验证。

生成 `reports/15_phase5a_corrections.md`，明确以上更正。

---

## 1. 冻结 Phase 5A

创建 `artifacts/releases/phase5a_v1/`，保存并 hash：

- reports/06–14
- artifacts/phase5a
- configs
- code commit
- dependency lock/environment
- pair manifests

不得覆盖历史结果。

---

## 2. 建立 Core 与 Extended benchmark

### CrossPiezo-Core

默认要求：

- T1a
- total piezoelectric stress tensor `e`
- contribution/unit/convention complete
- source-native structure 和 tensor frame 可恢复
- 无未解决 mapping ambiguity
- source-native symmetry audit 通过，或残差来源被明确解释
- 不按 discrepancy 大小筛选

### CrossPiezo-Extended

- 全部 538 Tier-1 pairs
- 含 T1b/T1c
- 主要使用 orbit-/rotation-/domain-invariant metrics
- 不用于未说明的逐分量误差

输出：

- `artifacts/phase5b/core_pairs.parquet`
- `artifacts/phase5b/extended_pairs.parquet`
- `reports/16_core_extended_panels.md`

报告 Core N、排除原因及选择偏差。

---

## 3. 恢复 source-native tensor frame

分别定位并读取：

- JARVIS DFPT 原始/最接近原始 calculation structure
- MP DFPT calculation structure
- tensor native Cartesian frame
- source conventional/primitive relation
- exact transform to matched common structure
- parser transformation history

优先搜索本地 raw DFPT payload、JARVIS structure index、MP piezo documents 和 manifest。

每个 source/pair 计算：

1. `native_residual`：tensor 对 source native calculation structure 点群的 residual
2. `transport_residual`：运输后 tensor 对 transported source point group 的 residual
3. `common_residual`：对 matched common structure 点群的 residual

输出：

- `artifacts/phase5b/source_native_residuals.parquet`
- `reports/17_source_native_frame_audit.md`

若 native residual 仍普遍很高：

- 检查 tensor field、Voigt、unit、contribution 和 frame
- quarantine 无法闭合的记录
- 主论文不得使用这些记录做 componentwise/cosine/directional comparison
- 可保留 rotation-invariant norm/ranking benchmark

---

## 4. 差异层级

冻结并报告：

- source-native invariant norm discrepancy
- exact transported tensor discrepancy
- proper-orbit discrepancy
- domain-aware discrepancy
- point-group-equivalent discrepancy
- symmetry-projected discrepancy
- T1a/T1b structure-mediated discrepancy

规则：

- Frobenius norm 是主 screening metric，因为对正交 frame 和整体符号不敏感。
- component/cosine/directional metrics 只在 Core panel 上使用。
- symmetry projection 后 normalized discrepancy 不得与 raw normalized discrepancy直接比较；同时报告 norm-retention ratio。

输出 `reports/18_discrepancy_hierarchy.md`。

---

## 5. 训练真正有 skill 的 equivariant baseline

### 训练数据

对每个 source：

- 使用该 source 全部可用训练记录
- 排除 Core/Extended test material、formula group、prototype group
- paired test panel 冻结
- 不只使用 423 个 paired calibration records

### Baselines

至少：

1. zero predictor
2. train mean/median predictor
3. composition baseline
4. source-specific O(3)-equivariant tensor model
5. pooled equivariant model
6. source-token equivariant model

优先复现一个强公开模型（GMTNet、EATGNN、GoeCTP 或 CEITNet）和一个架构不同的验证模型。无需全部复现。

### 有效模型 Gate

只有满足以下条件的模型才可进入 PMR 分母：

- in-source test 明显优于 zero 和 mean/median
- 至少 3 seeds
- exact same split/metric/unit/convention
- Core panel skill score > 0
- 无 near-zero collapse
- equivariance/symmetry audit 通过

输出：

- `artifacts/phase5b/model_metrics.parquet`
- `reports/19_equivariant_baselines.md`

---

## 6. 有效 PMR

对每个合格模型计算：

- PMR_mean_absolute
- PMR_median_absolute
- PMR_mean_normalized
- PMR_median_normalized
- PMR_Core
- PMR_high_response
- PMR_T1a

要求：

- mean/mean，median/median
- 同一 test universe
- 同一 contribution/unit/tensor metric
- paired bootstrap 95% CI
- 报告全部合格模型，不只选最有利者

另报告：

`SPG = protocol_discrepancy / (best_valid_in_source_error + epsilon)`

输出 `reports/20_valid_pmr.md`。

---

## 7. Leaderboard stress test

每个有效模型报告：

- legacy/in-source
- formula-disjoint
- prototype-disjoint
- source-held-out
- paired counterfactual
- Core
- Extended invariant
- high-response
- T1a/T1b

回答：

1. in-source 模型排名是否在 source-held-out 下反转？
2. 更强的 in-source 模型是否更接近 alternate-source labels？
3. source-token 是否改善跨源风险，还是只拟合 source？
4. 不同模型产生的 top-k 候选是否 protocol 稳健？

输出：

- `artifacts/phase5b/leaderboard.parquet`
- `reports/21_leaderboard_stress_test.md`

---

## 8. 轻量 calibration（仅在有效点模型后执行）

只比较：

- deep ensemble
- source-stratified residual scaling
- heteroscedastic source head
- conservative split/grouped conformal

报告：

- source-conditional coverage
- Core/T1a coverage
- OOD risk-coverage
- abstention 后 tail error
- top-k screening stability
- sharpness

不开发完整 PULSE，不声称 conditional coverage guarantee。

输出 `reports/22_lightweight_calibration.md`。

---

## 9. Full Λ recovery audit

检查 PiezoJet strict-completion 原始缓存中是否存在：

- atom-resolved `3N × 6` 或等价完整 Λ
- exact atom mapping
- factor convention
- stable optical mask
- transformation history

当前 `(3,3,3)` 字段可能是派生或压缩表示，不能直接当完整 Λ。

输出 `reports/23_full_lambda_recovery_audit.md`。

决策：

- 若完整 Λ 可恢复：允许一次真正的 mode-resolved exploratory analysis。
- 若不可恢复，或 grouped-CV 仍为负：从标题、摘要和主贡献删除 soft-mode mechanism，仅在 Discussion 标为未验证假设。

---

## 10. Version shift 与 third-protocol 方案

本阶段不自动运行新 DFT，但必须制定两个升级路径。

### Path A：database version shift

调查历史 MP piezo snapshot 是否能提供：

- exact old tensor
- old structure
- current tensor
- version provenance
- matched overlap

### Path B：third-protocol DFPT

制定固定 48–96 材料计划，按以下分层预注册：

- Core high/low disagreement
- high/low response
- multiple crystal systems
- small-cell compute feasibility
- ranking consensus/dispute

明确：

- ABINIT 或 Quantum ESPRESSO
- functional
- pseudopotential library
- convergence
- tensor convention
- budget
- fail policy

输出：

- `reports/24_adjudication_options.md`
- `configs/third_protocol_plan.yaml`

---

## 11. Phase 5B 判定

### Model-Validated Benchmark Go

需要：

- Core source-native frame 闭合
- 至少一个 equivariant model 显著优于 zero/mean
- valid PMR 或 leaderboard shift 显著
- ranking instability 在 Core 保持
- calibration 改善 risk-coverage 或 screening reliability

允许：

- 撰写 CrossPiezo benchmark + leaderboard stress-test 论文
- 保留轻量 source-aware calibration
- 进入 third-protocol adjudication

不允许自动开发复杂 PULSE。

### Data-Only Benchmark Go

- 差异和 ranking instability 成立
- 但模型无有效 skill 或 calibration 无增益

定位为数据/benchmark 论文，不声称 protocol gap 超过 competent model error。

### No-Go / Representation Paper

如果 source-native frame 恢复后大部分差异消失，转为 tensor provenance、coordinate convention 和 database interoperability 论文。

---

## 12. LaTeX 更新规则

不要直接修改主 LaTeX。

生成：

- `manuscript_notes/phase5b_revised_title_and_abstract.md`
- `manuscript_notes/phase5b_results_table_template.md`
- `manuscript_notes/phase5b_claim_matrix.md`

当前建议标题：

> CrossPiezo: Cross-Protocol Evaluation Reveals Unstable Rankings in AI Screening of Piezoelectric Materials

只有 valid PMR 建立后，才恢复：

> Protocol-Induced Uncertainty Limits AI Screening of Piezoelectric Materials

---

## 13. 完成后停止

生成 `reports/25_phase5b_decision.md`，汇报：

1. Core/Extended N
2. source-native residual
3. valid baseline skill
4. valid PMR
5. leaderboard inversion
6. calibration value
7. full Λ recovery
8. version/third-protocol recommendation
9. revised paper title and target venue
10. 是否批准 adjudication
11. 未执行事项

完成后停止，不运行第三协议，不开发完整 PULSE。
