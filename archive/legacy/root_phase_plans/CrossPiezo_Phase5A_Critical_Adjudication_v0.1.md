# CrossPiezo Phase 5A：关键复核与 Full-Go 判定

> 版本：v0.1  
> 前置状态：Phase 0–4 已完成，当前判定为 provisional Narrow Go  
> 当前严格配对：538  
> 当前报告指标：median absolute Frobenius = 0.805 C/m²；median normalized discrepancy = 1.597；top-50 Jaccard = 0.075；Kendall τ = 0.257  
> 本阶段目标：证明上述差异是可复现的 protocol effect，而不是结构映射、极性畴、O(3) 运输、对称投影、贡献类型或排名口径错误。  
> 结束条件：给出 `Strong Go / Benchmark Go / No-Go`，然后停止。不得进入完整 PULSE。

---

## 0. 执行边界

本阶段可以：

- 复核现有 538 个 Tier-1 配对；
- 修复明确的代码或 convention bug；
- 重建 pair manifest，并保留旧版本；
- 训练少量、同指标、同拆分的 in-source baseline；
- 做 PiezoJet strict-factor 交集和软模 feasibility；
- 生成新的审计报告。

本阶段不可以：

- 开发完整 PULSE；
- 大规模 SOTA 超参数搜索；
- 新增 DFT/DFPT；
- 联网下载数据；
- 用结果反向调整结构匹配容差；
- 将计算中心称为真实张量；
- 在 LaTeX 中写入未经最终冻结的主结论。

---

# 1. 首先冻结 Phase 0–4

创建：

```text
artifacts/releases/phase0_4_v1/
```

保存并 hash：

- 配置；
- strict/quarantine pair manifests；
- feasibility summary；
- reports 00–05；
- 当前代码 commit；
- 环境和依赖锁文件。

不得覆盖原结果。所有 Phase 5A 结果使用新版本号。

生成：

```text
reports/06_phase0_4_freeze.md
```

---

# 2. 纠正数据描述

必须核实并在报告中纠正：

- `4,995` 是 parsed JARVIS DFPT payload 数；
- `1,638` 才是 strict internal-strain completion 数；
- 不得把 4,995 全部称为 strict-factor records。

分别报告可用字段：

- force constants；
- Born effective charges；
- dielectric；
- electronic/ionic/total piezo；
- strict internal strain；
- stable optical status。

---

# 3. 对称残差红旗审计

当前报告：

- JARVIS median symmetry residual = 0.286；
- MP median symmetry residual = 0.090。

在继续任何物理结论前，明确：

1. residual 是 absolute 还是 relative；
2. 分母定义；
3. 使用 raw source point group 还是 common matched structure point group；
4. 投影前后的单位；
5. 是否包含 near-zero tensors；
6. 是否在正确 Cartesian frame 中计算；
7. 是否在 primitive/conventional cell 变换前后重复计算。

输出：

```text
reports/07_symmetry_residual_audit.md
artifacts/phase5a/symmetry_residuals.parquet
```

必须给出：

- raw residual；
- normalized residual；
- source point-group residual；
- common point-group residual；
- 投影改变量；
- 按晶系、幅值和 match quality 分层。

若 JARVIS median relative residual 在 convention 修复后仍大于 5%–10%，必须解释来源；否则不能将 projected tensor 当作无争议标签。

---

# 4. 人工可审计配对样本

选择固定的 60 对，不根据期望结论挑选：

- discrepancy 最低 15；
- 中位区间 15；
- 最高 15；
- sign/cosine/对称残差异常 15。

为每一对生成一个审计包：

```text
artifacts/phase5a/manual_pair_audit/<pair_id>/
├── summary.json
├── jarvis_structure.cif
├── mp_structure.cif
├── common_frame_structure.cif
├── mapping.json
├── tensors.json
└── comparison.md
```

`mapping.json` 至少包含：

- lattice transforms；
- Cartesian orthogonal transform；
- determinant；
- atom permutation；
- site RMS；
- lattice RMS；
- source and common space groups；
- handedness；
- ambiguity；
- forward/reverse reconstruction residual。

`tensors.json` 至少包含：

- raw source tensors；
- Cartesian tensors；
- transported tensors；
- symmetry-projected tensors；
- contribution type；
- units and Voigt history；
- exact-frame and domain-aware discrepancies。

生成总报告：

```text
reports/08_manual_pair_audit.md
```

---

# 5. 极性畴、手性和 O(3) 张量运输

压电张量是 polar third-rank tensor。必须审计结构匹配中的：

- proper rotation；
- improper rotation；
- inversion-related polar domains；
- enantiomorphic structures；
- mirror-related settings；
- axis permutation and handedness。

实现并分别报告以下差异：

## 5.1 Exact transported discrepancy

使用结构匹配给出的完整 O(3) 变换，将一侧张量运输到另一侧的同一 Cartesian frame。

## 5.2 Proper-orbit discrepancy

只在允许的 proper rotations / common crystallographic setting 内比较。

## 5.3 Domain-aware discrepancy

若两个结构被确认是 inversion-related polar domains，分别报告：

- signed tensor discrepancy；
- domain-equivalent discrepancy；
- 将其标记为 `polar_domain_flip`，不得静默取较小值。

## 5.4 Point-group-equivalent discrepancy

在共同点群操作下最小化，但不允许使用不属于该晶体物理等价类的任意旋转。

输出：

```text
artifacts/phase5a/discrepancy_variants.parquet
reports/09_domain_and_o3_transport.md
```

核心表必须显示每一步后：

- pair count；
- median absolute discrepancy；
- median normalized discrepancy；
- sign-flip fraction；
- top-k overlap；
- Kendall τ。

如果巨大差异在 domain/O(3) 修复后显著消失，必须降低论文 claim。

---

# 6. 结构差异分层

Tier 1 仍可能含不同 relaxed structures。对 538 对报告：

- volume ratio；
- lattice strain；
- site RMS；
- maximum site displacement；
- space-group equality/change；
- Wyckoff mapping；
- point-group equality；
- polarity direction relation。

建立三个冻结子层：

```text
T1a: near-identical relaxed structures
T1b: same symmetry, measurable relaxation shift
T1c: symmetry/polar-domain ambiguity but verified relation
```

主 protocol floor 必须优先在 T1a 上报告。

模型分析中将 structure shift 与 protocol label shift 分开：

```text
discrepancy ~ structure_mismatch + chemistry + source/protocol sensitivity
```

输出：

```text
reports/10_structure_mediated_shift.md
```

---

# 7. 重新验证排名不稳定性

确认 top-50 Jaccard = 0.075 和 Kendall τ = 0.257 是否满足：

1. 排名宇宙完全相同，只包含同一批严格 paired records；
2. 两侧使用同一个 performance functional；
3. 同一个 contribution（total/ionic/electronic）；
4. 同一个 tensor norm 和单位；
5. 不混入 unpaired source records；
6. near-zero 和 ties 使用固定规则；
7. 不以 source-specific derived field 代替完整张量；
8. ranking 在 domain-aware tensor transport 后重新计算。

预注册主指标：

- symmetry-adapted Frobenius norm；
- maximum longitudinal response（公式固定）；
- maximum shear response（公式固定）。

如有同源完整 `C`，次指标：

- `d = e C^{-1}`；
- 明确的方向响应。

报告：

- top-20/50/100 Jaccard；
- intersection count；
- Kendall τ-b；
- Spearman；
- bootstrap CI；
- rank shift distribution；
- 按 T1a/T1b、幅值、晶系分层。

输出：

```text
reports/11_ranking_revalidation.md
artifacts/phase5a/ranking_metrics.parquet
```

---

# 8. 可比的 in-source baseline 和 PMR

不得直接用文献中不同 split、单位或指标的 MAE 作为 PMR 分母。

## 8.1 冻结 paired test panel

从 538 对中按 formula/prototype group 建立：

- paired calibration；
- paired test；
- 不与任一 source training set 泄漏。

同一结构及相关 prototype 必须在同一 split。

## 8.2 Baselines

至少实现：

1. zero baseline；
2. source-specific simple structural baseline；
3. 一个可审计的 O(3)-equivariant tensor baseline；
4. pooled source-agnostic baseline；
5. source-token baseline。

如复用 GMTNet/EATGNN/PiezoJet 代码，记录：

- commit；
- exact architecture；
- target convention；
- split；
- seed；
- parameter count；
- training budget。

不需要追求 SOTA，只要求相同指标和公平比较。

## 8.3 2×2 counterfactual matrix

对同一个 paired test panel：

```text
train JARVIS → evaluate JARVIS label
train JARVIS → evaluate MP label
train MP     → evaluate MP label
train MP     → evaluate JARVIS label
```

另加 pooled/source-token。

报告相同的：

- absolute Frobenius；
- normalized Frobenius；
- component MAE；
- cosine/amplitude；
- rank metrics。

## 8.4 PMR

至少给出：

```text
PMR_absolute
PMR_normalized
PMR_high_response
PMR_T1a
```

使用 bootstrap CI。

输出：

```text
reports/12_in_source_and_pmr.md
artifacts/phase5a/baseline_metrics.parquet
```

不得在模型指标明显低于 zero baseline 时解释 PMR。

---

# 9. Soft-mode feasibility

## 9.1 交集

构建：

```text
strict Tier-1 pairs
∩ PiezoJet strict internal-strain completion
∩ stable optical references
∩ convention-complete factors
```

报告每一步的剩余数量。

建议解释门槛：

- `N >= 150`：可做较强 grouped analysis；
- `80 <= N < 150`：探索性机制分析；
- `N < 80`：不能作为主要机制 claim。

## 9.2 变量

计算或读取：

- minimum positive optical eigenvalue；
- optical condition number；
- ionic fraction；
- Born-charge norm；
- internal-strain norm；
- force-constant norm；
- volume；
- `S_soft`；
- mode-contribution concentration；
- near-cancellation index。

## 9.3 注意

这些 microscopic factors 主要来自 JARVIS，不能直接归因：

```text
JARVIS factor difference → MP/JARVIS protocol difference
```

因为没有 MP 同构 factor labels。

允许的 claim：

> JARVIS-side physical sensitivity indicators predict which materials exhibit larger cross-protocol discrepancy.

不允许的 claim：

> 某个具体 JARVIS–MP 计算设置导致了差异。

## 9.4 分析

- chemistry-only baseline；
- structure-shift baseline；
- factor baseline；
- combined model；
- formula/prototype grouped CV；
- robust regression；
- bootstrap CI；
- remove lowest-eigenvalue tail；
- partial correlation；
- permutation test。

输出：

```text
reports/13_soft_mode_feasibility.md
artifacts/phase5a/soft_mode_metrics.parquet
```

---

# 10. Phase 5A 决策

## Strong Go

需同时满足：

1. domain/O(3)/symmetry 复核后，严格 paired `N >= 400`；
2. T1a 子集数量足以独立报告；
3. cross-protocol discrepancy 仍与模型误差可比：
   - 至少一个预注册 PMR 的 95% CI 下界 ≥ 0.5；
4. 至少一个预注册 top-k Jaccard ≤ 0.5 或 Kendall τ ≤ 0.7；
5. 排名不稳定不由 near-zero、极少数 outlier 或单一晶系驱动；
6. 以下至少一个成立：
   - soft-mode/factor mechanism 在 grouped CV 中稳定；
   - version shift 有独立证据；
   - 可以制定第三协议 adjudication 的固定采样方案。

Strong Go 后才能设计 PULSE。

## Benchmark Go

满足：

- 配对和跨协议差异有效；
- PMR 或排名不稳定成立；
- 但 soft-mode 机制弱、样本不足或 source-aware learning 增益尚不明确。

下一步做 CrossPiezo benchmark、leaderboard stress test 和轻量 calibration，不开发过度复杂的 PULSE。

## No-Go / Claim Downgrade

出现以下任一：

- domain/O(3)/convention 修复后 paired 数小于 250；
- discrepancy 大幅下降并主要由表示错误解释；
- T1a 上差异远小于模型误差；
- 排名在同宇宙、同指标后稳定；
- soft-mode 交集不足且无其他机制或 version 证据；
- source-aware baseline 没有校准或决策增益。

---

# 11. LaTeX 更新规则

本阶段不直接修改主 `.tex` 的 Abstract、Results 或 Discussion。

只生成：

```text
manuscript_notes/phase5a_verified_numbers.md
manuscript_notes/phase5a_claim_changes.md
```

其中区分：

- verified and frozen；
- provisional；
- rejected；
- requires third protocol；
- requires model phase。

只有人工批准后，才将冻结数字写入 LaTeX。

---

# 12. 最终产物

```text
reports/
├── 06_phase0_4_freeze.md
├── 07_symmetry_residual_audit.md
├── 08_manual_pair_audit.md
├── 09_domain_and_o3_transport.md
├── 10_structure_mediated_shift.md
├── 11_ranking_revalidation.md
├── 12_in_source_and_pmr.md
├── 13_soft_mode_feasibility.md
└── 14_phase5a_decision.md

artifacts/phase5a/
├── discrepancy_variants.parquet
├── symmetry_residuals.parquet
├── ranking_metrics.parquet
├── baseline_metrics.parquet
├── soft_mode_metrics.parquet
├── manual_pair_audit/
└── frozen_summary.json
```

完成 `reports/14_phase5a_decision.md` 后停止。

---

# 13. 最终汇报格式

回复研究者时必须明确：

1. 原 538 对中保留多少；
2. symmetry residual 的确切定义和修复结果；
3. 有多少 polar-domain / improper-transform 问题；
4. 差异在每一步审计后如何变化；
5. 排名指标是否在同一 paired universe 上复现；
6. 2×2 baseline counterfactual matrix；
7. PMR 和 CI；
8. strict-factor 交集数量；
9. soft-mode 结果和限制；
10. Strong Go / Benchmark Go / No-Go；
11. 下一阶段是否允许开发 PULSE；
12. 没有执行的任务。
