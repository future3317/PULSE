# PULSE / CrossPiezo 代码正确性审计与重建任务书

> 版本：v0.1  
> 审计基线：`future3317/PULSE`，`master` at `ee419516f8641fba18e99266c3288349f96e2785`  
> 数据根目录：本地 `E:/DATA` 或远程 `~/DATA`，永久只读  
> 本阶段性质：**correctness reset**  
> 当前所有 Phase 0–5B 数字均视为 provisional，直至本任务完成  
> 本阶段结束前：不得进入 MP version shift、第三协议、PULSE 模型或论文结果写作

---

## 0. 目标

对当前仓库做一次独立、可追踪、以物理和数学正确性为优先的代码审计，并重新生成所有核心数据结果。

本任务不以“让现有结果继续成立”为目标。允许并要求得出以下结论之一：

- 旧结果基本正确；
- 旧结果部分成立但数值需要修订；
- 旧结果主要由 convention / frame / matching / metric bug 引起；
- 当前 benchmark 不成立，需要重新立项。

任何修复都不得以保住 538、Jaccard 0.075 或既有论文叙事为目的。

---

# 1. Git 与产物保护

## 1.1 创建审计分支

```bash
git checkout master
git pull --ff-only
git rev-parse HEAD
git checkout -b audit/correctness-v1
```

确认基线 SHA 是：

```text
ee419516f8641fba18e99266c3288349f96e2785
```

如果不是，记录差异并停止等待确认。

## 1.2 冻结旧结果

复制或建立不可变 manifest：

```text
artifacts/releases/pre_audit_ee4195/
```

包含：

- 当前 configs；
- reports 00–25；
- pair manifests；
- Phase 5A/5B artifacts；
- commit SHA；
- Python/package environment；
- 文件 hashes。

不得覆盖旧产物。

## 1.3 新命名空间

所有审计后结果写入：

```text
artifacts/correctness_v1/
reports/correctness_v1/
```

旧报告顶部增加非破坏性声明文件，不直接重写旧报告：

```text
reports/CURRENT_STATUS.md
```

内容必须说明 Phase 0–5B 结果正在 correctness audit，尚不可用于论文主张。

---

# 2. 已识别的高风险问题

以下是静态代码审查发现的风险。先写失败测试，再判断是否为真实 bug。不得直接照结论修改而不验证。

## C-01：压电 Voigt ↔ Cartesian 的 shear scaling

当前：

```python
voigt_to_cartesian(..., engineering_shear=True)
```

对 shear 分量乘 `0.5`，反向乘 `2.0`。

对压电**应力系数** `e_{iJ}`，若工程应变定义为：

```text
eta_V = [eps_xx, eps_yy, eps_zz, 2 eps_yz, 2 eps_xz, 2 eps_xy]
```

且：

```text
e_iJ eta_J = e_ijk eps_jk
```

则对称 Cartesian tensor 的 shear 分量一般应满足 `e_i4 = e_i23`，不能仅靠普通 symmetric-strain converter 假设 `0.5`。

任务：

1. 明确区分：
   - piezoelectric stress tensor `e`；
   - piezoelectric strain tensor `d`；
   - elastic stiffness/compliance；
   - generic symmetric tensor。
2. 为每种类型建立独立 converter，不再使用一个模糊的 `engineering_shear` bool。
3. 用功共轭恒等式做 oracle test。
4. 用 `pymatgen.analysis.piezo.PiezoTensor.from_vasp_voigt` 做独立对照。
5. 审计 `piezo_cartesian_total` 是由什么代码产生的；当前数据若直接读取此字段，确认它是否已经受同一错误影响。
6. 不允许仅用 round-trip test，因为一对互逆的错误 converter 也会通过 round trip。

必测：

```text
e_voigt · eta_engineering == sum_ijk e_cart[i,j,k] eps[j,k]
```

对随机 strain、随机 tensor、每个 shear basis 单独测试。

---

## C-02：polar rank-3 tensor 的 improper O(3) 变换

当前 `_transport_tensor` 先计算：

```text
R ⊗ R ⊗ R : e
```

然后在 `det(R) < 0` 时额外乘 `-1`。

对 ordinary polar rank-3 tensor，`R ⊗ R ⊗ R` 已经包含 inversion 下的负号；额外 `det(R)` 因子属于 pseudotensor/axial transformation，而不是 polar tensor。

任务：

1. 将 `transform_polar_rank3` 与 `transform_axial_rank3` 完全分开。
2. 对 `R=-I`，polar rank-3 必须得到 `-e`。
3. 对 reflection、rotoreflection 做直接 index-contraction oracle。
4. 重新审计 177 个 `polar_domain_flip`。
5. 旧 domain-aware / point-group-equivalent 数字全部失效，必须重算。

---

## C-03：fractional symmetry matrices 被直接用于 Cartesian tensor

当前 `point_group_rotations` 从 `SpaceGroup.symmetry_ops` 读取 rotation matrix，并直接作用于 Cartesian tensor。

风险：

- space-group operation 通常在 fractional basis；
- 非正交晶胞下矩阵不是 Cartesian orthogonal rotation；
- 仅有国际空间群号不能恢复实际 CIF setting/orientation；
- centering translations 可能导致重复 rotations。

任务：

1. 删除在 Cartesian tensor 上使用抽象 `SpaceGroup` fractional matrices 的路径。
2. 从实际 `Structure` 构建：
   ```python
   SpacegroupAnalyzer(structure, symprec=..., angle_tolerance=...)
       .get_point_group_operations(cartesian=True)
   ```
3. 去重 rotations。
4. 检查：
   - `R.T @ R == I`；
   - `det(R) ∈ {+1,-1}`；
   - group closure；
   - identity；
   - inverse。
5. 同一结构在不同 unimodular cell representations 下投影结果必须协变。
6. 对 triclinic、monoclinic、orthorhombic、trigonal、hexagonal、cubic 各建 synthetic tests。
7. 重新计算全部 symmetry residual。
8. 不再通过空间群号构建“source-native” symmetry。

---

## C-04：`source-native` audit 实际未使用两个 source-native structures

当前 `compute_source_native_residuals`：

- 使用单一 `row["space_group"]`；
- `space_group` 来自 JARVIS row；
- 对 JARVIS 和 MP 都使用相同 `common_rots`；
- 未从 source calculation structure 恢复各自 operations。

因此当前 `native_residual`、Core=15 不能视为 source-native 结论。

任务：

1. 分别恢复：
   - JARVIS calculation/native structure；
   - MP calculation/native structure；
   - 每个 tensor 的实际 reporting frame；
   - source parser transformation history。
2. 为每个 source 单独计算 Cartesian point-group operations。
3. 明确状态：
   ```text
   native_frame_verified
   native_frame_reconstructed
   native_frame_unresolved
   ```
4. 无法恢复时不得计算伪 `native_residual`。
5. Core panel 必须完全重建，不继承 N=15。

---

## C-05：structure-match rotation 的估计方式可能把 cell basis change 当物理旋转

当前做法：

```text
s2_like_s1 lattice × inverse(original s2 lattice)
→ polar decomposition
→ force det=+1
```

风险：

- unimodular cell relabeling不是物理 Cartesian rotation；
- `scale=True` 引入体积缩放；
- polar decomposition 不一定是结构的真实刚体 rotation；
- 强制 proper 会抹掉真实 improper relation；
- tensor transport 未通过 atom-position reconstruction 证明。

任务：

1. 使用 `StructureMatcher.get_transformation` 获取：
   - supercell/basis matrix；
   - fractional translation；
   - site mapping。
2. 将“晶胞整数基变换”与“实验室 Cartesian rigid rotation”分开。
3. 通过 matched atom Cartesian coordinates 做 Kabsch/Procrustes，仅在映射闭合时恢复 orthogonal `Q`。
4. 周期 image、translation、centering 必须显式处理。
5. 不强制 `det(Q)=+1`；保留 handedness 并分类。
6. 对每个 match 输出：
   - integer basis matrix；
   - translation；
   - atom permutation；
   - Cartesian `Q`；
   - det；
   - reconstruction RMS；
   - forward/reverse closure。
7. Synthetic tests：
   - 同一结构仅换 unimodular cell：Cartesian transport 应为 identity；
   - 结构整体 proper rotation：恢复该 rotation；
   - inversion/reflection relation：恢复 improper Q；
   - supercell/primitive representation；
   - atom reordering；
   - periodic translation。
8. 如果无法唯一恢复 Q，标记 unresolved，不用于 componentwise tensor comparison。

---

## C-06：RMS 与 max distance 可能反置

`StructureMatcher.get_rms_dist` 返回：

```text
(rms displacement, maximum distance)
```

当前代码把第一项赋给 `max_dist`，第二项赋给 `rms_dist`。

任务：

1. 用当前 pymatgen 官方行为和 synthetic pair 测试确认。
2. 修复字段命名。
3. 重算 T1a/T1b/T1c。
4. 审计所有旧报告中对 `rms_distance` / `max_distance` 的使用。

---

## C-07：crystal-system mapping 缺少 orthorhombic

当前阈值映射将空间群 16–74 归入 monoclinic。

任务：

1. 删除手写阈值表。
2. 使用 `SpacegroupAnalyzer.get_crystal_system()` 或经测试的标准映射。
3. 重算所有晶系分层统计。

---

## C-08：orbit discrepancy 没有先做 exact matched-frame transport

当前 `proper_orbit_discrepancy` 和 `point_group_equivalent_discrepancy` 直接对 raw right tensor 施加 common point-group operations，未先运输到 left/common frame。

任务：

1. 定义严格顺序：
   ```text
   source-native tensor
   → verified exact mapping to common Cartesian frame
   → optional physical symmetry/domain equivalence
   → discrepancy
   ```
2. orbit minimization只能使用 actual common structure 的 Cartesian symmetry operations。
3. 不允许 point group 被用作任意 frame alignment 替代品。
4. 重算 discrepancy hierarchy。

---

## C-09：longitudinal/shear ranking functional 不是旋转不变量

当前：

```python
max_longitudinal = max(|e_xxx|, |e_yyy|, |e_zzz|)
max_shear = max(all components except those three)
```

这些只是当前坐标轴分量，不是“最大方向响应”。

任务：

1. 主 benchmark 暂时只保留经过 convention 验证的 Cartesian norm/invariants。
2. 真正 longitudinal functional 应明确为例如：
   ```text
   max_{||n||=1} | n_i e_ijk n_j n_k |
   ```
   使用可验证的球面优化/多起点优化。
3. shear functional 必须给出明确的正交方向约束和物理定义；定义不清时删除。
4. 随机 O(3) rotation 前后 functional 数值必须一致。
5. 与高密度球面 brute-force oracle 比较。
6. 所有旧 longitudinal/shear rank 结果撤回并重算。

---

## C-10：e3nn 输出 symmetrize 了错误的 tensor axes

若输出 shape 是：

```text
(batch, 3, 3, 3)
```

当前：

```python
cart.transpose(1, 2)
```

交换的是第一和第二 tensor index；压电 `e_ijk` 应对最后两个 strain indices `j,k` 对称。

任务：

1. 使用：
   ```python
   cart.transpose(-1, -2)
   ```
   或直接输出正确的 symmetry-adapted irreps。
2. 加测试：
   ```text
   pred[i,j,k] == pred[i,k,j]
   ```
   且不错误强制 `pred[i,j,k] == pred[j,i,k]`。
3. 使用 known tensor with only one allowed shear component 进行测试。

---

## C-11：当前 e3nn graph 不含周期边界

数据集计算了 PBC edges，但模型没有使用；`gate_points_2101.Network` 仅从 `pos` 构建普通 radius graph。

这会把晶体单位胞当作有限原子团。

任务：

1. 在修复前将当前 e3nn 标为 `invalid_crystal_baseline`。
2. 实现或复用经过验证的 periodic graph：
   - edge index；
   - periodic image shifts；
   - Cartesian edge vectors；
   - lattice；
   - batch。
3. 模型必须真正消费这些 edges。
4. 测试：
   - 原子平移一个 lattice vector，输出不变；
   - 不同等价 unit-cell choices，输出协变；
   - 周期边界附近邻居被正确包含。
5. 若无法快速可靠实现，移除 e3nn，不将其列入科学 baseline。

---

## C-12：所谓 prototype split 实际是“元素集合 split”

当前 `_formula_to_prototype` 返回 sorted element set，例如所有 Na-Cl 化合物可能同组；这不是结构 prototype。

而且训练池只排除了 test IDs，没有按 formula/prototype group 排除。

任务：

1. 更名现有字段为 `chemical_system`，不得称 prototype。
2. prototype 使用至少一种：
   - AFLOW prototype label；
   - anonymous structure matcher connected component；
   - 已有 T2C/Alex prototype or matcher envelope。
3. 创建 pair-connected split manifest。
4. 训练时实际排除：
   - material IDs；
   - formula group；
   - prototype component；
   - duplicate/matcher envelope。
5. 增加 leakage tests，证明交集为零。
6. 重算所有 formula/prototype-disjoint metrics。

---

## C-13：source-held-out / counterfactual evaluation 标签不准确

当前 pooled/e3nn 模型训练包含两个 source，却生成名为 `source_held_out_*` 的结果；部分 `paired_counterfactual` eval map 与 in-source eval set 相同。

任务：

1. 明确定义：
   - train J → test J；
   - train J → same-pair MP；
   - train MP → test MP；
   - train MP → same-pair J；
   - pooled train → each source；
   - leave-one-source-out train（只有 ≥3 sources 时才真正成立）。
2. 两来源情况下不要把 pooled model 叫 source-held-out。
3. 每个 prediction artifact 保存 material IDs 和 pair IDs，验证同一 universe。
4. report 不得只按 split name 推断实验设计。

---

## C-14：报告中硬编码旧指标

`compile_phase5b_reports.py` 硬编码 top-50 Jaccard 和 Kendall τ，而不是从当前 artifact 读取。

任务：

1. 禁止报告脚本出现科学数值常量。
2. 所有数字来自 hash-bound artifact。
3. 报告记录 artifact hash、config hash、commit。
4. 若依赖 artifact 缺失，fail closed，不使用旧数字。
5. lambda audit 也不得 hardcode 0。
6. 添加“修改输入 artifact 后报告数字随之变化”的 regression test。

---

## C-15：PMR 实现存在指标混用风险

当前 `PMR_mean_normalized` 使用 absolute paired discrepancy 除以 normalized model error；并且 CI 只 bootstrap numerator。

任务：

1. 每个 scope 保存 per-sample：
   - absolute protocol discrepancy；
   - normalized protocol discrepancy；
   - per-sample model error。
2. PMR：
   - mean absolute / mean absolute；
   - median absolute / median absolute；
   - mean normalized / mean normalized；
   - median normalized / median normalized。
3. paired bootstrap 对完整 ratio 重采样，而不是只重采样 numerator。
4. 同一 pair universe、同一样本权重。
5. 模型无 skill 时不计算科学 PMR。
6. 添加 synthetic test，手算期望 PMR。

---

## C-16：模型任务与 zero-inflated label 不匹配

全样本 absolute tensor MAE 被大量近零标签主导。一个有用的高响应筛选模型可能不击败 zero MAE。

任务：

在代码正确性通过后才设计，但现在先建立 metric tests：

- high-response classification PR-AUC；
- recall@k；
- precision@k；
- NDCG；
- Spearman/Kendall；
- log-amplitude regression；
- non-zero conditional error。

本阶段不得用这些新任务掩盖 convention bug。它们只有在 correctness gate 通过后启用。

---

# 3. 测试策略

## 3.1 先写失败测试

每个 C-01 至 C-15 至少一个 red test。

## 3.2 Oracle tests

必须包含：

1. pymatgen `PiezoTensor.from_vasp_voigt` 对照；
2. `SymmOp.transform_tensor` 对照；
3. `SpacegroupAnalyzer(...).get_point_group_operations(cartesian=True)`；
4. work-conjugacy identity；
5. analytic inversion/reflection；
6. brute-force directional response；
7. StructureMatcher return-order test；
8. synthetic cell-setting transformations。

## 3.3 Property-based tests

使用 Hypothesis/randomized tests：

- random O(3)；
- random symmetric-last-two polar tensors；
- random engineering strains；
- random unimodular matrices；
- random atom permutations；
- random lattice translations。

## 3.4 End-to-end synthetic mini benchmark

构建不依赖 `E:/DATA` 的 6–12 个结构测试集：

- same tensor under coordinate rotation；
- same structure under basis relabeling；
- polar-domain partner；
- intentionally different tensor；
- invalid convention；
- ambiguous match。

预期 pair tiers、transforms、residuals、ranking全部手工可判定。

## 3.5 测试门槛

不得以“5 passed”作为完成。

最低应包含：

```text
tests/conventions/
tests/symmetry/
tests/matching/
tests/transport/
tests/ranking/
tests/reports/
tests/models/
tests/integration/
```

所有 critical paths 必须有独立 oracle，而不只是 round-trip。

---

# 4. 独立 source reconstruction audit

从 JARVIS 和 MP 各固定抽样至少 30 条：

- 10 个低响应；
- 10 个中等响应；
- 10 个高响应；
- 覆盖多个晶系。

对每条生成：

```text
artifacts/correctness_v1/source_reconstruction/<source>/<id>/
```

包含：

- raw source structure；
- raw source Voigt tensor；
- trusted library conversion；
- project conversion；
- native Cartesian operations；
- symmetry residual；
- source-derived scalar reproduced；
- transformation history；
- pass/fail。

优先复现来源自身字段：

- JARVIS `max_pza` 或等价；
- MP `e_ij_max` / max direction 或等价；
- 公开 raw tensor 与 parquet tensor。

如果不能复现来源自身派生量，则所有 benchmark 数字停止发布。

---

# 5. 重建 pipeline

顺序固定：

```text
A. raw source lineage
B. trusted tensor conversion
C. actual Cartesian source symmetry
D. strict structure correspondence
E. verified Cartesian mapping
F. invariant and componentwise discrepancy
G. ranking metrics
H. only then model/data benchmark
```

不能从旧 parquet 的 `piezo_cartesian_total` 直接开始并假定正确。

所有新 artifact schema 包含：

```text
schema_version
code_commit
config_hash
source_artifact_hash
converter_name_and_version
structure_setting
tensor_setting
mapping_status
validation_status
```

---

# 6. 审计产物

必须生成：

```text
reports/correctness_v1/
├── 00_static_code_review.md
├── 01_tensor_conversion_audit.md
├── 02_o3_parity_audit.md
├── 03_cartesian_symmetry_audit.md
├── 04_structure_matching_audit.md
├── 05_source_native_lineage.md
├── 06_ranking_functional_audit.md
├── 07_model_and_split_audit.md
├── 08_reporting_and_statistics_audit.md
├── 09_old_vs_corrected_results.md
└── 10_correctness_decision.md
```

`09_old_vs_corrected_results.md` 按每个旧主数字列出：

```text
old value
corrected value
status: confirmed / revised / invalid / unresolved
root cause
affected claims
```

至少覆盖：

- 538 pairs；
- T1a/T1b/T1c；
- Core=15；
- symmetry residuals；
- exact/domain/point-group discrepancy；
- polar-domain flip count；
- top-50 Jaccard；
- Kendall τ；
- longitudinal/shear ranks；
- baseline metrics；
- PMR/SPG；
- source-held-out claims。

---

# 7. Correctness Gate

## Pass

只有以下全部满足才通过：

1. e/d/elastic converters 有 oracle tests；
2. Cartesian symmetry operations来自实际 structure；
3. polar O(3) parity通过；
4. structure transform 在 synthetic cases 中闭合；
5. source reconstruction 抽样通过率 ≥ 95%，失败有可解释 quarantine；
6. 报告无 hardcoded scientific numbers；
7. split leakage tests 通过；
8. old vs corrected result table完成；
9. 全部 tests、ruff、mypy 通过；
10. 另一个独立实现或 trusted library 与主实现一致。

## Conditional Pass

source raw lineage 缺失，导致只能做 invariant benchmark：

- 明确删除 componentwise claims；
- benchmark 只发布经过验证的 invariants；
- 不使用“source-native tensor”措辞。

## Fail

若以下任一发生：

- source reconstruction 无法闭合；
- 主要 ranking instability 对正确 shear conversion 不稳健；
- 538 pairing 对严格重建不稳健；
- 报告不可从 raw data 重现。

此时停止后续科学工作。

---

# 8. 执行边界

本阶段不允许：

- 新 DFT；
- MP version shift；
- 第三协议；
- 完整 PULSE；
- 新论文结果写入；
- 追求模型性能；
- 根据结果调 matching tolerance；
- 删除失败样本；
- 覆盖旧 artifacts；
- 直接在 master 开发。

完成 `10_correctness_decision.md` 后停止，等待人工审核。

---

# 9. 最终汇报

必须逐项回复：

1. 发现了哪些真实 bug；
2. 哪些最初怀疑被证明不是 bug；
3. 每个 bug 影响哪些旧数字；
4. 修复前后测试；
5. source reconstruction 成功率；
6. 修复后 pair counts；
7. 修复后 ranking stability；
8. 哪些旧报告正式撤回；
9. correctness gate 结论；
10. 是否允许开展 version/third-protocol；
11. commit SHA 和 branch；
12. 未执行事项。
