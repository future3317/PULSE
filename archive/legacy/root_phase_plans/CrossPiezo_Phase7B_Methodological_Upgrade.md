# CrossPiezo Phase 7B：方法复核与一区论文升级

> 基线：`research/screening-resolution-v1` @ `ede4066921b4d22bcc5688ae91febcb429871dbb`  
> 新分支：`research/screening-resolution-v2`  
> 数据：`E:/DATA` 或 `~/DATA`，只读  
> 目标：修正 Phase 7A 的统计与 portfolio 问题，形成可审稿的 screening-resolution 论文。  
> 禁止：新 DFT、PULSE、PMR、componentwise/O(3)、soft-mode 因果主张、修改 P0/P2 成员。

## 1. 已知问题：先写 red tests

1. `AUC-Concordance` 使用百分比横轴，数值 5.15 不是“5.15%”；改为归一化 nAUCC。
2. `q_consensus` 是 50 个 pointwise CI 中首次越零，存在多重比较和偶然 crossing；改为 simultaneous band + persistent onset。
3. source-wise quantile normalization 是单调变换，Kendall tau/top-k 必然不变；不得作为实验发现。
4. property-control 报告的差值符号解释反了；所谓 grouped bootstrap 未真正实现。
5. portfolio：
   - Borda 排序方向错误；
   - NDCG 的 ideal 仅在 selected set 内计算，不能评价选集质量；
   - rank regret 实际是 selected ranks 的极差；
   - `set(list(...))` 截断不确定；
   - selection budget 与 target top-k 混为同一个 k；
   - union 的 0.5 recall 部分由构造保证。
6. 阴离子分类使用字符串子串，`Se` 会被判作 sulfide，`Ho/Co` 可能被判作 oxide。
7. Phase 7A “Material Go” 中独立验证写成 `pass by construction`，不能作为 gate。
8. v0.4 只是 51 行骨架，literature matrix 仅 6 条，不是完整论文。

所有问题至少一个测试。旧 Phase 7A 数字保留但标记 provisional。

## 2. Screening-resolution 重算

对 P0/P2、F1/F3/F4、q=1%–50%：

- exact hypergeometric null；
- chance-adjusted Jaccard；
- paired bootstrap simultaneous 95% band；
- normalized AUC：

\[
\mathrm{nAUCC}=\frac{1}{q_{\max}-q_{\min}}\int C(q)\,dq;
\]

- persistent consensus onset：simultaneous LCB 连续至少 5 个 q 超过阈值；
- 主阈值 `delta=0.05`，补充 `0` 和 `0.10`；
- high-response 和 P2 sensitivity。

输出：

```text
results/phase7b/concordance_curve.csv
results/phase7b/concordance_summary.csv
```

独立脚本重算主结果，不复用主函数。

## 3. Scale / order / tail

删除“quantile normalization 不改变 tau”的发现。

改为：

- scale：ECDF、log-ratio、quantile mapping 后的 value error 和 threshold agreement；
- order：Kendall/Spearman、rank displacement；
- tail：top-q crossing 和 concordance curve；
- 明确：任何严格单调校准都不能改变 rank/top-q。

## 4. Property controls

使用体积、带隙、能量高于凸壳、介电迹；有数据再加入 formation energy/elastic。

要求：

- 记录每个属性的真实 source field、单位、版本和共同 N；
- 同 universe 分析 + available-case 分析；
- grouped paired bootstrap（按 reduced formula / matcher component）；
- 报告：

```text
Delta_tau = tau_control - tau_piezo
Delta_nAUCC = nAUCC_control - nAUCC_piezo
95% CI, permutation p, FDR
```

正值表示 control 更一致。不得再使用反向解释。

## 5. Heterogeneity

- 用 `pymatgen.Composition` 精确识别元素；
- 支持多阴离子类别，禁止字符串子串；
- subgroup N < 30 不作强结论；
- grouped bootstrap + FDR；
- 连续分析 RMS/lattice mismatch，不重新定义主 panel。

## 6. Robust portfolio：完全重写

固定：

- target elite fraction `q*=10%`；
- selection budget `b ∈ {1.0, 1.5, 2.0} × target size`；
- P0 primary，P2 sensitivity；
- F1/F3/F4。

策略：

- JARVIS-only；
- MP-only；
- average percentile；
- 正确 Borda；
- maximin percentile；
- intersection-first；
- balanced union；
- disagreement abstention；
- exact/near-exact minimax oracle。

指标：

- worst-source Recall of each source top-q；
- worst-source NDCG，IDCG 从全 universe 计算；
- worst-source normalized utility；
- minimax regret = source-specific optimum utility − achieved utility；
- budget/coverage Pareto frontier；
- bootstrap CI。

规则：

- 所有排序 deterministic；
- 不允许 set 随机截断；
- 有超参数的策略在 grouped development split 调参，在 frozen holdout 评估；
- 明确这是 two-source decision robustness，不是独立物理验证。

## 7. 独立审计与论文

脚本只产出 CSV/JSON，不在计算脚本中拼接长篇 Markdown。报告由独立 renderer 或读取 CSV 后编写。

生成：

```text
scripts/run_phase7b.py
scripts/verify_phase7b.py
results/phase7b/
reports/phase7b/
CrossPiezo_ScreeningResolution_Manuscript_v0.5.tex
CrossPiezo_ScreeningResolution_references.bib
```

论文要求：

- 完整 Introduction / Related Work / Methods / Results / Discussion；
- 30 篇以上高质量参考文献；
- 主 story：broad-region agreement vs elite-tail resolution limit；
- 主图：resolution curve、controls、scale/order/tail、portfolio frontier；
- 所有数字来自 manifest；
- 编译 PDF；
- 不覆盖旧稿。

## 8. CCF-A 立项复核

更新 `docs/ccfa_method_paper_concept_v0.2.md`：

- 至少 3 个真实多来源 benchmark；
- 通用 source-robust ranking objective；
- 理论命题；
- held-out-source protocol；
- baselines 和预算；
- `Method Go / No-Go`。

本轮不实现复杂模型。

## 9. Final Gate

### Q1 Material Go

全部满足：

1. nAUCC 与 persistent onset 在独立实现中一致；
2. P2/high-response 下 elite-tail gap 保持；
3. 至少两个 control 的 `Delta_tau` 或 `Delta_nAUCC` CI > 0；
4. portfolio 在 frozen holdout 上优于两个单来源策略，且指标定义正确；
5. 化学分类、FDR、missingness 审计通过；
6. v0.5 完整编译、结果可追踪。

否则：

```text
Benchmark Paper Only
```

输出：

```text
reports/phase7b/08_phase7b_decision.md
```

完成后停止，不执行第三协议。
