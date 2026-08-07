# CrossPiezo Phase 7A：Screening Resolution、Controls 与 Robust Portfolio

> 基线分支：`paper/invariant-manuscript-freeze-v1`  
> 基线提交：`687351861862e3460994bf9bab7f8d5e734ec153`  
> 新分支：`research/screening-resolution-v1`  
> 数据：`E:/DATA` 或 `~/DATA`，永久只读  
> 目标：把当前“weak concordance benchmark”提升为具有机制线索和决策价值的一区材料论文，并形成独立的 CCF-A 方法论文立项判断。

## 1. 角色与原则

你是材料信息学研究工程代理。以冻结数据、脚本生成结果和可追踪证据为准，不以保留现有 story 为目标。

允许：本地数据分析、轻量统计/排序方法、定向文献检索、论文重写。  
禁止：新 DFT、PULSE、componentwise tensor、source-native frame、PMR、soft-mode 因果主张、改动冻结的 573 对主 panel。

所有新结果写入：

```text
artifacts/phase7a/
results/phase7a/
reports/phase7a/
```

不覆盖 Phase 6A/6B。

---

## 2. 核心研究问题

从“两个数据库不一致”升级为：

> 两个数据库能否识别宽泛的高响应区域，但无法稳定解析最顶尖候选？这种筛选分辨率缺口是否为压电响应所特有，能否通过跨来源组合策略缓解？

---

## 3. Work Package A：筛选分辨率曲线

对 P0、P2 和 F1/F3/F4，令筛选比例 \(q=1\%\ldots50\%\)，计算：

- observed overlap/Jaccard；
- exact hypergeometric null；
- exact chance-adjusted Jaccard；
- bootstrap CI；
- overlap enrichment；
- source rank displacement。

新增两个预注册摘要量：

```text
AUC-Concordance: q-curve 的面积
q_consensus: adjusted overlap 的下置信界首次超过冻结阈值的 q
```

阈值和算法在运行前写入 `configs/phase7a.yaml`。不得按结果挑 q。

输出：

```text
reports/phase7a/01_screening_resolution.md
results/phase7a/concordance_curve.csv
```

---

## 4. Work Package B：尺度、顺序与尾部拆解

分别分析：

1. **Scale shift**：ECDF、分位数比、log Bland–Altman、来源幅值偏移。
2. **Order shift**：原始 rank 与 source-wise quantile-normalized rank。
3. **Tail shift**：top-q overlap、threshold crossing、rank-flow。

回答：简单单调校准能否恢复候选一致性？

输出：

```text
reports/phase7a/02_scale_order_tail.md
```

---

## 5. Work Package C：负对照属性

在相同或可明确对齐的结构匹配材料上，优先比较：

- 体积（必须）；
- 带隙；
- 形成能/稳定性；
- dielectric invariant；
- elastic invariant。

规则：

- 先报告各属性可用 N；
- 使用属性自己的共同 universe；
- 使用相同 q-curve、tau、rank displacement；
- 用 grouped bootstrap 比较压电与对照属性的 concordance；
- 不因缺失而伪造或插补主结果。

目标：检验压电高阶响应是否比简单结构/标量属性更难跨库复现。

输出：

```text
reports/phase7a/03_property_controls.md
results/phase7a/property_controls.csv
```

---

## 6. Work Package D：电子/离子贡献

仅在双方均有 convention-complete 的 electronic、ionic、total tensors 时执行。

比较 F1/F3/F4 的：

- total；
- electronic；
- ionic；
- cancellation index；
- 各贡献的 q-curve 和 tau。

允许结论：

> ionic/electronic contribution shows stronger or weaker cross-database concordance.

禁止结论：

> soft modes or a specific DFT setting causes the discrepancy.

若有效 N < 100，降为探索性补充。

输出：

```text
reports/phase7a/04_electronic_ionic_decomposition.md
```

---

## 7. Work Package E：异质性与匹配敏感性

分析 concordance 与以下变量的关系：

- 晶系/点群；
- 化学系统与阴离子族；
- 原子数；
- 极性；
- 重元素比例；
- response magnitude；
- RMS/lattice mismatch。

使用：

- grouped bootstrap；
- permutation；
- FDR correction；
- 连续 match-quality 曲线。

不得按结果重新定义主 panel。

输出：

```text
reports/phase7a/05_heterogeneity.md
```

---

## 8. Work Package F：跨来源稳健候选组合

以 source 内 percentile rank 为输入，比较：

- JARVIS-only；
- MP-only；
- average/Borda；
- maximin；
- consensus intersection；
- union portfolio；
- disagreement-aware abstention。

在没有真值的条件下，用以下决策指标：

```text
worst-source Recall@k
worst-source NDCG@k
worst-source rank regret
portfolio size / coverage
```

必须明确：这是 two-source robustness，不是物理验证。

输出：

```text
reports/phase7a/06_robust_portfolio.md
results/phase7a/portfolio_benchmark.csv
```

---

## 9. Work Package G：论文 story 与文献

定向检索并记录高质量来源，覆盖：

- tensor prediction；
- cross-database DFT variation；
- material dataset shift；
- ranking/top-k stability；
- robust ranking/decision；
- FAIR/provenance。

保存：

```text
results/phase7a/literature_matrix.csv
```

重写主稿但不覆盖旧稿：

```text
CrossPiezo_ScreeningResolution_Manuscript_v0.4.tex
CrossPiezo_ScreeningResolution_references.bib
```

推荐 story：

> Databases agree on broad promising regions but cannot reproducibly resolve the elite tail; CrossPiezo quantifies this screening-resolution gap and evaluates source-robust candidate portfolios.

主图建议：

1. 数据与分析流程；
2. screening concordance curve；
3. scale/order/tail decomposition；
4. property controls；
5. electronic/ionic decomposition（若成立）；
6. robust portfolio frontier。

---

## 10. Work Package H：CCF-A 方法论文立项书

不实现复杂模型。生成：

```text
docs/ccfa_method_paper_concept.md
```

包含：

- 通用问题：conflicting scientific labels 下的 source-robust ranking；
- 可能方法：DRO ranking、set-valued labels、abstention；
- 可证明的理论命题；
- 至少 3 个可构建的多来源 benchmark；
- baselines；
- held-out-source protocol；
- 计算预算；
- Go/No-Go 风险。

CrossPiezo 只能是其中一个 benchmark。

---

## 11. 验收与决策

### Q1 Material Go

同时满足：

1. elite-tail resolution gap 在 P2、高响应子集和至少两个指标上稳定；
2. 至少一个负对照属性明显比压电更一致，或电子/离子拆解提供可靠材料学解释；
3. robust portfolio 在 worst-source 指标上优于两个单来源策略；
4. 独立统计验证通过；
5. 新稿所有数字可追踪。

### Benchmark-Only

若排序不稳成立，但对照、拆解和 portfolio 没有新增洞察，则保持 Digital Discovery/Scientific Data 定位。

### CCF-A Method Go

只有在通用任务可扩展到至少 3 类多来源科学标签，并存在明确方法与理论增量时通过。

最终输出：

```text
reports/phase7a/07_phase7a_decision.md
```

完成后停止，不执行第三协议或方法模型。
