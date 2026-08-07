# CrossPiezo Phase 7C：Reviewer-Proofing 与 Q1 决策

> 基线：`research/screening-resolution-v2` @ `7c19a3a2401bc37b9f3dcb680c66c052d54f4c71`  
> 新分支：`paper/screening-resolution-q1-v1`  
> 目标：清除 Phase 7B 最后几个统计与叙事风险，冻结 Q1 投稿稿。  
> 禁止：新 DFT、PULSE、PMR、componentwise/O(3)、修改 P0/P2。

## 1. 先冻结远程结果

远程 Phase 7B 完成后：

- 核对 commit、配置、manifest、CSV hash；
- 运行 `verify_phase7b.py`；
- 与本地结果逐项比较；
- 不一致则停止。

## 2. 修正高响应敏感性

当前 pooled-F1 top-50% 子集由两来源结果共同定义，可能产生选择/碰撞偏差。不得用其负 nAUCC 单独证明“不由 near-zero 驱动”。

重做三类分析：

1. `min(F_J,F_M) > threshold` 的双边高响应子集；
2. JARVIS 定义高响应、在 MP 上评价；反向再做一次；
3. pooled selection 的 conditional permutation：每次置换后重新执行 selection。

报告 full/P2、F1/F3/F4、N、nAUCC、band。旧 pooled 结果标为 exploratory。

## 3. 细化 screening-resolution story

将 1%–50% 曲线拆成：

- elite：1%–10%；
- intermediate：10%–20%；
- broad：20%–50%。

分别报告 partial nAUCC 和 simultaneous band。

措辞规则：

- 不统一声称“broad regions agree”；
- 若仅 F3/F4 在约 40% 后持续越过阈值，写成：
  `concordance emerges only after substantially broadening the candidate pool`;
- F1 无 persistent onset 时必须明确。

## 4. Control provenance

对 volume、band gap、energy above hull、dielectric trace 建字段审计表：

```text
source field
source database
unit
calculation type/version
missingness
是否可能由另一来源复制/补全
```

检查同一材料的值不是同字段复制造成的高一致性。

保留同-universe `Delta_tau`、`Delta_nAUCC`、grouped CI、FDR。禁止由 controls 推断“主要来自 convention”；只允许说响应属性具有更高 workflow sensitivity。

## 5. Portfolio 公平评估

`balanced_union` 在 budget=2 时覆盖两个 source top-k 是构造性上界，不得作为方法增益。

主结果：

- equal budget `b=1.0`；
- cost–recall/NDCG frontier；
- `b=1.5/2.0` 作为预算敏感性；
- balanced union @2.0 标为 coverage upper bound；
- deterministic strategies 在 full P0/P2 评价；
- 仅有调参的 abstention 使用 nested grouped CV；
- 至少 5-fold grouped cross-evaluation；
- CI 比较 source-robust strategy 与两个 single-source baselines；
- 报告 paired difference CI，而非只比较点估计。

不得称这些简单 rank aggregators 为新算法。

## 6. 论文冻结

生成：

```text
CrossPiezo_ScreeningResolution_Manuscript_v0.6.tex
CrossPiezo_ScreeningResolution_references.bib
```

要求：

- 完整相关工作，≥30 篇可靠文献；
- 删除或修正：
  `large fraction stems from conventions`；
  `high-response negative nAUCC confirms...`；
  `balanced_union=1.0` 作为主要增益；
- 主图：
  1. resolution curve + elite/intermediate/broad；
  2. corrected high-response sensitivity；
  3. property controls；
  4. equal-budget portfolio frontier；
  5. P0/P2 summary；
- 所有数字由 manifest 追踪；
- 编译 PDF。

## 7. 第三协议预注册

不执行计算。更新 48-material plan：

- consensus elite；
- JARVIS-only elite；
- MP-only elite；
- consensus low；
- 平衡晶系、原子数、P2；
- 冻结 F1/F3/F4 与 portfolio 策略；
- 评价 Recall@k、NDCG、regret；
- 预算和失败策略。

输出 `configs/third_protocol_phase7c.yaml`。

## 8. Final Gate

### Q1 Manuscript Ready

全部满足：

1. 远程/本地结果 hash 一致；
2. corrected high-response 分析仍支持 elite-tail gap；
3. controls provenance 闭合且至少两个对照显著更一致；
4. equal-budget portfolio paired CI 优于单来源，或诚实降为决策示例；
5. v0.6 编译且 claim traceability 100%。

### Strong Q1 Requires Adjudication

若材料论文成立但缺少独立决策验证，明确建议先做 48 个第三协议再投 npj/Communications Materials。

### Benchmark Venue

若 equal-budget portfolio 或 corrected sensitivity 不成立，回到 Digital Discovery。

输出：

```text
reports/phase7c/07_final_decision.md
```

完成后停止。
