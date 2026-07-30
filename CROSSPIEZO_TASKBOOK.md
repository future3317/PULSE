# CrossPiezo / PULSE：Claude Code 项目任务书

> 版本：v0.1  
> 日期：2026-07-29  
> 项目工作目录：当前目录  
> 原始数据根目录：`E:/DATA`  
> 论文草稿：当前目录下的 `PiezoProtocol_Draft_v0.1.tex`  
> 参考文献：当前目录下的 `PiezoProtocol_references.bib`  
> 项目原则：**先做严格可行性审计；Go/No-Go 通过前，不开发完整 PULSE，不填写论文结果。**

---

## 0. 你的角色

你是本项目的研究工程代理。你的工作不是立刻实现一个复杂模型，而是：

1. 读取论文 LaTeX 初稿，提取已经预注册的科学问题、数据契约、假设和停止条件；
2. 对 `E:/DATA` 做只读扫描，找到实际可用的数据集、manifest、审计报告和 split；
3. 建立可复现的跨数据库压电张量严格匹配管线；
4. 完成最小可行性分析；
5. 基于冻结结果给出 Go / Narrow / No-Go 判断；
6. 只有人工明确批准后，才进入 PULSE 模型开发。

不得把“代码已经写好”当成科学结论。所有主张必须由版本化脚本生成的结果支持。

---

# 1. 项目核心命题

论文暂定标题：

> **Protocol-Induced Uncertainty Limits AI Screening of Piezoelectric Materials**

核心问题不是普通的数据清洗，也不是给模型添加一个 source token，而是检验：

1. 严格匹配的同一材料，在 JARVIS 与 Materials Project 等计算协议下，压电张量差异有多大？
2. 这种协议差异是否已经达到或超过单一数据库内 SOTA 模型误差？
3. 差异是否被软光学模、离子压电比例、Born 有效电荷和内应变耦合放大？
4. 常规随机拆分得到的模型排行榜，在 source-held-out、version-held-out 和 paired-counterfactual 测试中是否稳定？
5. 协议感知的不确定性是否会实质改变压电材料筛选结果？

论文中的计划名称：

- **CrossPiezo**：严格匹配、来源和版本绑定的跨协议基准。
- **PULSE**：Protocol-Uncertainty Learning for Symmetry-Equivariant tensors。

---

# 2. 不可违反的科学边界

## 2.1 不得伪造“真实张量”

只有两个数据库时，不能把平均张量称为物理真值：

```text
(JARVIS + MP) / 2
```

最多称为：

```text
computational center
```

所有模型输出必须区分：

- source-conditional prediction；
- protocol disagreement；
- epistemic uncertainty；
- source residual；
- structure-match uncertainty。

## 2.2 不得用化学式匹配代替结构匹配

只有化学式相同的记录不得进入主 paired benchmark。

匹配等级：

- Tier 0：相同上游结构/provenance，并验证原子映射；
- Tier 1：严格周期结构匹配，原子、晶格和点群映射通过；
- Tier 2：同原型但弛豫结构差异明显；
- Tier 3：仅化学式相同。

主分析只使用 Tier 0–1。Tier 2 只分析结构介导差异。Tier 3 不算配对标签。

## 2.3 不得静默修复张量

以下转换必须全部写入 transformation history：

- 原始 Voigt 顺序；
- 内部 Voigt 顺序；
- engineering shear / tensor shear；
- 单位；
- 压电应力张量 `e` 或压电应变张量 `d`；
- Cartesian 展开；
- 晶胞变换；
- 原子映射；
- proper/improper rotation；
- 符号或转置；
- relaxed-ion / clamped-ion / ionic / electronic。

遇到不明确的 convention，记录并 quarantine，不得猜测。

## 2.4 不得污染现有项目

`E:/DATA` 中的 PiezoJet、T2C-Flow、GaugeFlow、EviMem-RL 等均视为只读外部资产。

不得：

- 修改文件；
- 覆盖 manifest；
- 重写 split；
- 访问或使用冻结测试标签进行开发；
- 移动或复制大型数据；
- 将第三方数据提交到当前 Git 仓库。

## 2.5 不得提前写结果

在可行性审计完成前：

- 不修改 LaTeX 的 Abstract、Results、Discussion 中的 `TBD` 为具体结论；
- 不编造表格数字；
- 不以预计结果更新论文；
- 可以新增 `manuscript_notes/`，记录方法与实际数据不一致之处。

---

# 3. 第一轮执行范围

你现在只执行：

- Phase 0：环境、论文和范围审计；
- Phase 1：`E:/DATA` 只读资产扫描；
- Phase 2：数据契约、tensor convention 和结构匹配脚手架；
- Phase 3：最小可行性审计；
- Phase 4：Go / Narrow / No-Go 报告。

完成 Phase 4 后停止，等待人工批准。

现在不得执行：

- 完整 PULSE 训练；
- 大规模超参数搜索；
- 全量 SOTA 复现；
- 新 DFT/DFPT；
- 联网下载数据；
- 自动修改论文结果；
- 材料候选“发现”声明。

---

# 4. 建议仓库结构

在当前目录建立，不移动已有 `.tex` 和 `.bib`：

```text
.
├── CLAUDE.md
├── CROSSPIEZO_TASKBOOK.md
├── PiezoProtocol_Draft_v0.1.tex
├── PiezoProtocol_references.bib
├── pyproject.toml
├── uv.lock
├── README.md
├── configs/
│   ├── paths.example.yaml
│   ├── data_sources.yaml
│   ├── conventions.yaml
│   ├── matching.yaml
│   ├── feasibility.yaml
│   └── logging.yaml
├── src/crosspiezo/
│   ├── __init__.py
│   ├── cli.py
│   ├── logging.py
│   ├── schemas/
│   │   ├── provenance.py
│   │   ├── structures.py
│   │   ├── tensors.py
│   │   ├── matches.py
│   │   └── audit.py
│   ├── inventory/
│   │   ├── scanner.py
│   │   ├── manifests.py
│   │   └── reports.py
│   ├── conventions/
│   │   ├── voigt.py
│   │   ├── units.py
│   │   ├── cartesian.py
│   │   ├── rotations.py
│   │   └── symmetry.py
│   ├── adapters/
│   │   ├── base.py
│   │   ├── jarvis_piezo.py
│   │   ├── mp_piezo.py
│   │   ├── piezojet.py
│   │   ├── t2c_flow.py
│   │   ├── mp_dielectric.py
│   │   └── mp_elastic.py
│   ├── matching/
│   │   ├── normalize.py
│   │   ├── structure_matcher.py
│   │   ├── atom_mapping.py
│   │   ├── tensor_transport.py
│   │   └── tiers.py
│   ├── analysis/
│   │   ├── discrepancy.py
│   │   ├── ranking.py
│   │   ├── protocol_floor.py
│   │   ├── soft_mode.py
│   │   └── statistics.py
│   ├── baselines/
│   │   └── simple_models.py
│   ├── audit/
│   │   ├── leakage.py
│   │   ├── claims.py
│   │   └── reproducibility.py
│   └── reports/
│       ├── figures.py
│       └── tables.py
├── tests/
│   ├── unit/
│   ├── conventions/
│   ├── matching/
│   ├── invariance/
│   ├── data_contracts/
│   └── regression/
├── scripts/
│   ├── scan_e_data.py
│   ├── build_inventory.py
│   ├── build_strict_pairs.py
│   ├── run_feasibility_audit.py
│   └── compile_status_report.py
├── data/
│   ├── README.md
│   ├── manifests/
│   ├── interim/
│   └── processed/
├── artifacts/
│   ├── inventories/
│   ├── pair_manifests/
│   ├── feasibility/
│   └── logs/
├── reports/
│   ├── 00_environment_and_scope.md
│   ├── 01_data_inventory.md
│   ├── 02_convention_audit.md
│   ├── 03_pairing_audit.md
│   ├── 04_feasibility_results.md
│   ├── 05_go_no_go.md
│   └── open_questions.md
└── docs/
    ├── data_contract.md
    ├── tensor_conventions.md
    ├── matching_protocol.md
    ├── statistical_plan.md
    └── claim_boundary.md
```

---

# 5. 技术栈

最低要求：

- Python 3.11+
- `uv` 与标准 `pyproject.toml`
- NumPy、SciPy
- pandas、PyArrow、DuckDB
- Pydantic v2
- ASE
- pymatgen
- spglib
- xarray 或 Zarr
- Typer、Rich
- pytest、Hypothesis
- Ruff、mypy、pre-commit

第一轮不要求 PyTorch。只有进入模型阶段后才引入：

- PyTorch
- PyTorch Geometric
- e3nn
- GPyTorch 或等价概率工具

---

# 6. Phase 0：环境、论文与范围审计

## 6.1 读取文件

必须首先读取：

- `PiezoProtocol_Draft_v0.1.tex`
- `PiezoProtocol_references.bib`
- 当前目录内其他任务书、README、报告或代码
- `E:/DATA` 根目录下的报告、MANIFEST、README、audit、split 和 license 文件

不要只搜索固定文件名。先递归列出候选，但避免读取大型二进制内容。

## 6.2 提取 LaTeX 合同

从 `.tex` 中生成：

```text
reports/00_environment_and_scope.md
docs/claim_boundary.md
docs/statistical_plan.md
```

至少提取：

- 研究问题；
- 数据资产；
- 主公式；
- 数据匹配等级；
- PULSE 的建模假设；
- evaluation splits；
- preregistered hypotheses；
- stop conditions；
- 所有 `TBD`；
- LaTeX 中与实际文件可能不一致的内容。

## 6.3 环境报告

记录：

- OS 和 shell；
- Python 版本；
- 可用磁盘；
- 是否可访问 `E:/DATA`；
- 是否安装 Git、uv、LaTeX；
- 可用 CPU/GPU；
- 所有外部软件只记录，不自动安装重型依赖。

若 `E:/DATA` 不可访问，立即停止，并在 `reports/open_questions.md` 写明。

---

# 7. Phase 1：E:/DATA 只读扫描

## 7.1 扫描规则

默认只读取：

- 文件路径；
- 大小；
- 修改时间；
- 扩展名；
- README；
- MANIFEST；
- audit report；
- schema；
- split；
- license；
- 小型元数据或表头。

不得默认计算所有多 GB 文件的完整 SHA256。采用：

- 小文件：完整 SHA256；
- manifest、split、schema、audit：完整 SHA256；
- 大文件：大小、mtime、头尾采样 hash；
- 提供 `--full-hash` 显式选项，但第一轮不启用。

## 7.2 优先定位的资产

实际文件名可能不同，使用语义搜索和目录扫描定位：

```text
E:/DATA/AI4Materials_Dataset_Report.md
E:/DATA/PiezoJet/
E:/DATA/T2C-Flow/
E:/DATA/T2C-Flow/processed/piezo_unified.*
E:/DATA/T2C-Flow/MANIFEST.json
E:/DATA/Tpami/mp_dielectric/
E:/DATA/Tpami/mp_elastic/
E:/DATA/MatPES-R2SCAN-2025.2/
E:/DATA/LeMat-BulkUnique/
E:/DATA/EviMem-RL/
```

重点查找：

- JARVIS piezo records；
- MP piezo records；
- JARVIS–MP mapping；
- structure/CIF/atoms；
- source_database；
- material_id；
- total/electronic/ionic piezo；
- dielectric；
- elastic；
- space group；
- original MP ID；
- internal strain；
- Born charges；
- force constants；
- split hashes；
- frozen test manifests；
- provenance 和 audit reports。

## 7.3 输出

生成：

- `artifacts/inventories/data_inventory.parquet`
- `artifacts/inventories/data_inventory.json`
- `reports/01_data_inventory.md`

报告必须区分：

- 实际存在；
- 文档声称但未找到；
- 可解析；
- 需要适配器；
- 许可不明确；
- frozen/holdout；
- 可用于主分析；
- 只能作为辅助；
- 不得使用。

---

# 8. Phase 2：数据契约和张量约定

## 8.1 核心 schema

实现以下 Pydantic 数据类型。

### `SourceArtifact`

```text
source_name
source_version
path
sha256_or_fingerprint
license
parser_version
frozen_status
```

### `StructureRecord`

```text
source_name
material_id
source_structure_id
formula
atomic_numbers
lattice
fractional_coordinates
space_group
primitive_or_conventional
structure_hash
provenance
```

### `TensorRecord`

```text
structure_key
tensor_type
contribution
raw_shape
raw_voigt_order
internal_voigt_order
shear_convention
unit
cartesian_tensor
point_group
symmetry_residual
source_functional
source_code
transformation_history
```

### `MatchRecord`

```text
left_structure_key
right_structure_key
match_tier
unimodular_cell_transform
cartesian_rotation
atom_permutation
lattice_distance
site_distance
space_group_relation
ambiguity
pass_fail_reasons
```

### `AuditEvent`

```text
timestamp
artifact
check_name
status
message
code_commit
config_hash
```

## 8.2 内部 convention

内部默认：

- 压电应力张量 `e`；
- 完整 Cartesian `3×3×3`；
- 最后两个应变指标对称；
- SI：`C/m^2`；
- 内部 engineering Voigt 顺序：
  `xx, yy, zz, yz, xz, xy`；
- 转成 Cartesian 后再比较；
- 原始张量和转换后张量都保留。

如果源数据是 `d`，不得直接和 `e` 比较。只有存在同源、自洽的完整弹性/柔度时才转换，并保留不确定性和来源。

## 8.3 必须测试

- source Voigt → Cartesian → source Voigt round trip；
- engineering shear；
- 二阶、三阶和四阶张量旋转；
- proper/improper parity；
- 随机 O(3) 协变；
- point-group Reynolds 投影；
- unit conversion；
- 晶胞改变后张量传输；
- 原子 permutation 不改变晶体级张量；
- 零张量和近零张量；
- 已知晶系允许分量。

所有测试不得使用伪造科学结果，可使用合成张量做代数测试。

---

# 9. 严格结构匹配协议

## 9.1 匹配顺序

1. 先用明确 cross-reference 或原始 MP ID 找候选；
2. 再用 formula、atom count 和 species set 缩小范围；
3. 用 spglib/pymatgen 标准化，但保留原始结构；
4. 尝试 primitive/conventional cell 映射；
5. 使用冻结容差的 `StructureMatcher`；
6. 输出具体 unimodular transform、rotation 和 atom permutation；
7. 验证正向/反向重建；
8. 检查点群和极性；
9. 运输张量到共同 Cartesian frame；
10. 分配 Tier 和 quarantine 原因。

## 9.2 不允许

- 根据张量值选择匹配；
- 根据化学式强行一一对应；
- 先看差异再调匹配容差；
- 自动取多个候选中差异最小的那个；
- 将空间群变化静默忽略。

## 9.3 容差

容差必须写入 `configs/matching.yaml` 并在运行前冻结。

首轮可以提出建议值，但必须做：

- 容差敏感性分析；
- 匹配唯一性统计；
- 正向/反向一致性；
- 边界样本人工报告。

## 9.4 输出

- `artifacts/pair_manifests/strict_pairs.parquet`
- `artifacts/pair_manifests/quarantined_pairs.parquet`
- `artifacts/pair_manifests/match_config_hash.txt`
- `reports/03_pairing_audit.md`

---

# 10. Phase 3：最小可行性审计

完成严格配对后，按冻结配置运行以下分析。

## 10.1 分析 A：严格配对数量

报告：

- preliminary overlap；
- Tier 0；
- Tier 1；
- Tier 2；
- Tier 3；
- quarantine；
- 每种失败原因；
- 化学、晶系、空间群和幅值分布；
- 与全部数据库的选择偏差。

## 10.2 分析 B：跨协议差异

至少报告：

- absolute Frobenius；
- relative Frobenius；
- cosine；
- amplitude ratio；
- sign flip；
- principal response direction；
- irreducible/channel-wise discrepancy；
- symmetry projection前后差异；
- 电子/离子/总张量，若双方可得；
- 结构匹配距离分层；
- 晶系和化学分层。

禁止只报告平均值。必须报告：

- median；
- IQR；
- 5/95 percentile；
- bootstrap CI；
- tail cases；
- near-zero target 的独立处理。

## 10.3 分析 C：协议差异与模型误差

第一轮不必复现全部 SOTA。采用两级方案：

### C0：文献可比尺度

从 LaTeX/BibTeX 和本地报告中提取已有公开误差，但必须记录：

- 数据集；
- split；
- metric；
- unit；
- tensor convention；
- 是否真正可比。

不可比的数字不得放进同一比值。

### C1：轻量可复现 baseline

训练或运行一个简单而透明的 baseline，仅在数据和 split 已冻结后：

- composition/statistical baseline；
- CGCNN/简单结构模型，如本地已有可靠实现；
- source-specific 与 pooled；
- 多随机种子。

目的只是估计内部模型误差尺度，不声称 SOTA。

计算 PMR 或等价 protocol-to-model ratio，并提供指标定义。

## 10.4 分析 D：排名稳定性

预注册至少两个材料级功能：

1. `||e||_F` 或 symmetry-adapted norm；
2. 最大方向响应或另一个由完整 tensor 明确定义的指标。

如果同源 `C` 和 `epsilon` 可用，再分析：

- `d = e C^{-1}`；
- 方向相关响应；
- 明确的 coupled metric。

报告：

- top-20、50、100 Jaccard；
- Kendall tau；
- Spearman；
- rank shifts；
- 高性能/高争议候选；
- 高性能/高一致候选。

不得在看到结果后选择最不稳定的 metric 作为主 metric。

## 10.5 分析 E：软模机制可行性

在能与 PiezoJet strict factors 严格交叉的子集上：

- 计算/读取 optical force constants；
- Born effective charges；
- internal strain；
- electronic/ionic piezo；
- minimum optical eigenvalue；
- ionic fraction；
- `S_soft` 或 LaTeX 中预注册的敏感性指标。

比较：

- chemistry-only 模型；
- structure mismatch 模型；
- soft-mode/microscopic factor 模型；
- nested cross-validation；
- formula/prototype grouped split；
- 排除近不稳定 tail 后的结果。

不满足稳定性、factor convention 或 exact mapping 的记录不得进入。

---

# 11. Go / Narrow / No-Go 判定

## 11.1 Full Go：继续 PULSE 和强论文

建议至少满足：

1. Tier 0–1 严格配对 `N >= 300`；
2. convention 修复后仍存在显著残余协议差异；
3. 满足以下至少一项：
   - `PMR >= 0.5`，且置信区间不是纯偶然；
   - 预注册 top-50 Jaccard `<= 0.70`；
   - 预注册排名 Kendall tau `<= 0.80`；
4. source-aware uncertainty 或 soft-mode 机制至少有一个清晰、可复现的增益方向；
5. 结果不是由少量 near-zero、结构错配或单一化学族驱动。

## 11.2 Narrow Go：做 CrossPiezo/benchmark 论文

适用情况：

- 配对数量足够；
- 协议差异真实；
- 但 soft-mode 机制弱；
- 或 source-aware 模型增益有限。

项目收窄为：

- 数据与版本审计；
- benchmark；
- leaderboard stress test；
- calibration protocol；
- 不强行发展复杂 PULSE。

## 11.3 No-Go

出现以下任一情况：

- Tier 0–1 `N < 150`；
- 大部分差异来自可修复 convention 错误；
- 协议差异远小于合理模型误差；
- 候选排名高度稳定；
- strict-factor 交集太小，无法支持机制结论；
- 数据许可或 provenance 不能支持发布和重建。

`150 <= N < 300` 时由人工决定是否只做数据/短文，不自动继续模型。

## 11.4 输出

生成：

```text
reports/04_feasibility_results.md
reports/05_go_no_go.md
artifacts/feasibility/summary.json
artifacts/feasibility/frozen_metrics.parquet
artifacts/feasibility/figures/
```

`05_go_no_go.md` 必须包含：

- 明确判定；
- 支持证据；
- 反证；
- 数据局限；
- 下一阶段预算；
- 应删除或修改的 LaTeX claim；
- 不确定事项。

完成后停止。

---

# 12. 测试和质量要求

每个阶段运行：

```bash
uv run ruff check .
uv run mypy src/crosspiezo
uv run pytest
```

最低要求：

- 所有 public API 有类型注解；
- 核心 schema 和 convention 使用 strict mypy；
- 不允许 `except Exception: pass`；
- 不允许 silent fallback；
- 随机过程有 seed；
- 配置和 split 有 hash；
- 报告和图表由脚本生成；
- 大文件流式读取；
- 任意数据转换可追踪；
- Windows 路径必须使用 `pathlib`；
- 不把 `E:/DATA` 路径写死在代码内部，只写配置示例。

---

# 13. CLI

第一轮实现：

```bash
crosspiezo doctor
crosspiezo manuscript audit --tex PiezoProtocol_Draft_v0.1.tex
crosspiezo data scan --root E:/DATA --dry-run
crosspiezo data inventory --config configs/data_sources.yaml
crosspiezo conventions validate
crosspiezo pairs build --config configs/matching.yaml --dry-run
crosspiezo pairs audit
crosspiezo feasibility run --config configs/feasibility.yaml
crosspiezo report status
```

要求：

- 默认不联网；
- 默认不写入 `E:/DATA`；
- 所有危险操作支持 `--dry-run`；
- 大规模执行前打印估计数据量、内存和时间；
- 失败返回非零退出码；
- JSON 日志可选；
- 报告中记录 command、config hash、commit。

---

# 14. 完成标准

第一轮完成后，研究者应能运行：

```bash
uv sync
uv run pytest
uv run crosspiezo doctor
uv run crosspiezo manuscript audit --tex PiezoProtocol_Draft_v0.1.tex
uv run crosspiezo data scan --root E:/DATA --dry-run
uv run crosspiezo data inventory --config configs/data_sources.yaml
uv run crosspiezo conventions validate
uv run crosspiezo pairs build --config configs/matching.yaml
uv run crosspiezo pairs audit
uv run crosspiezo feasibility run --config configs/feasibility.yaml
uv run crosspiezo report status
```

并得到：

- LaTeX 科学合同；
- 本地数据清单；
- 张量 convention 审计；
- 严格配对 manifest；
- quarantine manifest；
- 跨协议差异；
- 排名稳定性；
- 初步协议差异与模型误差比较；
- soft-mode 机制 feasibility；
- Go / Narrow / No-Go 报告。

---

# 15. 完成后向研究者汇报

最终回复必须包含：

1. 新建和修改的文件；
2. 所有测试结果；
3. `E:/DATA` 中实际找到的数据；
4. 实际 pair 数量；
5. convention 问题；
6. 主要图表和指标；
7. Go / Narrow / No-Go；
8. 与 LaTeX 计划不一致之处；
9. 下一步建议；
10. 明确说明没有执行哪些任务。

完成 Phase 4 后不要自行进入 PULSE 模型开发。
