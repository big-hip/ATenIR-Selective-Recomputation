# 15 — 实验输出说明文档

> **文档定位**: 说明当前 3 个实验脚本的 CSV 输出、7 类论文图表 (F1-F7) 的读图方法和论文引用建议。
>
> 所有输出保存在 `toolkit_examples/outputs/` 目录下。

---

## 一、总览

### 实验脚本与 CSV 输出

| 脚本 | 论文章节 | 内容 | CSV 文件 |
|------|---------|------|---------|
| `ex_sim_accuracy.py` | §5.1 | 12 策略 L2/L2.5/L3 仿真精度 | `ex_sim_accuracy.csv` |
| `ex_peak_phase.py` | §5.2 | batch×optimizer 峰值阶段分析 | `ex_peak_phase.csv` |
| `ex_model_generalization.py` | §5.3 | 3 模型×6 策略通用性验证 | `ex_model_generalization.csv` |

另有 `ex1_multi_model_capture.py` 作为 Demo（无 CSV 输出，仅控制台）。

### 论文图表 (F1-F7)

由 `generate_paper_figures.py` 生成，输出至 `outputs/paper_figures/`，每张图同时保存 PDF + PNG 两种格式。

| 编号 | 内容 | 数据来源 | 文件数 |
|------|------|---------|-------|
| F1 | 三模型内存组成堆叠柱状图 | L1 估算 | 2 (PDF+PNG) |
| F2 | 12 策略总览 + Pareto 图 | ex_sim_accuracy.csv | 2 |
| F3 | RT vs L2 vs L2.5 vs L3 分组柱状图 | ex_sim_accuracy.csv | 2 |
| F4 | L2/L2.5/L3 MRE 对比 | ex_sim_accuracy.csv | 2 |
| F5 | 峰值阶段热力图 (per strategy) | ex_peak_phase.csv | 8 (4策略×2格式) |
| F6 | FW/BW/OPT 三阶段堆叠柱状图 | ex_peak_phase.csv | 4 (2策略×2格式) |
| F7 | 模型×策略 MRE 热力图 (per level) | ex_model_generalization.csv | 6 (3层×2格式) |

共 26 个文件。

---

## 二、CSV 字段说明

### ex_sim_accuracy.csv — 12 策略仿真精度

**模型配置**: LLaMA ~870M (16L/2048H/16heads/8kv/5504I), B=8, S=512, Adam(fused=True)

| 字段 | 含义 |
|------|------|
| `strategy` | 策略名（如 "S07 inductor(b=1.0)"） |
| `group` | 分组（G1 Eager / G2 Compiled / G3 AC+Compiled） |
| `rt_fw_peak` / `rt_bw_peak` / `rt_opt_peak` | 运行时各阶段峰值 (bytes) |
| `rt_fwbw_peak` / `rt_true_peak` | 运行时组合峰值 |
| `rt_peak_phase` | 运行时峰值阶段 (FW/BW/OPT) |
| `rt_step_ms` | 运行时单步时间 (ms) |
| `rt_base` | 运行时基础内存 (bytes) |
| `l2_fw_peak` ... `l2_true_peak` | L2 仿真各阶段峰值（G1 无值） |
| `l2_peak_phase` | L2 峰值阶段 |
| `mre` / `direction` | L2 MRE + 方向 (over/under) |
| `err_total` / `err_pct` / `err_fixed` / `err_act` / `err_alloc_overhead` | 误差分解 |
| `l25_true_peak` / `l25_mre` / `l25_direction` | L2.5 融合感知仿真 |
| `l3_true_peak` / `l3_mre` / `l3_direction` | L3 Scheduler 仿真 |

**关键特征**:
- G1 策略（eager）：有运行时数据，**无仿真数据**（不经过 torch.compile）
- G2 aot_eager 策略：L2 MRE ~1-3%（极高精度），无 L2.5/L3
- G2/G3 inductor 策略：L2 MRE ~28-38%（不建模 fusion），L2.5 MRE ~8-12%，L3 MRE ~5-7%

### ex_peak_phase.csv — 峰值阶段分析

**模型配置**: LLaMA ~870M, S=512

| 字段 | 含义 |
|------|------|
| `optimizer` | SGD / Adam / Adam(fused) |
| `strategy` | eager / inductor(b=1.0) / inductor(b=0.0) / ac+inductor |
| `batch` | 批次大小 (2/4/8/16) |
| `fw_peak` / `bw_peak` / `opt_peak` | 运行时各阶段峰值 (bytes) |
| `fwbw_peak` / `true_peak` | 运行时组合峰值 |
| `peak_phase` | 峰值阶段 (FW/BW/OPT) |
| `step_ms` | 单步时间 (ms) |
| `base` | 基础内存 (bytes) |

**关键特征**:
- **SGD**: base 最低（无优化器状态），peak 始终在 FW
- **Adam**: base 高（+momentum+variance），小 batch 时 peak 可能在 OPT
- **Adam(fused)**: opt_peak 降低（无 foreach 临时变量），OPT 不再成为瓶颈
- **B=16 + Adam/Adam(fused) + 非 AC 策略**: 超出 48GB GPU 内存 → OOM（无数据行）

### ex_model_generalization.csv — 多模型通用性

**配置**: GPT-2 124M / LLaMA ~870M / Mistral ~550M, B=8, S=512, Adam(fused=True)

| 字段 | 含义 |
|------|------|
| `model` | 模型名 (gpt2/llama/mistral) |
| `strategy` | eager / aot_eager+default / inductor(b=1.0) / inductor(b=0.0) / ac+inductor / sac_mm+inductor |
| `rt_true_peak` / `rt_fw_peak` / `rt_bw_peak` / `rt_opt_peak` | 运行时峰值 (bytes) |
| `rt_peak_phase` | 运行时峰值阶段 |
| `l2_true_peak` / `l25_true_peak` / `l3_true_peak` | 三层仿真峰值 |
| `l2_mre` / `l25_mre` / `l3_mre` | 三层 MRE |
| `l2_direction` / `l25_direction` / `l3_direction` | 估算方向 |
| `step_ms` | 单步时间 (ms) |

**关键特征**:
- eager 策略：仅有运行时数据（无仿真值）
- aot_eager 策略：L2 MRE 随架构差异较大（GPT-2 较高，LLaMA 较低）
- inductor 策略：L2.5 和 L3 显著优于 L2

---

## 三、论文图表 (F1-F7) 详细说明

### F1: Memory Composition — `F1_composition.{pdf,png}`

- **类型**: Stacked Bar Chart（三根柱子 = 三个模型）
- **含义**: 对 GPT-2 124M / LLaMA ~870M / Mistral ~550M，将 L1 估算的训练峰值内存分解为 param / grad / optimizer / activation 四部分
- **数据来源**: `estimate_from_config()` (L1 公式法)，使用与实验相同的放大版模型配置
- **读图方法**: 柱子总高度 = 估算训练峰值。红色（activation）部分越大，重计算策略的潜在收益越大
- **论文用途**: 第 3 章开篇，展示不同架构的内存组成差异，引出"activation 是主要优化目标"

### F2: Strategy Overview — `F2_strategy_overview.{pdf,png}`

- **类型**: 双子图（上: 柱状图 12 策略峰值 + Pareto 散点图）
- **数据来源**: `ex_sim_accuracy.csv`
- **含义**: 展示 12 种策略的 Runtime true_peak 对比 + 显存-时间权衡关系
- **副标题**: "LLaMA ~870M | B=8, S=512 | Adam (fused)"
- **论文用途**: §5.1 核心图，一图总览全部策略的内存效率排名 + Pareto 前沿

### F3: Peak Comparison — `F3_peak_comparison.{pdf,png}`

- **类型**: Grouped Bar Chart（每策略最多 4 根柱: RT / L2 / L2.5 / L3）
- **数据来源**: `ex_sim_accuracy.csv`
- **含义**: 对可仿真策略（G2/G3），并排展示运行时和三层仿真的峰值估算
- **读图方法**: 柱子越接近运行时（深色）= 仿真越准确。L2.5/L3 柱通常比 L2 柱更接近 RT
- **论文用途**: §5.1 第二张图，直观展示四层仿真的精度差异

### F4: MRE — `F4_mre.{pdf,png}`

- **类型**: Grouped Bar Chart（每策略最多 3 根柱: L2 MRE / L2.5 MRE / L3 MRE）
- **数据来源**: `ex_sim_accuracy.csv`
- **含义**: 量化展示每种策略在不同仿真层级下的 MRE 百分比
- **读图方法**: 柱子越矮 = 误差越小。aot_eager 策略 L2 已足够准确；inductor 策略需要 L2.5/L3 才能达到可用精度
- **论文用途**: §5.1 精度分析的核心数据图

### F5: Peak Phase Heatmap — `F5_peak_phase_{strategy}.{pdf,png}`

- **类型**: Heatmap，X = batch size，Y = optimizer
- **每策略一张**: eager / inductor(b=1.0) / inductor(b=0.0) / ac+inductor（共 4 策略 × 2 格式 = 8 文件）
- **数据来源**: `ex_peak_phase.csv`
- **含义**: 以离散颜色展示每个 batch×optimizer 组合的峰值阶段（FW/BW/OPT）
- **颜色**: FW = 蓝色, BW = 橙色, OPT = 红色
- **读图方法**: 红色区域（OPT 瓶颈）集中在 Adam + 小 batch。Adam(fused) 的 OPT 红色通常消失。AC 策略整体偏 FW
- **论文用途**: §5.2 核心图，展示"何时 OPT 成为瓶颈"的完整矩阵

### F6: Phase Stack — `F6_phase_stack_{strategy}.{pdf,png}`

- **类型**: Stacked Bar Chart，按 batch×optimizer 分组
- **每策略一张**: eager / inductor(b=1.0)（共 2 策略 × 2 格式 = 4 文件）
- **数据来源**: `ex_peak_phase.csv`
- **含义**: 将 fw_peak / bw_peak / opt_peak 堆叠展示，直观看三阶段的绝对大小和相对比例
- **读图方法**: 三层堆叠中最高的色块 = 该阶段占据的峰值份额。batch 增大 → FW 层增厚；SGD → OPT 层很薄
- **论文用途**: §5.2 辅助图，展示峰值的阶段组成变化趋势

### F7: Model Heatmap — `F7_model_heatmap_{level}.{pdf,png}`

- **类型**: Heatmap，X = 策略，Y = 模型
- **每仿真层一张**: l2 / l25 / l3（共 3 层 × 2 格式 = 6 文件）
- **数据来源**: `ex_model_generalization.csv`
- **含义**: 以颜色深浅展示每个模型×策略组合在指定仿真层级下的 MRE（%）
- **颜色**: 浅 = 低 MRE = 高精度，深 = 高 MRE
- **单元格标注**: MRE 百分比数值
- **读图方法**: 比较同一行（同模型）不同策略的精度差异；比较同一列（同策略）不同模型的精度差异
- **论文用途**: §5.3 核心图，证明仿真精度的架构通用性

---

## 四、论文引用建议

| 论文章节 | 推荐图表 | 推荐 CSV |
|----------|---------|---------|
| §3 方法论 — 内存组成分析 | F1 | — |
| §5.1 仿真精度验证 | F2, F3, F4 | ex_sim_accuracy.csv |
| §5.2 峰值阶段分析 | F5, F6 | ex_peak_phase.csv |
| §5.3 多模型通用性 | F7 (l2, l25, l3) | ex_model_generalization.csv |
| §5.4 讨论 — 时间-内存权衡 | F2 (Pareto 子图) | ex_sim_accuracy.csv |

### 论文正文中引用格式建议

- 图表: "如图 X 所示（对应 F3_peak_comparison）"
- 数据: "详细数据见附录表 X（对应 ex_sim_accuracy.csv）"
- 实验配置: 每个实验的模型配置写在脚本文件的 docstring 中，可直接复制到论文的实验设置节
