# 实验输出说明文档

本文档详细说明 `toolkit_examples/` 中 ex1–ex8 所有实验的图表和 CSV 输出，包含每个输出的含义、数据来源、读图方法和论文引用建议。

所有输出保存在 `toolkit_examples/outputs/` 目录下。

---

## 总览

| 实验 | 主题 | 图表输出 | CSV 输出 |
|------|------|----------|----------|
| ex1 | L1/L2 多模型对比 | 2 张 PNG | — |
| ex2 | L2 仿真精度验证 | 2 张 PNG | — |
| ex3 | 12 策略全对比 | 3 张 PNG | 1 个 CSV |
| ex4 | Batch 规模与多策略效果 | 2 张 PNG | 1 个 CSV |
| ex5 | L2/L3 仿真 vs 运行时 | 2 张 PNG | 1 个 CSV |
| ex6 | Inductor 双层仿真 | 2 张 PNG | 2 个 CSV |
| ex7 | 多模型×策略矩阵 | 2 张 PNG | 1 个 CSV |
| ex8 | Fused Optimizer 对比 | 2 张 PNG | 1 个 CSV |

---

## ex1: L1/L2 多模型对比

### ex1_breakdown.png — 内存组成堆叠柱状图

- **类型**: Stacked Bar Chart
- **含义**: 对 GPT-2、LLaMA、Mistral 三个模型，将训练峰值内存分解为四个组成部分:
  - **Param** (蓝色): 模型参数占用
  - **Grad** (橙色): 梯度占用（= Param，因为每个参数有一个同形状梯度）
  - **Optimizer** (绿色): 优化器状态占用（SGD = 0，Adam = 2× Param）
  - **Activation** (红色): 前向激活占用（= true_peak − base − grad）
- **数据来源**: L2 仿真（ATen IR 图遍历），通过 `capture_graphs()` + `estimate_training_peak()` 计算
- **读图方法**: 柱子总高度 = 训练真实峰值（L2 估算）。红色部分越大，说明激活占比越高，重计算策略的收益空间越大
- **论文用途**: 展示不同架构的内存组成差异，说明 activation 是主要的优化目标

### ex1_timeline.png — 训练步内存时间线

- **类型**: Multi-line Chart，7 个采样点
- **含义**: 模拟一个完整训练步（Forward → Backward → Optimizer Step）中 GPU 内存的变化轨迹
- **7 个采样点**: Base → FW Peak → After FW → BW Peak → After BW → OPT Peak → After OPT
- **峰值标注**: 菱形标记（◆）标出 FW Peak / BW Peak / OPT Peak 的绝对值
- **数据来源**: L2 仿真 + 运行时 `measure_phased()` 采集
- **读图方法**: 每条线代表一个模型。折线高点即为该阶段的峰值内存。线条整体越高，模型越占内存
- **论文用途**: 直观展示 peak 出现在哪个阶段（FW / BW / OPT），辅助说明"峰值阶段决定优化策略"

---

## ex2: L2 仿真精度验证

### ex2_l2_vs_runtime_timeline.png — L2 仿真 vs 运行时时间线

- **类型**: Multi-line Chart
- **含义**: 对同一模型的同一训练步，分别绘制 L2 仿真预测的内存轨迹和 CUDA 运行时实测的内存轨迹
- **数据来源**: L2 仿真 (`estimate_training_peak`) 和运行时 (`measure_phased`)
- **读图方法**: 两条线越接近，说明 L2 仿真越准确。观察 FW Peak / BW Peak 的偏差即为误差来源
- **论文用途**: 核心验证图，证明 L2 仿真的有效性和误差范围

### ex2_phase_grouped.png — 分阶段 L2 vs Runtime 柱状图

- **类型**: Grouped Bar Chart
- **含义**: 将 FW Peak / BW Peak / OPT Peak 三个阶段的 L2 估算和运行时实测并排展示
- **三组柱子**: 蓝色 = FW Peak，橙色 = BW Peak，红色 = OPT Peak
- **每策略两根**: `{name}:L2` 和 `{name}:RT` 分别代表仿真和运行时
- **读图方法**: 同色柱子的 L2 和 RT 高度差 = 该阶段的仿真误差。关注哪个阶段偏差最大
- **论文用途**: 分阶段展示 L2 误差分布，说明 FW/BW/OPT 各阶段的仿真精度

---

## ex3: 12 策略全对比

### ex3_overview.png — 全策略峰值内存总览

- **类型**: Vertical Bar Chart，12 根柱子
- **含义**: 12 种重计算策略的 `fwbw_peak`（FW 和 BW 阶段中的最大峰值）对比
- **颜色分组**:
  - **蓝色 (G1)**: Eager 模式（S01 eager, S02 classic_ac, S03 sac_save_matmuls, S04 sac_recompute_all）
  - **橙色 (G2)**: Compiled 模式（S05-S09: aot_eager, inductor 不同 budget）
  - **绿色 (G3)**: AC/SAC + Compiled（S10-S12: ac+aot_eager, ac+inductor, sac+inductor）
- **参考线**: 虚线标出 S01 (Eager baseline) 和 S05 (Compiled baseline) 的峰值
- **柱顶标注**: 相对于组内 baseline 的节省百分比
- **读图方法**: 柱子越矮 = 峰值内存越小 = 策略越有效。负百分比表示比 baseline 更差
- **论文用途**: 核心对比图，一图展示全部 12 种策略的内存效率排名

### ex3_savings_per_group.png — 分组内存节省百分比

- **类型**: 3 个水平柱状子图（Horizontal Bar）
- **含义**: 每个 Group 内部各策略相对于 baseline 的 `fwbw_peak` 节省百分比
  - 子图 1: G1 (Eager) — baseline = S01 eager
  - 子图 2: G2 (Compiled) — baseline = S05 aot_eager+default
  - 子图 3: G3 (AC/SAC+Compiled) — baseline = S05 aot_eager+default
- **颜色**: 正值（节省）为组色，负值（增加）为红色
- **柱端标注**: 节省百分比数值
- **读图方法**: 柱子越长越好。红色柱表示该策略比 baseline 更差。关注每组中最优策略
- **论文用途**: 分组讨论策略效果时的配图，比 overview 更适合逐组分析

### ex3_pareto.png — 内存-时间 Pareto 散点图

- **类型**: Scatter Plot
- **含义**: 以 step_ms（单步训练时间）为 X 轴、fwbw_peak（峰值内存 MB）为 Y 轴，展示 12 种策略的时间-内存权衡关系
- **颜色**: 同 overview（蓝=G1, 橙=G2, 绿=G3）
- **每个点标注**: 策略名称
- **读图方法**: 左下角 = 最优（时间短 + 内存小）。位于 Pareto 前沿（左下边界）的策略为最佳权衡。右上角的策略时间和内存都差
- **论文用途**: 展示"没有绝对最优策略"的结论，inductor 时间短但内存高，AC 内存低但时间长

### ex3_all_strategies.csv — 全策略数据表

| 字段 | 含义 |
|------|------|
| `name` | 策略编号+名称（如 "S07 inductor(b=1.0)"） |
| `strategy` | 同 name |
| `group` | 分组（G1/G2/G3） |
| `fw_peak` | 前向阶段峰值内存 (bytes) |
| `bw_peak` | 反向阶段峰值内存 (bytes) |
| `opt_peak` | 优化器步峰值内存 (bytes) |
| `fwbw_peak` | max(fw_peak, bw_peak) |
| `true_peak` | max(fw_peak, bw_peak, opt_peak) |
| `peak_phase` | 峰值出现在哪个阶段 (FW/BW/OPT) |
| `step_ms` | 单步训练时间 (ms) |
| `base` | 训练前基础内存占用 (bytes) |
| `after_fw` / `after_bw` / `after_opt` | 各阶段结束后的内存占用 (bytes) |

---

## ex4: Batch 规模与多策略效果

### ex4_phase_grouped.png — 不同 Batch 的分阶段峰值

- **类型**: Grouped Bar Chart
- **含义**: 对 batch = {1, 2, 4, 8}，展示 4 种策略（eager, classic_ac, inductor(b=1.0), ac+inductor）的 FW Peak / BW Peak / OPT Peak
- **读图方法**:
  - 小 batch（1-2）: BW Peak 接近 FW Peak，各策略节省不明显
  - 大 batch（4-8）: FW Peak >> BW Peak，AC 和 Inductor 均大幅降低峰值
- **核心指标**: `activation_ratio = fw_peak / param_bytes`
  - ratio >> 1: 激活主导，所有策略均有效
  - ratio 接近 1: 效果微弱
  - ratio < 1: 梯度主导，策略无效
- **论文用途**: 揭示不同策略效果的 batch 依赖性，是"何时该用哪种策略"的核心论据

### ex4_batch_scaling_line.png — 真实峰值随 Batch 变化折线图

- **类型**: Line Chart，4 条折线
- **含义**: 以 batch size 为 X 轴、true_peak (MB) 为 Y 轴，对比 4 种策略的峰值内存随 batch 的增长趋势
- **4 条折线**: eager（基线）、classic_ac、inductor(b=1.0)、ac+inductor
- **每点标注**: 峰值内存数值 (MB)
- **读图方法**: 折线越平缓 = 策略对 batch 增长的抵抗力越强。ac+inductor 通常在大 batch 时与 classic_ac 接近，但编译开销更低
- **论文用途**: 直观展示各策略的扩展性差异，支撑"大 batch 场景下 AC+Inductor 组合最优"的结论

### ex4_batch_scaling.csv — Batch 规模数据表

| 字段 | 含义 |
|------|------|
| `batch` | 批次大小 |
| `strategy` | eager / classic_ac / inductor(b=1.0) / ac+inductor |
| `fw_peak` / `bw_peak` / `fwbw_peak` / `true_peak` | 各阶段峰值 (bytes) |
| `peak_phase` | 峰值阶段 (FW/BW) |
| `param_bytes` | 模型参数大小 (bytes) |
| `step_ms` | 单步时间 (ms) |

---

## ex5: L2/L3 仿真 vs 运行时

### ex5_peak_comparison.png — 运行时 vs L2 vs L3 峰值对比

- **类型**: Grouped Bar Chart，每策略 3 根柱子
- **含义**: 对 12 种策略，并排展示运行时实测峰值、L2 仿真峰值、L3 仿真峰值
- **颜色**: 蓝色 = Runtime，橙色 = L2 Sim，绿色 = L3 Sim
- **柱顶标注**: 峰值 MB 数值
- **读图方法**: 蓝-橙-绿三根柱子越接近，说明仿真越准确。G1 策略（eager）无仿真值（柱高为 0）。inductor 策略中 L3（绿）比 L2（橙）更接近蓝色（运行时）
- **论文用途**: 一图总览全部策略的仿真精度，直观展示 L3 相比 L2 的改进

### ex5_mre_bar.png — L2 vs L3 MRE 对比

- **类型**: Grouped Bar Chart，每策略 2 根柱子
- **含义**: 对有仿真数据的策略，展示 L2 MRE 和 L3 MRE 的对比
- **颜色**: 橙色 = L2 MRE，绿色 = L3 MRE
- **柱顶标注**: MRE 百分比
- **读图方法**: 柱子越矮 = 误差越小。aot_eager 策略 L2 MRE 极低（1-4%），inductor 策略 L2 MRE 高（28-47%）但 L3 MRE 显著降低（5-21%）
- **论文用途**: 量化展示 L3 仿真相对 L2 的精度提升幅度

### ex5_sim_vs_runtime.csv — 全策略仿真精度表

| 字段 | 含义 |
|------|------|
| `strategy` | 策略名 |
| `group` | 分组 |
| `rt_fw_peak` ... `rt_true_peak` | 运行时实测的各阶段峰值 (bytes) |
| `rt_peak_phase` | 运行时峰值阶段 |
| `rt_step_ms` | 运行时单步时间 |
| `rt_base` | 运行时基础内存 |
| `l2_fw_peak` ... `l2_true_peak` | L2 仿真估算的各阶段峰值（仅 G2/G3 策略有值） |
| `l2_peak_phase` | L2 峰值阶段 |
| `mre` | L2 相对误差 = \|l2_true_peak − rt_true_peak\| / rt_true_peak |
| `direction` | over (高估) 或 under (低估) |
| `l3_true_peak` | L3 仿真峰值（仅 inductor 策略有值） |
| `l3_mre` | L3 相对误差 |
| `l3_direction` | L3 方向 |

**关键数据特征**:
- G1 策略（eager）无仿真值（不经过 torch.compile）
- aot_eager 策略: L2 MRE 1-2%（非常准确）
- inductor 策略: L2 MRE 28-47%（因不建模 fusion），L3 MRE 5-14%（融合感知后显著改善）

---

## ex6: Inductor 双层仿真

### ex6_l2_vs_l3_mre.png — L2 vs L3 MRE 对比 (batch=8)

- **类型**: Grouped Bar Chart，每策略 2 根柱子
- **含义**: 固定 batch=8，对 5 种 inductor 策略展示 L2 MRE 和 L3 MRE 的对比
- **颜色**: 橙色 = L2 MRE，绿色 = L3 MRE
- **柱顶标注**: MRE 百分比
- **读图方法**: 绿色柱（L3）几乎总是低于橙色柱（L2），证明 Scheduler 融合建模的价值。不同 budget 的 L3 改善幅度不同
- **论文用途**: 聚焦 inductor 策略，量化 L3 相对 L2 的精度提升

### ex6_batch_scaling.png — inductor(b=1.0) 峰值随 Batch 变化

- **类型**: Line Chart，3 条折线
- **含义**: 以 batch size 为 X 轴，对 inductor(b=1.0) 策略展示 Runtime、L2 Sim、L3 Sim 三种峰值估算随 batch 的变化趋势
- **3 条折线**: 蓝色实线 = Runtime，橙色虚线 = L2 Sim，绿色点划线 = L3 Sim
- **每点标注**: 峰值 MB 数值
- **读图方法**: L3 折线紧贴 Runtime，而 L2 折线偏离较大。随 batch 增大，L2 的高估趋势更明显
- **论文用途**: 展示仿真精度的 batch 稳定性，说明 L3 在不同 batch 下均保持较高精度

### ex6_inductor_dual_sim.csv — L2/L3 双层精度表

| 字段 | 含义 |
|------|------|
| `strategy` | inductor 策略（5 种 × 3 个 batch） |
| `batch` | 批次大小 (2/4/8) |
| `l2_true_peak` / `l3_true_peak` / `rt_true_peak` | L2/L3/运行时峰值 (bytes) |
| `l2_mre` / `l3_mre` | L2/L3 相对误差 |
| `l2_direction` / `l3_direction` | 估算方向 |
| `l2_peak_phase` / `l3_peak_phase` / `rt_peak_phase` | 各层峰值阶段 |
| `step_ms` | 运行时单步时间 |

**读数方法**: 对比 `l2_mre` 和 `l3_mre` 列 — L3 在绝大多数情况下误差更小，证明 Scheduler 融合建模的价值。

### ex6_fusion_stats.csv — Inductor 图融合统计

| 字段 | 含义 |
|------|------|
| `strategy` / `batch` | 策略和批次 |
| `fw_total_nodes` / `bw_total_nodes` | 前向/反向图的总节点数 |
| `fw_alloc_nodes` / `bw_alloc_nodes` | 前向/反向图中实际分配内存的节点数 |
| `fw_total_alloc_mb` / `bw_total_alloc_mb` | 前向/反向图的总分配量 (MB) |
| `sched_fw_peak_mb` / `sched_bw_peak_mb` | Scheduler 估算的前向/反向峰值 (MB) |

**读数方法**:
- `sched_fw_peak_mb` 远小于 `fw_total_alloc_mb` 说明 Inductor 的 buffer fusion 和内存复用非常有效
- 不同 budget 的 `bw_total_nodes` 差异反映重计算带来的额外节点

---

## ex7: 多模型×策略矩阵

### ex7_heatmap_mre.png — MRE 热力图 (模型 x 策略)

- **类型**: Heatmap，行 = 模型（gpt2/llama/mistral），列 = 可仿真策略
- **含义**: 以颜色深浅展示每个模型-策略组合的 L2 MRE（%）
- **颜色**: YlOrRd 色标，浅黄 = 低 MRE，深红 = 高 MRE
- **单元格标注**: MRE 百分比数值（NaN = 无仿真数据）
- **读图方法**: 找到颜色最浅的区域 = 仿真最准确的组合。inductor 策略行通常较深（L2 不建模 fusion），aot_eager 行通常最浅
- **论文用途**: 展示仿真精度的模型通用性，证明 L2 仿真在不同架构上的表现差异

### ex7_peak_by_model.png — 各模型运行时峰值对比

- **类型**: Grouped Bar Chart，按模型分组
- **含义**: 对每个模型（gpt2/llama/mistral），展示所有 6 种策略的运行时真实峰值 (MB)
- **颜色**: 每种策略一个颜色
- **柱顶标注**: 峰值 MB 数值
- **读图方法**: 同一模型内比较不同策略的峰值差异；跨模型比较同一策略的效果是否一致
- **论文用途**: 验证策略效果的架构通用性，是"方法是否普适"的关键论据

### ex7_multi_model_matrix.csv — 通用性验证矩阵

| 字段 | 含义 |
|------|------|
| `model` | 模型名 (gpt2/llama/mistral) |
| `strategy` | 策略名 (eager, classic_ac, aot_eager+default, aot_eager+min_cut, inductor(b=1.0), ac+inductor) |
| `rt_true_peak` | 运行时峰值 (bytes) |
| `l2_true_peak` | L2 仿真峰值（eager/ac 无值） |
| `l3_true_peak` | L3 仿真峰值（仅 inductor 策略） |
| `l2_mre` / `l3_mre` | 各层 MRE |
| `l2_direction` / `l3_direction` | 估算方向 |
| `rt_peak_phase` | 运行时峰值阶段 |
| `step_ms` | 单步时间 |

**关键数据特征**:
- LLaMA/Mistral 的 L2 (aot_eager) MRE < 4% — 高精度
- GPT-2 的 L2 MRE ~17% (低估) — 因 GPT-2 的 `Conv1D` 实现产生不同的中间 tensor 模式
- L3 在所有三个模型上均优于 L2（inductor 策略下）

---

## ex8: Fused Optimizer 对比

### ex8_optimizer_savings.png — 优化器 x 策略峰值对比 (batch=8)

- **类型**: Grouped Bar Chart，按优化器分组
- **含义**: 固定 batch=8，对每种优化器（SGD/Adam/Adam(fused)）展示 4 种策略的真实峰值 (MB)
- **颜色**: 每种策略一个颜色
- **柱顶标注**: 峰值 MB 数值
- **读图方法**: 同一优化器内比较策略效果；跨优化器比较 base 内存差异。Adam(fused) 与 Adam 的峰值差异体现 fused 优化的价值
- **论文用途**: 展示"优化器选择影响峰值阶段"的核心结论，证明 fused optimizer 消除 OPT 瓶颈

### ex8_peak_phase_heatmap.png — 峰值阶段热力图

- **类型**: Heatmap，行 = 优化器+batch+策略组合，列标签隐含
- **含义**: 以离散颜色展示每个实验组合的峰值阶段（FW/BW/OPT）
- **颜色**: 蓝色 = FW，橙色 = BW，红色 = OPT
- **读图方法**: 红色区域（OPT 峰值）集中在 Adam 大 batch 的 eager/ac 策略。Adam(fused) 行中红色完全消失 = fused 成功消除 OPT 瓶颈
- **论文用途**: 直观展示 fused optimizer 的核心价值——将峰值阶段从 OPT 转移回 FW/BW，使重计算策略能充分发挥作用

### ex8_fused_optimizer.csv — 优化器类型对比表

| 字段 | 含义 |
|------|------|
| `optimizer` | SGD / Adam / Adam(fused) |
| `strategy` | eager / classic_ac / inductor(b=1.0) / ac+inductor |
| `batch` | 批次大小 (4/8) |
| `fw_peak` / `bw_peak` / `opt_peak` | 各阶段峰值 (bytes) |
| `fwbw_peak` / `true_peak` | 组合峰值 |
| `peak_phase` | 峰值阶段 (FW/BW/OPT) |
| `step_ms` | 单步时间 |
| `base` | 基础内存（含优化器状态） |

**关键数据特征**:
- **SGD**: base ≈ 215 MB（无 optimizer state），peak_phase 在 FW 或 BW
- **Adam**: base ≈ 610 MB（含 momentum + variance），peak_phase 出现 OPT（3/8 场景）
- **Adam(fused)**: base ≈ 610 MB，但 opt_peak 降低约 13%，且 **peak_phase 中 OPT 完全消失** (0/8)
- **核心结论**: fused Adam 的价值不仅在于降低 opt_peak 绝对值，更在于消除 OPT 阶段成为瓶颈的可能性，使得重计算策略只需关注 FW/BW 阶段

---

## 论文引用建议

| 论文章节 | 推荐使用的输出 |
|----------|---------------|
| 系统架构 / 三层仿真介绍 | ex1_breakdown.png, ex1_timeline.png |
| L2 仿真精度验证 | ex2_l2_vs_runtime_timeline.png, ex2_phase_grouped.png, ex5_peak_comparison.png, ex5_mre_bar.png |
| 策略全对比 | ex3_overview.png, ex3_pareto.png, ex3 CSV |
| 分组策略分析 | ex3_savings_per_group.png |
| 多策略 batch 扩展性 | ex4_phase_grouped.png, ex4_batch_scaling_line.png, ex4 CSV |
| L3 vs L2 精度提升 | ex6_l2_vs_l3_mre.png, ex6_batch_scaling.png, ex6 双 CSV |
| 多架构通用性 | ex7_heatmap_mre.png, ex7_peak_by_model.png, ex7 CSV |
| 优化器对峰值阶段的影响 | ex8_optimizer_savings.png, ex8_peak_phase_heatmap.png, ex8 CSV |
| 内存-时间权衡讨论 | ex3_pareto.png |
