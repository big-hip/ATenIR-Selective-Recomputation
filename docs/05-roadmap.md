# 05 — 后续执行计划（Phase A/B/C）

> **文档定位**: 基于 `03-experiments.md` 九章结论 + 外部审查验证，规划后续策略扩展与论文实验。
> 原始来源: `plan.md` v4.2（2025-04-15），Phase 0 已完成部分移至 `04-dev-log.md`。

---

## 核心发现摘要（驱动后续设计）

1. **fwbw_peak = max(fw, bw)** 是重计算策略的正确指标（非 fw_peak 单独）
2. **activation_ratio = activation_bytes / grad_bytes** 决定 AC 是否有效
   - batch=1: -1.4%（无效）→ batch=4: +52% → batch=8: +71.5%
3. **Adam(fused=True)** 使 opt_peak 降 50%，AC true_peak 节省从 2.8%→**52%**
4. **Inductor 编译 19-22s**，steady-state 快 15%，budget=0 慢 35%
5. **FlashAttention / 碎片化假说均不成立**（见 `03-experiments.md` §八）

---

## 关键设计原则

**所有 `fw_peak`/`bw_peak`/`opt_peak` 统一为绝对峰值**
（含 param + grad + optim + activation + temp），
与 runtime `measure_phased` 的 `torch.cuda.max_memory_allocated()` 语义一致。
三层（L1 / L2 / Runtime）同名同义，可直接对比。

---

## ✅ Phase 0: 四峰值全链路贯通（已完成）

Phase 0 已在 Phase 10.0 中实现，详细记录见 `04-dev-log.md`。

**完成内容**:
- L1 `config_estimator.py`: 支持 `fused_optimizer`，返回 fw/bw/opt/fwbw/true_peak + peak_phase + timeline 7点
- L2 `graph_estimator.py`: 绝对峰值 + fw/bw_graph_peak 别名（向后兼容）+ opt_peak
- PhaseResult: 新增 `grad_bytes`, `fwbw_peak`, `peak_phase`
- validator.py: 支持 PhaseResult 分阶段 MRE
- Web UI: Simulation 三列统一对比表、phase_grouped_bar、stacked_breakdown
- 75 tests 全部通过（v6.1：web 测试随 web 目录删除，新增 output 测试）

---

## Phase A: Strategy Page 策略扩展

### A1. 扩展策略列表（6 → 12 策略）

基于 `03-experiments.md` §七 修正后的策略分组设计：

**两个独立变量**:
- **Partition**: default（不重计算）vs min_cut（graph-level 重计算）
- **Backend**: make_boxed_func（无融合）vs Inductor（有融合）
- 注意: Inductor 内部强制 min_cut，无法做 Inductor + default

#### Group 1: Eager（无编译）

| # | 策略 | 说明 |
|---|------|------|
| ① | **eager baseline** | 无编译、无重计算 |
| ② | classic_ac | 整块 checkpoint |
| ③ | sac_save_matmuls | 按 op 选择性保存 |
| ④ | sac_recompute_all | 全部偏向重算 |

对比: ①→②③④ 量化用户级 region-based 重计算效果

#### Group 2: Compiled（逐步加变量）

| # | 策略 | 说明 |
|---|------|------|
| ⑤ | **aot_eager + default** | 图捕获，无融合，无重计算（Baseline） |
| ⑥ | aot_eager + min_cut | 图捕获，无融合，有重计算 |
| ⑦ | inductor (budget=1.0) | 有融合 + min-cut 保守（Baseline B） |
| ⑧ | inductor (budget=0.5) | 有融合 + min-cut 中等 |
| ⑨ | inductor (budget=0.0) | 有融合 + min-cut 激进 |

对比链路:
- ⑤→⑥: min-cut 单独的内存收益（无融合干扰）
- ⑥→⑦: Inductor 融合的额外收益
- ⑦→⑨: budget 参数的边际效果

#### Group 3: AC/SAC + Compiled（两层重计算叠加）

| # | 策略 | 说明 |
|---|------|------|
| ⑩ | ac + aot_eager(default) | 仅用户级 AC，无融合、无 min-cut |
| ⑪ | ac + inductor | AC + min-cut + 融合 |
| ⑫ | sac_mm + inductor | SAC + min-cut + 融合 |

对比链路:
- ⑤→⑩: 用户级 AC 在图捕获下的单独效果
- ⑩→⑪: AC 基础上加 min-cut + 融合的额外收益
- ⑪ vs ⑫: AC vs SAC 策略差异

### A2. Fused optimizer 选项

- 增加 Checkbox: "Fused Optimizer (Adam)"
- 影响 optimizer 创建 + simulation `fused_optimizer` 参数
- 实验表明 fused=True 是解锁 AC 实际效果的关键（§八.4）

### A3. 测量指标

每个策略输出以下指标（对齐 `03-experiments.md` §七测量指标）：
- **fw_peak**: 前向峰值
- **bw_peak**: 反向峰值（含梯度+重算开销），标注 grad_bytes 占比
- **opt_peak**: 优化器步骤峰值
- **fwbw_peak = max(fw, bw)**: 前反向峰值
- **true_peak = max(fw, bw, opt)**: 训练总峰值
- **peak_phase**: 标注 true_peak 出现在哪个阶段
- **step_ms**: 单步训练耗时
- **compile_s**: 编译耗时

### A4. 图表增强

- `savings_waterfall()`: 节省率瀑布图
- `activation_ratio` 标注
- `phase_timeline_chart()`: L1/L2/Runtime 三线对比

预计工时: ~1.5h

---

## Phase B: 大 batch 验证实验（论文关键数据）

### B1. 正式实验脚本

文件: `toolkit_examples/exp_batch_scaling.py`

| 参数 | 值 |
|------|-----|
| 模型 | llama-12L-1024H |
| seq | 256 |
| batch | [1, 2, 4, 8, 16] |
| 策略 | eager, eager_ac, inductor(b=1.0), inductor(b=0.0), ac+inductor |
| 优化器 | SGD + Adam(fused=True) + Adam(fused=False) |
| 输出 | CSV + 图表 (activation_ratio vs AC 节省率) |

### B2. 论文核心结论验证

1. 验证 **activation_ratio → AC 收益** 的线性关系
2. 验证 **fused Adam 解锁 AC 效果**（2.8%→52%）
3. 验证 **Inductor 在大 batch 下的相对表现**
4. 验证 **compiled 策略是否在大 batch 下反超 eager**

### B3. 预期输出

- 论文图表: activation_ratio vs savings 散点图 + 拟合线
- 论文表格: 3 优化器 × 5 batch × 5 策略 的 fw/bw/opt/true_peak 矩阵
- 关键结论: batch scaling 临界点、fused Adam 影响量化

预计工时: ~1h（需 GPU）

---

## Phase C: Static/Runtime 校准闭环

### C1. validator.py 扩展

文件: `toolkit/profiler/validator.py`

- 支持 PhaseResult（分阶段对账）
- 对 fw_peak/bw_peak/opt_peak 分别计算 static vs runtime MRE
- 输出最终校准报告

### C2. 全量校准实验

| 维度 | 值 |
|------|-----|
| 模型 | GPT-2 / LLaMA / Mistral |
| batch | 2, 4, 8 |
| 策略 | eager, AC, inductor |
| 输出 | 校准报告（分阶段 MRE 矩阵） |

### C3. 论文精度结论

- L1 各阶段 MRE
- L2 各阶段 MRE（预期 fw/bw <10%, opt <5%）
- 误差来源定量分解

预计工时: ~1h

---

## 执行状态（2025-04-19 · 全部完成）

```
✅ Phase 0 (四峰值贯通 + Web)
✅ Phase A (12策略扩展)     → ex3_simulation_accuracy.py
✅ Phase B (大 batch 验证)  → ex4_batch_scaling.py + ex8_fused_optimizer.py
✅ Phase C (校准闭环)       → ex5 + ex6 + ex7
```

### Phase A 实际交付

- 12 策略在 `ex3_simulation_accuracy.py` 中完整实现（3 组 × 4 策略）
- `ex5_simulation_vs_runtime.py` 扩展了 L2+L3 仿真支持
- 图表: overview bar chart / per-group savings / Pareto scatter

### Phase B 实际交付

- `ex4_batch_scaling.py`: batch [1,2,4,8] × eager/ac，验证 activation_ratio → savings 关系
- `ex8_fused_optimizer.py`: SGD/Adam/Adam(fused) × 4 策略 × 2 batch，验证 fused Adam 解锁 AC

### Phase C 实际交付

- `ex5`: 12 策略 L2+L3 vs Runtime，aot_eager MRE 1-2.4%，L3 MRE 5-7%
- `ex6`: 5 策略 × 3 batch Inductor 双层，L2 avg 21.2%，L3 avg 10.7%
- `ex7`: GPT-2/LLaMA/Mistral × 6 策略矩阵，跨模型一致性验证
