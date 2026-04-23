# 18 — 毕设论文写作指南

> **目的**: 把当前代码、数据和图表口径整理成可直接写论文的路线图。  
> **当前数据版本**: 2026-04-23 重跑后的 `toolkit_examples/outputs/*.csv` 与 F0-F9 PNG。

---

## 一、论文主线

一句话主线：

> 本项目基于 PyTorch 2.6 的 ATen IR 与 Inductor 编译产物，构建从配置公式、图级 live-range、融合感知仿真到 Scheduler 级峰值估算的多层静态显存预测框架，并用真实 GPU profiling 验证不同重计算策略下的峰值显存。

建议突出三点贡献：

1. **策略统一注入**：AC / SAC / min-cut / memory budget 都通过同一套 capture + profiler 流程比较。
2. **多层静态仿真**：L1、ShapeSum、L2、L2.5、L3 形成精度-成本阶梯。
3. **Runtime 对齐验证**：`measure_phased()` 提供 FW/BW/OPT 分阶段 ground truth，能解释不同策略的峰值阶段。

---

## 二、推荐章节结构

| 章节 | 建议内容 | 对应文档/图表 |
|------|---------|--------------|
| 第 1 章 绪论 | 显存瓶颈、重计算背景、静态估算意义、本文贡献 | `02-research.md` |
| 第 2 章 系统设计 | 四支柱架构：策略注入、图捕获、静态仿真、运行时验证 | `01-architecture.md`, F0 |
| 第 3 章 图捕获与策略建模 | AOTAutograd / Inductor 捕获，AC/SAC/min-cut/budget 如何改变图 | `07`, `10` |
| 第 4 章 静态显存仿真方法 | L1、ShapeSum、L2 live-range、view alias、L2.5、L3 | `08`, `13`, `16`, `17` |
| 第 5 章 实验与分析 | 12 策略、峰值阶段、多模型泛化、横向方法对比 | `15`, F1-F9 |
| 第 6 章 总结与展望 | 局限：L2.5 保守、L3 依赖 Inductor、ShapeSum 只是 baseline | `17` |

---

## 三、图表使用顺序

| 图 | 放置位置 | 论文里要说明的点 |
|----|---------|----------------|
| F0 | 方法章节开头 | 解释策略注入、图捕获、ShapeSum/L1/L2/L2.5/L3 静态估算和 runtime profile 的关系 |
| F1 | 系统/实验开头 | 参数、梯度、优化器、激活的组成差异，引出 activation 优化 |
| F2 | 实验总览 | 12 策略显存-时间权衡，说明 AC/budget/SAC 的 trade-off |
| F3 | 仿真精度 | Runtime vs L2/L2.5/L3 的峰值接近程度 |
| F4 | 仿真精度 | MRE 数字化比较，主实验 L2.5 平均 4.9% |
| F5 | 峰值阶段 | 不同 batch/optimizer 下峰值阶段从 FW/BW/OPT 切换 |
| F6 | 峰值组成 | FW/BW/OPT 三阶段绝对峰值对比 |
| F7 | 泛化实验 | GPT-2/LLaMA/Mistral 上 L2.5/L3 overall 平均 7.0% |
| F8 | 横向方法 | ShapeSum 明显 over，说明 live-range 的必要性 |
| F9 | 消融分析 | L2 → fusion-only → safe-reuse → L3 的变化 |

注意：F8/F9 推荐使用 GPT-2 medium 配置（8L/512H, B=8, S=256）。它比 quick 更接近论文实验规模，同时比 paper 三模型横向对比更省时间；若作为强定量结论，可再补跑 paper 配置。

---

## 四、当前可引用的关键数字

### 主实验：`ex_sim_accuracy.csv`

| 层级 | 样本数 | 平均 MRE | 范围 |
|------|-------|---------|------|
| L2 | 8 | 8.0% | 1.9%-18.5% |
| L2.5 | 5 | 4.9% | 1.9%-9.0% |
| L3 | 5 | 9.1% | 0.9%-19.7% |

重点例子：S09 `inductor(b=0.0)` 中 L2=18.5%，L2.5=6.3%，L3=1.7%，适合展示重计算强场景下 L3 的价值。

### 多模型泛化：`ex_model_generalization.csv`

| 层级 | 样本数 | overall 平均 MRE | 最大 MRE |
|------|-------|------------------|---------|
| L2 | 15 | 8.4% | 21.6% |
| L2.5 | 12 | 7.0% | 16.8% |
| L3 | 12 | 7.0% | 13.4% |

### 横向对比：`ex_horizontal_comparison.csv`

| 方法 | 平均 MRE |
|------|---------|
| L1 config formula | 38.8% |
| ShapeSum_graph | 315.1% |
| L2 live-range | 8.5% |
| L2.5 fusion/safe | 10.6% |
| L3 Scheduler | 5.4% |

ShapeSum 是 naive baseline，不等价于 DNNMem/LLMem/xMem。它的作用是证明“不做 live-range 只累加 shape 会严重高估”。

---

## 五、写作口径注意事项

- 不要再引用 `test_old/` 或 4 月 20 日旧 CSV 的 L2.5 MRE。
- L2.5 当前是保守口径：unknown/materializing ops 默认 barrier，目标是避免低估危险场景。
- L3 是独立路径，不再与 L2.5 做 `min(sched_bw, fusion_bw)`。
- 论文中把 L3 写成 **compiler-assisted static estimate**：它 hook PyTorch Inductor Scheduler，需要 GPU/Inductor/Triton 捕获环境，但不是 `torch.cuda.max_memory_allocated()` 的真实 runtime profile。
- `ShapeSum_graph` 只是横向 baseline，不要写成复现 DNNMem。
- 主指标对齐 `torch.cuda.max_memory_allocated()`；reserved memory 只能作为 allocator 碎片讨论。
- 测试很慢的原因是 Inductor/Triton 编译与 Scheduler hook；日常只跑 `-m "not inductor"`，定稿前跑全量。本轮实测：非 Inductor 83 tests 约 40 秒，全量 95 tests 约 5 分 36 秒。

---

## 六、代码冻结建议

现在适合进入论文写作，只做低风险变更：

1. 文档、图注、表格口径修订。
2. 重新生成 CSV/PNG。
3. 修复明显 typo 或测试中暴露的小 bug。

不建议再大改：

1. L2/L2.5/L3 核心公式。
2. 新增复杂 allocator/BFC 仿真。
3. 新增未验证的大型实验维度。

---

## 七、最终复现命令

日常快速验证：

```bash
conda run -n torch2.6-gpu python -m pytest tests/ -x -q -m "not inductor"
```

定稿前完整验证：

```bash
conda run -n torch2.6-gpu python -m pytest tests/ -x -q
conda run -n torch2.6-gpu python toolkit_examples/ex_sim_accuracy.py
conda run -n torch2.6-gpu python toolkit_examples/ex_model_generalization.py
conda run -n torch2.6-gpu python toolkit_examples/ex_peak_phase.py
conda run -n torch2.6-gpu python toolkit_examples/ex_horizontal_comparison.py --medium
conda run -n torch2.6-gpu python toolkit_examples/generate_paper_figures.py
```
