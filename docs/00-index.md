# ATenIR-Selective-Recomputation — 项目文档索引

> **一句话定义**: 基于 PyTorch 2.6 ATen IR 的静态内存仿真框架，不跑完训练即可估算不同重计算策略下的峰值显存。
>
> **环境**: PyTorch 2.6.0 + CUDA 12.4 · transformers 4.35.2 · conda env `torch2.6-gpu`

---

## 文档目录

| 编号 | 文件 | 内容 | 论文对应 |
|------|------|------|---------|
| 00 | **本文件** | 项目总览 + 索引 | — |
| 01 | `01-architecture.md` | 四支柱架构、五种策略、四层仿真、设计原则 | 第 2 章 §2.1 |
| 02 | `02-research.md` | PyTorch partitioner 源码分析、文献综述、工业调研 | 第 1 章 §1.2 |
| 03 | `03-experiments.md` | Min-Cut 策略分析报告（早期数据 + 结论） | 第 3 章 §3.3-3.4 |
| 04 | `04-dev-log.md` | Phase 1-10 开发记录、Bug 全表、验证数据 | 附录 |
| 05 | `05-roadmap.md` | Phase A/B/C 执行计划（已全部完成） | — |
| 06 | `06-activation-memory-budget.md` | `activation_memory_budget` 全链路源码深度解析 | 第 2 章 §2.6 |
| 07 | `07-ac-sac-deep-dive.md` | AC / SAC 原理、源码解析、与 min-cut 组合机制 | 第 2 章 §2.1 |
| 08 | `08-static-simulation-deep-dive.md` | 静态仿真引擎：L1/L2/L2.5 原理、view 检测、四峰值 | 第 2 章 §2.3-2.5 |
| 09 | `09-runtime-profiler-deep-dive.md` | 运行时 Profiler：CUDA 内存 API、分阶段测量、IQR mean | 第 2 章 §2.7 |
| 10 | `10-graph-capture-deep-dive.md` | 图捕获：AOTAutograd + Inductor 双层捕获机制 | 第 2 章 §2.2 |
| 11 | `11-l2-accuracy-improvement.md` | L2 精度优化：view-base pin + grad 去重，MRE 21%→7% | 第 2 章 §2.4 |
| 12 | `12-inductor-memory-analysis.md` | Inductor 后端内存分析：管线、L3 Scheduler hook | 第 2 章 §2.6 |
| 13 | `13-l2.5-fusion-aware-design.md` | L2.5 融合感知仿真：fusion group 识别与 internal 消除 | 第 2 章 §2.5 |
| 14 | `14-aten-recompute-key-ideas.md` | 已删除 aten_recompute 目录关键设计存档 | 附录 |
| 15 | `15-experiment-outputs.md` | 实验输出说明：3 个 CSV + 7 类论文图表 (F1-F7) | 第 3 章 |
| 16 | `16-param-double-counting-fix.md` | 参数双重计数修复：`peak_ph_alive` + `_forwarded_primal_bytes`，MRE 12%→8.8% | 第 2 章 §2.4 |
| — | `sim-optimization-plan.md` | 仿真精度优化方案：Overlap-Aware Peak + 编译器元数据重计算检测 | 第 2 章 §2.4 |

---

## 项目现状（2025-04-20 · 文档整理）

### 框架实现

| Phase | 模块 | 关键成果 |
|-------|------|---------|
| 1 | `toolkit/utils/` | `_cdata` storage aliasing view 检测（修正 4.8%）+ SymInt 处理 |
| 2 | `toolkit/models/` | GPT-2 + LLaMA + Mistral 离线创建（ModelRegistry） |
| 3 | `toolkit/capture/` | AOT 捕获 (`capture_graphs`) + Inductor 双层捕获 (`capture_inductor_graphs`) |
| 4 | `toolkit/simulation/` | L1 公式法 + L2 图遍历 + L2.5 融合感知 + L3 Scheduler |
| 5 | `toolkit/strategy/` | AC / SAC / partition / memory_budget 完整 |
| 6 | `toolkit/profiler/` | `measure_phased` (IQR mean) + `validate` + `analyze_error_sources` |
| 7 | `toolkit/output/` | console / charts / export + pub_charts (F1-F7) + pub_style |

### 论文实验脚本

| 脚本 | 论文章节 | 内容 | 输出 |
|------|---------|------|------|
| `ex1_multi_model_capture.py` | Demo | L1/L2 三模型捕获演示 | 控制台输出 |
| `ex_sim_accuracy.py` | §5.1 | 12 策略 L2/L2.5/L3 仿真精度 (LLaMA ~870M, B=8, S=512) | `ex_sim_accuracy.csv` |
| `ex_peak_phase.py` | §5.2 | batch×optimizer 峰值阶段分析 (LLaMA ~870M) | `ex_peak_phase.csv` |
| `ex_model_generalization.py` | §5.3 | 3 模型×6 策略通用性验证 (GPT-2/LLaMA/Mistral) | `ex_model_generalization.csv` |
| `generate_paper_figures.py` | — | 论文图表生成 (F1-F7, PDF+PNG) | `outputs/paper_figures/` |

### 论文图表 (F1-F7)

| 编号 | 内容 | 数据来源 |
|------|------|---------|
| F1 | 三模型内存组成堆叠柱状图 | L1 估算 |
| F2 | 12 策略总览 + Pareto 图 | ex_sim_accuracy.csv |
| F3 | RT vs L2 vs L2.5 vs L3 分组柱状图 | ex_sim_accuracy.csv |
| F4 | L2/L2.5/L3 MRE 对比 | ex_sim_accuracy.csv |
| F5 | 峰值阶段热力图 (batch × optimizer) | ex_peak_phase.csv |
| F6 | FW/BW/OPT 三阶段堆叠柱状图 | ex_peak_phase.csv |
| F7 | 模型 × 策略 MRE 热力图 | ex_model_generalization.csv |

### 测试: 10 个文件, 85 个 test 函数

---

## 版本演进

| 版本 | 日期 | 关键突破 |
|------|------|---------|
| v1 | 2025-04 初 | 初始 verify 脚本 + 调研 |
| v2 | 2025-04 中 | 引入 ModelRegistry + 多模型测试 |
| v3 | 2025-04 中 | 6 文件计划体系，Phase 1-9 框架搭建完成 |
| v3.5 | 2025-04-14 | B10 发现：CE loss 修正 → L2 MRE 50%→6.9%；深度审计 16 个 Bug |
| v3.6 | 2025-04-14 | Web UI 23 项审计；Phase 10.0 全量修复 |
| v4.2 | 2025-04-15 | 基于 research-mincut-analysis 九章结论重设计 Phase A/B/C |
| v5 | 2025-04-15 | 文档整合：20 个散乱文件 → 13 个结构化文档 |
| v6 | 2025-04-19 | Inductor 双层仿真 + L2.5 融合感知 + 全量实验 |
| v6.1 | 2025-04-19 | 代码清理 + 审查修复：bw_peak 一致性、output 测试、stale ref 清除 |
| v6.2 | 2025-04-20 | 脚本合并 (ex1-ex8 → 3 脚本) + 论文图表系统 (F1-F7) + TF32/警告管理 |
| v7 | 2025-04-20 | 文档全面整理：按代码最终状态重写 docs/ + plans/，论文可直接引用 |
| **v7.1** | **2025-04-21** | **项目整理：README/requirements/Docker/Makefile 对齐实际代码，清理诊断脚本** |
