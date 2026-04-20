# ATenIR-Selective-Recomputation — 项目文档索引

> **一句话定义**: 基于 PyTorch 2.6 ATen IR 的静态内存仿真框架，不跑完训练即可估算不同重计算策略下的峰值显存。
>
> **环境**: PyTorch 2.6.0 + CUDA 12.4 · transformers 4.35.2 · conda env `torch2.6-gpu`

---

## 文档目录

| 编号 | 文件 | 内容 | 建议阅读顺序 |
|------|------|------|-------------|
| 00 | **本文件** | 项目总览 + 索引 | 1️⃣ |
| 01 | `01-architecture.md` | 四支柱架构、五种策略、九条设计原则、L1/L2 仿真细节 | 2️⃣ |
| 02 | `02-research.md` | PyTorch partitioner 源码分析、文献综述、工业调研 | 3️⃣ |
| 03 | `03-experiments.md` | Min-Cut 九章实验报告（含 batch scaling、fused Adam 验证） | 4️⃣ |
| 04 | `04-dev-log.md` | Phase 1-10 开发记录、B1-B16 / W1-W23 Bug 全表、验证数据 | 按需 |
| 05 | `05-roadmap.md` | Phase A/B/C 后续执行计划 | 5️⃣ |
| 06 | `06-activation-memory-budget.md` | `activation_memory_budget` 全链路源码深度解析 | 按需 |
| 07 | `07-ac-sac-deep-dive.md` | AC / SAC 原理、源码解析、与 min-cut 组合机制 | 按需 |
| 08 | `08-static-simulation-deep-dive.md` | 静态仿真引擎：L1/L2 原理、view 检测、四峰值、验证 | 按需 |
| 09 | `09-runtime-profiler-deep-dive.md` | 运行时 Profiler：CUDA 内存 API、分阶段测量、IQR mean、Validator | 按需 |
| 10 | `10-graph-capture-deep-dive.md` | 图捕获：Dynamo→AOTAutograd→联合图→partition 拆分机制 | 按需 |
| 11 | `11-l2-accuracy-improvement.md` | L2 仿真精度优化：view-base pin + grad 去重，MRE 21%→7% | 按需 |
| 12 | `12-inductor-memory-analysis.md` | Inductor 后端 L2 内存分析可行性调研：管线、实验、方案评估 | 按需 |
| 13 | `13-l2.5-fusion-aware-design.md` | L2.5 融合感知仿真设计：fusion group 识别与 internal 消除 | 按需 |
| 14 | `14-aten-recompute-key-ideas.md` | 已删除 aten_recompute 目录关键设计存档 | 按需 |
| 15 | `15-experiment-outputs.md` | ex1-ex8 实验输出图表与 CSV 说明 | 按需 |

---

## 项目现状（2025-04-19 · 代码冻结）

### 已完成

| Phase | 模块 | 关键结论 |
|-------|------|---------|
| 1 | `toolkit/utils/` | `_cdata` view 检测修正 4.8%，SymInt 0 失败 |
| 2 | `toolkit/models/` | GPT-2 + LLaMA + Mistral 离线创建 |
| 3 | `toolkit/capture/` | AOT + Inductor 双层捕获，CE loss 修正 MRE 50%→6.9% |
| 4 | `toolkit/simulation/` | L1 公式法 + L2 图遍历 + L3 Inductor Scheduler |
| 5 | `toolkit/strategy/` | AC/SAC/partition/budget 完整 |
| 6 | `toolkit/profiler/` | StepResult / PhaseResult / validator / IQR mean |
| 7 | `toolkit/output/` | console / charts / export |
| 8 | `toolkit_examples/` | **ex1-ex8 全部完成并验证** |
| 9 | ~~toolkit/web/~~ | Gradio Web UI（已删除，聚焦 CLI + 脚本输出） |
| 10.0 | 全量 Bug 修复 | B1-B16 + W 系列核心项 |
| A | 12 策略扩展 | 3 组 12 策略 + ex3 全对比 + 3 图表 |
| B | 大 batch 验证 | ex4 batch scaling + ex8 fused optimizer |
| C | 校准闭环 | ex5 L2+L3 vs Runtime + ex6 Inductor 双层验证 + ex7 多模型矩阵 |

### 论文实验数据 (ex1-ex8)

| 脚本 | 内容 | 核心数据 |
|------|------|---------|
| ex1 | L1/L2 对比 | L1 vs L2 peak 差异 36-49% |
| ex2 | 仿真精度 | L2 MRE 15-18%（含 dark memory 分析） |
| ex3 | 12 策略全对比 | AC -30.8%, inductor(b=0.5) -34.4%, ac+inductor -39.5% |
| ex4 | batch scaling | act_ratio vs AC savings 线性关系 |
| ex5 | L2+L3 vs Runtime | aot_eager MRE 1-2.4%, L3 MRE 5-7% |
| ex6 | Inductor 双层 | L2 avg 21.2%, L3 avg 10.7% |
| ex7 | 3 模型矩阵 | GPT-2/LLaMA/Mistral × 6 策略 |
| ex8 | Fused optimizer | SGD/Adam/Adam(fused) × 4 策略 × 2 batch |

### 测试: 75 tests passed

---

## 版本演进

| 版本 | 日期 | 关键突破 |
|------|------|---------|
| v1 | 2025-04 初 | 初始 verify 脚本 + 调研 |
| v2 | 2025-04 中 | 引入 ModelRegistry + 多模型测试 |
| v3 | 2025-04 中 | 6 文件计划体系，Phase 1-9 框架搭建完成 |
| v3.5 | 2025-04-14 | ★ B10 发现：L2 MRE 50%→6.9%；深度审计 16 个 Bug |
| v3.6 | 2025-04-14 | Web UI 23 项审计；Phase 10.0 全量修复 |
| v4.2 | 2025-04-15 | 基于 research-mincut-analysis 九章结论重设计 Phase A/B/C |
| v5 | 2025-04-15 | 文档整合：20 个散乱文件 → 13 个结构化文档 |
| v6 | 2025-04-19 | Inductor 双层仿真 + ex1-ex8 全量实验 + 代码冻结 |
| **v6.1** | **2025-04-19** | **代码清理 + 审查修复：L1/L2 bw_peak 一致性、output 测试、stale ref 清除** |
