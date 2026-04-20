# ATenIR Selective Recomputation

> **基于 PyTorch 2.6 ATen IR 的训练显存静态仿真与重计算策略分析框架**

本项目提供纯静态的 GPU 显存峰值估算工具，通过捕获 ATen IR 中间表示（FX 图）并遍历 alloc/free 事件流，在**不执行实际训练**的情况下预测不同重计算策略的峰值显存，并与运行时测量交叉验证。

**核心指标**:
- L2 (ATen 图遍历) MRE ≈ 1-7%（aot_eager 后端）
- L2.5 (融合感知图遍历) MRE ≈ 5-12%（inductor 后端）
- L3 (Inductor Scheduler) MRE ≈ 5-7%（inductor 后端）

---

## 环境

- **PyTorch** 2.6.0 + CUDA 12.4
- **transformers** 4.35.2
- **conda env**: `torch2.6-gpu`

```bash
conda activate torch2.6-gpu
pip install -r requirements.txt
```

---

## 快速开始

```bash
# Demo: L1/L2 静态估算（3 模型，~30s）
python toolkit_examples/ex1_multi_model_capture.py

# 实验 1: 12 策略 L2/L2.5/L3 仿真精度验证（LLaMA ~870M, B=8, S=512）
python toolkit_examples/ex_sim_accuracy.py

# 实验 2: 峰值阶段分析（batch × optimizer 矩阵）
python toolkit_examples/ex_peak_phase.py

# 实验 3: 多模型通用性验证（GPT-2 / LLaMA / Mistral × 6 策略）
python toolkit_examples/ex_model_generalization.py

# 生成论文图表（F1-F7, PDF+PNG）
python toolkit_examples/generate_paper_figures.py

# 运行全部测试（83 tests）
python -m pytest tests/ -x -q
```

---

## 项目结构

```text
ATenIR-Selective-Recomputation/
├── toolkit/                    # ★ 核心框架（论文方法论实现）
│   ├── capture/                #   图捕获: AOT + Inductor 双层
│   │   ├── aot_capture.py      #     capture_graphs() — ATen IR FW/BW 图捕获
│   │   ├── inductor_capture.py #     capture_inductor_graphs() — post-grad 图 + L3 Scheduler hook
│   │   └── analysis.py         #     graph_stats, count_fw_outputs, count_fw_output_bytes
│   ├── simulation/             #   静态仿真引擎
│   │   ├── config_estimator.py #     L1: 公式法估算 (estimate_from_config)
│   │   ├── graph_estimator.py  #     L2: 图遍历仿真 (estimate_graph_peak)
│   │   │                       #     L2.5: 融合感知仿真 (fusion_aware=True)
│   │   │                       #     L3: Inductor 三层封装 (estimate_inductor_training_peak)
│   │   ├── fusion_groups.py    #     L2.5: 融合组识别 (identify_fusion_groups)
│   │   └── fusion_ops.py       #     L2.5: 算子分类 (EXTERN_OPS, is_fusable_op)
│   ├── profiler/               #   运行时测量 + 验证
│   │   ├── step_profiler.py    #     measure_phased() — 分阶段 GPU 内存采集 + IQR mean
│   │   └── validator.py        #     validate() — MRE 计算 + analyze_error_sources
│   ├── strategy/               #   重计算策略
│   │   ├── classic_ac.py       #     Activation Checkpointing (wrap_with_checkpoint)
│   │   ├── sac.py              #     Selective AC (wrap_with_sac, SAC_POLICIES)
│   │   ├── partition.py        #     min_cut / default partition (get_partition_fn)
│   │   └── memory_budget.py    #     Inductor memory budget (set_memory_budget)
│   ├── models/                 #   模型注册表 (GPT-2 / LLaMA / Mistral)
│   │   ├── registry.py         #     ModelRegistry — 离线创建 HuggingFace 模型
│   │   └── adapters.py         #     统一适配器 (get_hidden, get_num_layers, ...)
│   ├── output/                 #   输出: 表格 / 图表 / CSV / 论文图表
│   │   ├── console.py          #     print_comparison_table, print_step_result
│   │   ├── charts.py           #     7 种交互图表函数
│   │   ├── export.py           #     to_csv, to_json
│   │   ├── pub_charts.py       #     7 类论文图表 (F1-F7, plot_f1..plot_f7)
│   │   └── pub_style.py        #     paper_style 上下文管理器 + savefig_pub
│   └── utils/                  #   工具函数
│       ├── view_ops.py         #     is_view_node — storage aliasing 检测
│       ├── tensor_utils.py     #     val_bytes, count_unique_params
│       ├── formatting.py       #     format_bytes, normalize_row
│       └── env.py              #     setup_experiment_env — TF32 + 警告过滤
│
├── toolkit_examples/           # ★ 论文实验脚本
│   ├── ex1_multi_model_capture.py    # Demo: L1/L2 三模型捕获
│   ├── ex_sim_accuracy.py            # 实验 1: 12 策略仿真精度 (→ ex_sim_accuracy.csv)
│   ├── ex_peak_phase.py              # 实验 2: batch×optimizer 峰值阶段 (→ ex_peak_phase.csv)
│   ├── ex_model_generalization.py    # 实验 3: 3 模型通用性验证 (→ ex_model_generalization.csv)
│   ├── generate_paper_figures.py     # 论文图表生成 (F1-F7, PDF+PNG)
│   └── outputs/                      # CSV + 论文图表输出
│       └── paper_figures/            #   F1-F7 (PDF+PNG, 共 29 个文件)
│
├── tests/                      # 测试 (10 个文件, 83 个 test 函数)
│   ├── test_capture.py         #   AOT 图捕获 (4)
│   ├── test_inductor_capture.py#   Inductor 双层捕获 (7)
│   ├── test_simulation.py      #   L2/L2.5/L3 仿真 (14)
│   ├── test_fusion_aware.py    #   融合感知仿真 (10)
│   ├── test_profiler.py        #   运行时 Profiler (6)
│   ├── test_strategy.py        #   AC/SAC/Partition (5)
│   ├── test_models.py          #   模型注册表 (5)
│   ├── test_val_bytes.py       #   tensor bytes (4)
│   ├── test_view_ops.py        #   view 检测 (3)
│   └── test_output.py          #   输出模块 (25)
│
├── docs/                       # 项目文档 (00-15, 共 16 篇)
└── requirements.txt
```

---

## 核心架构：四层仿真

| 层级 | 方法 | 输入 | 精度 (MRE) | 速度 | 建模 fusion |
|------|------|------|-----------|------|------------|
| **L1** | 公式法 | 模型配置 | 15-25% | < 1s | ❌ |
| **L2** | ATen IR 图遍历 | FX GraphModule | **1-7%** (aot_eager) | 5-10s | ❌ |
| **L2.5** | 融合感知图遍历 | FX GraphModule | **5-12%** (inductor) | 5-10s | ✅ 近似 |
| **L3** | Inductor Scheduler | Scheduler 编译产物 | **5-7%** (inductor) | 30-60s | ✅ 精确 |

### L2 核心流程

```
Model → torch.compile(aot_eager) → capture FW/BW FX graphs
  → estimate_training_peak(fw_gm, bw_gm, model)
  → {fw_peak, bw_peak, opt_peak, true_peak, peak_phase}
```

### L2.5 + L3 扩展流程

```
Model → torch.compile(inductor) → capture post-grad FX + Scheduler hook
  → estimate_inductor_training_peak(capture_result, model)
  → L2 fields + L2.5 fields (fusion_aware) + L3 fields (Scheduler)
```

---

## 实验结果概要

三个正式实验均使用放大版模型配置，在 A6000 48GB GPU 上运行。

### 实验 1: 12 策略仿真精度 (ex_sim_accuracy)

**LLaMA ~870M** (16L/2048H, B=8, S=512, Adam fused)

| 组 | 代表策略 | Runtime true_peak | L2 MRE | L2.5 MRE | L3 MRE |
|----|---------|------------------|--------|----------|--------|
| G1 Eager | eager baseline | — | — | — | — |
| G1 Eager | classic_ac | — | — | — | — |
| G2 Compiled | aot_eager+default | ✓ | ~1-3% | — | — |
| G2 Compiled | inductor(b=1.0) | ✓ | ~28% | ~8-12% | ~5-7% |
| G3 组合 | ac+inductor | ✓ | ~38% | ~8-12% | ~5-7% |

### 实验 2: 峰值阶段分析 (ex_peak_phase)

**LLaMA ~870M**, Batch=[2,4,8,16] × Optimizer=[SGD, Adam, Adam(fused)]

关键发现:
- **SGD**: peak 始终在 FW（无优化器状态）
- **Adam(non-fused)**: 小 batch 时 peak 在 OPT，大 batch 时 peak 迁移到 FW
- **Adam(fused)**: opt_peak 降低（无 foreach 临时变量），peak 更早由 FW 主导

### 实验 3: 多模型通用性 (ex_model_generalization)

**GPT-2 124M / LLaMA ~870M / Mistral ~550M** × 6 策略

验证 L2/L2.5/L3 仿真在三种不同 Transformer 架构上的精度一致性。

---

## 论文图表 (F1-F7)

```bash
python toolkit_examples/generate_paper_figures.py
```

| 编号 | 内容 | 数据来源 |
|------|------|---------|
| **F1** | 三模型内存组成堆叠柱状图 (param/grad/optim/act) | L1 估算 |
| **F2** | 12 策略总览 + Pareto 图 (显存 vs 时间) | ex_sim_accuracy.csv |
| **F3** | RT vs L2 vs L2.5 vs L3 分组柱状图 | ex_sim_accuracy.csv |
| **F4** | L2/L2.5/L3 MRE 对比 | ex_sim_accuracy.csv |
| **F5** | 峰值阶段热力图 (batch × optimizer) | ex_peak_phase.csv |
| **F6** | FW/BW/OPT 三阶段堆叠柱状图 | ex_peak_phase.csv |
| **F7** | 模型 × 策略 MRE 热力图 | ex_model_generalization.csv |

---

## 测试

```bash
python -m pytest tests/ -x -q        # 全部 83 tests
python -m pytest tests/ -x -q -k "not inductor"  # 跳过 inductor 测试（无 GPU 时）
```

---

## 文档

详见 [`docs/00-index.md`](docs/00-index.md)，按主题组织为 16 篇文档，涵盖架构、实验、仿真引擎深度解析、精度优化、开发日志等。

---

## 历史模块说明

早期的 `aten_recompute/` (graph-level 重计算引擎) 和 `examples/` (Transformer 实验脚本) 已删除，关键设计要点存档于 [`docs/14-aten-recompute-key-ideas.md`](docs/14-aten-recompute-key-ideas.md)。
