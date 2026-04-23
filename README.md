# ATenIR Selective Recomputation

> **基于 PyTorch 2.6 ATen IR 的训练显存静态仿真与重计算策略分析框架**

通过捕获 `torch.compile` 生成的 ATen IR（FX 图），遍历 alloc/free 事件流，在**不执行实际训练**的情况下预测不同重计算策略下的峰值 GPU 显存，并与运行时测量交叉验证。

**实测精度** (LLaMA ~870M, B=8, S=512, Adam fused, 12 种策略):

| 仿真层级 | 方法 | 代表性 MRE |
|---------|------|----------|
| **L2** | ATen IR 图遍历 | 1.9% ~ 18.5% (均值 8.0%) |
| **L2.5** | 融合感知图遍历 | 1.9% ~ 9.0% (均值 4.9%) |
| **L3** | Inductor Scheduler | 0.9% ~ 19.7% (均值 9.1%) |

---

## 环境

| 依赖 | 版本 |
|------|------|
| PyTorch | 2.6.0 + CUDA 12.4 |
| transformers | 4.35.2 |
| Python | ≥ 3.10 |

```bash
# conda (推荐)
conda activate torch2.6-gpu
pip install -r requirements.txt

# Docker
docker compose up --rm        # 一键运行 Demo
docker compose run --rm atenir bash  # 交互式终端
```

---

## 快速开始

```bash
# 快速 Demo: 重计算 IR 对比 + 仿真验证（GPT-2, <20s）
python toolkit_examples/ex_quick_demo.py

# Demo: L1/L2 静态估算（3 模型，~30s）
python toolkit_examples/ex1_multi_model_capture.py

# 实验 1: 12 策略仿真精度验证（LLaMA ~870M, B=8, S=512, 需 GPU ≥40GB）
python toolkit_examples/ex_sim_accuracy.py

# 实验 2: 峰值阶段分析（batch × optimizer 矩阵）
python toolkit_examples/ex_peak_phase.py

# 实验 3: 多模型通用性验证（GPT-2 / LLaMA / Mistral × 6 策略）
python toolkit_examples/ex_model_generalization.py

# 生成论文图表（F0-F9, PNG）
python toolkit_examples/generate_paper_figures.py

# 运行全部测试
python -m pytest tests/ -x -q
```

---

## 项目结构

```text
ATenIR-Selective-Recomputation/
├── toolkit/                        # 核心框架
│   ├── capture/                    #   图捕获
│   │   ├── aot_capture.py          #     capture_graphs() — ATen IR FW/BW 图
│   │   ├── inductor_capture.py     #     capture_inductor_graphs() — post-grad + L3
│   │   └── analysis.py             #     graph_stats, count_fw_output_bytes
│   ├── simulation/                 #   静态仿真引擎
│   │   ├── config_estimator.py     #     L1: 公式法 (estimate_from_config)
│   │   ├── graph_estimator.py      #     L2/L2.5/L3: 图遍历 (estimate_graph_peak)
│   │   ├── fusion_groups.py        #     L2.5: 融合组识别
│   │   └── fusion_ops.py           #     L2.5: 算子分类 (extern/fusable)
│   ├── profiler/                   #   运行时测量
│   │   ├── step_profiler.py        #     measure_phased() + IQR mean
│   │   └── validator.py            #     validate() + analyze_error_sources
│   ├── strategy/                   #   重计算策略注入
│   │   ├── classic_ac.py           #     wrap_with_checkpoint
│   │   ├── sac.py                  #     wrap_with_sac
│   │   ├── partition.py            #     get_partition_fn (default/min_cut)
│   │   └── memory_budget.py        #     set_memory_budget (Inductor)
│   ├── models/                     #   模型注册表 (GPT-2/LLaMA/Mistral)
│   │   ├── registry.py             #     ModelRegistry — 离线创建
│   │   └── adapters.py             #     统一适配器
│   ├── output/                     #   输出
│   │   ├── console.py              #     表格打印
│   │   ├── charts.py               #     交互图表
│   │   ├── export.py               #     CSV/JSON 导出
│   │   ├── pub_charts.py           #     论文图表 (F0-F9)
│   │   └── pub_style.py            #     论文样式
│   └── utils/                      #   工具函数
│       ├── view_ops.py             #     is_view_node (storage aliasing)
│       ├── tensor_utils.py         #     val_bytes, count_unique_params
│       ├── formatting.py           #     format_bytes, normalize_row
│       └── env.py                  #     setup_experiment_env
│
├── toolkit_examples/               # 论文实验脚本
│   ├── ex_quick_demo.py            #   快速 Demo: IR 对比 + 仿真验证
│   ├── ex1_multi_model_capture.py  #   Demo: L1/L2 三模型捕获
│   ├── ex_sim_accuracy.py          #   实验 1: 12 策略仿真精度
│   ├── ex_peak_phase.py            #   实验 2: 峰值阶段分析
│   ├── ex_model_generalization.py  #   实验 3: 多模型通用性
│   ├── generate_paper_figures.py   #   论文图表 (F0-F9)
│   └── outputs/                    #   论文 CSV + PNG 输出
│
├── tests/                          # 测试 (10 文件, 95 tests)
├── docs/                           # 项目文档 (18 篇)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── Makefile
```

---

## 核心架构：四层仿真

| 层级 | 方法 | 输入 | 速度 | 建模 fusion |
|------|------|------|------|------------|
| **L1** | 公式法 | 模型配置 | < 1s | ❌ |
| **L2** | ATen IR 图遍历 | FX GraphModule | 5-10s | ❌ |
| **L2.5** | 融合感知图遍历 | FX GraphModule | 5-10s | ✅ 近似 |
| **L3** | Inductor Scheduler | Scheduler 编译产物 | 30-60s | ✅ 精确 |

```
L2:    Model → torch.compile(aot_eager) → FW/BW FX graphs → estimate_training_peak
L2.5:  Model → torch.compile(inductor)  → post-grad FX    → estimate_inductor_training_peak
L3:    同 L2.5，额外 hook Inductor Scheduler 的 estimate_peak_memory
```

---

## 实验结果概要

### 实验 1: 12 策略仿真精度

**LLaMA ~870M** (16L/2048H, B=8, S=512, Adam fused)

| 策略 | Runtime | L2 MRE | L2.5 MRE | L3 MRE |
|------|---------|--------|----------|--------|
| S05 aot_eager+default | 29.2 GB | **3.7%** | — | — |
| S07 inductor(b=1.0) | 23.5 GB | 1.9% | **1.9%** | 13.4% |
| S08 inductor(b=0.5) | 17.9 GB | 4.6% | **4.6%** | 19.7% |
| S09 inductor(b=0.0) | 23.8 GB | 18.5% | **6.3%** | 1.7% |
| S11 ac+inductor | 16.1 GB | 8.4% | 9.0% | **9.6%** |
| S12 sac_mm+inductor | 21.1 GB | 3.0% | **3.0%** | 0.9% |

### 实验 2: 峰值阶段分析

Batch=[2,4,8,16] × Optimizer=[SGD, Adam, Adam(fused)]

- **SGD**: peak 始终在 FW（无优化器状态）
- **Adam(fused)**: opt_peak 降低 ~50%，peak 由 FW/BW activation 主导
- **AC 有效性**依赖 `fw_peak / grad_bytes` 比值和 fused optimizer

### 实验 3: 多模型通用性

GPT-2 124M / LLaMA ~870M / Mistral ~550M × 6 策略，验证仿真精度跨架构一致性。

---

## 论文图表

```bash
python toolkit_examples/generate_paper_figures.py
```

| 编号 | 内容 | 数据来源 |
|------|------|---------|
| F0 | 方法总览图（含 ShapeSum/L1/L2/L2.5/L3 与 runtime profile） | 代码结构示意 |
| F1 | 三模型内存组成堆叠柱状图 | L1 估算 |
| F2 | 12 策略总览 + Pareto 图 | ex_sim_accuracy.csv |
| F3 | RT vs L2 vs L2.5 vs L3 分组柱状图 | ex_sim_accuracy.csv |
| F4 | L2/L2.5/L3 MRE 对比 | ex_sim_accuracy.csv |
| F5 | 峰值阶段热力图 | ex_peak_phase.csv |
| F6 | FW/BW/OPT 三阶段堆叠柱状图 | ex_peak_phase.csv |
| F7 | 模型 × 策略 MRE 热力图 | ex_model_generalization.csv |
| F8 | 横向方法对比 (L1→L2→L2.5→L3) | ex_horizontal_comparison.csv |
| F9 | L2.5 消融实验 | ex_horizontal_comparison.csv |

---

## 测试

```bash
python -m pytest tests/ -x -q                    # 全部 95 tests
python -m pytest tests/ -x -q -m "not inductor"  # 跳过 Inductor/Triton 慢测试
```

---

## 文档

详见 [`docs/00-index.md`](docs/00-index.md)，共 18 篇文档，涵盖架构设计、仿真引擎原理、精度优化、实验分析、开发日志等。
