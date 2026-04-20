# ATenIR Selective Recomputation

> **基于 PyTorch 2.6 ATen IR 的训练显存静态仿真与重计算策略分析框架**

本项目提供纯静态的 GPU 显存峰值估算工具，通过捕获 ATen IR 中间表示（FX 图）并遍历 alloc/free 事件流，在**不执行实际训练**的情况下预测不同重计算策略的峰值显存，并与运行时测量交叉验证。

**核心指标**: L2 (ATen 图遍历) MRE ≈ 6.9%（aot_eager），L3 (Inductor Scheduler) MRE ≈ 5-7%（ac+inductor）

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
# L1/L2 静态估算（无 GPU 训练，~30s）
python toolkit_examples/ex1_multi_model_capture.py

# L2 仿真精度验证（含 Runtime 对比，~50s）
python toolkit_examples/ex2_strategy_comparison.py

# 12 策略全对比（含 Inductor 编译，~3min）
python toolkit_examples/ex3_simulation_accuracy.py

# 运行全部测试
python -m pytest tests/ -x -q
```

---

## 项目结构

```text
ATenIR-Selective-Recomputation/
├── toolkit/                    # ★ 核心框架（论文方法论实现）
│   ├── capture/                #   图捕获: AOT + Inductor 双层
│   │   ├── aot_capture.py      #     capture_graphs() — ATen IR 捕获
│   │   ├── inductor_capture.py #     capture_inductor_graphs() — L3 Scheduler hook
│   │   └── analysis.py         #     graph_stats, count_fw_outputs
│   ├── simulation/             #   静态仿真引擎
│   │   ├── config_estimator.py #     L1: 公式法估算
│   │   └── graph_estimator.py  #     L2/L3: 图遍历仿真 + Inductor 扩展
│   ├── profiler/               #   运行时测量 + 验证
│   │   ├── step_profiler.py    #     measure_phased() — 分阶段 GPU 内存采集
│   │   └── validator.py        #     validate() — L2/Runtime MRE 计算
│   ├── strategy/               #   重计算策略
│   │   ├── classic_ac.py       #     Activation Checkpointing
│   │   ├── sac.py              #     Selective AC
│   │   ├── partition.py        #     min_cut / default partition
│   │   └── memory_budget.py    #     Inductor memory budget 控制
│   ├── models/                 #   模型注册表 (GPT-2 / LLaMA / Mistral)
│   ├── output/                 #   输出: 表格 / 图表 / CSV
│   └── utils/                  #   工具: view 检测 / tensor bytes / 格式化
│
├── toolkit_examples/           # ★ 论文实验脚本 (ex1-ex8)
│   ├── ex1_multi_model_capture.py      # L1 vs L2 对比（3 模型）
│   ├── ex2_strategy_comparison.py      # L2 仿真精度验证（MRE 分析）
│   ├── ex3_simulation_accuracy.py      # 12 策略全对比（3 组 × 图表）
│   ├── ex4_batch_scaling.py            # batch 规模 → AC 效果关系
│   ├── ex5_simulation_vs_runtime.py    # L2+L3 vs Runtime 全 12 策略
│   ├── ex6_inductor_dual_sim.py        # Inductor 双层仿真精度验证
│   ├── ex7_multi_model_matrix.py       # 3 模型 × 6 策略矩阵
│   ├── ex8_fused_optimizer.py          # Fused optimizer 对 AC 效果影响
│   └── outputs/                        # CSV + 图表输出
│
├── tests/                      # 测试 (75 tests)
│   ├── test_capture.py         #   AOT 图捕获
│   ├── test_inductor_capture.py#   Inductor 双层捕获
│   ├── test_simulation.py      #   L2/L3 仿真
│   ├── test_profiler.py        #   运行时 Profiler
│   ├── test_strategy.py        #   AC/SAC/Partition
│   ├── test_models.py          #   模型注册表
│   ├── test_val_bytes.py       #   tensor bytes
│   ├── test_view_ops.py        #   view 检测
│   ├── test_fusion_aware.py    #   融合感知仿真
│   └── test_output.py          #   输出模块 (charts/console/export)
│
├── docs/                       # 项目文档 (00-15)
└── requirements.txt
```

---

## 核心架构：三层仿真

| 层级 | 方法 | 输入 | 精度 (MRE) | 速度 |
|------|------|------|-----------|------|
| **L1** | 公式法 | 模型配置 | 15-25% | < 1s |
| **L2** | ATen IR 图遍历 | FX GraphModule | **~1-7%** (aot_eager) | 5-10s |
| **L3** | Inductor Scheduler | Scheduler 编译产物 | **~5-7%** | 30-60s (含编译) |

### L2 核心流程

```
Model → torch.compile(aot_eager) → capture FW/BW FX graphs
  → estimate_training_peak(fw_gm, bw_gm, model)
  → {fw_peak, bw_peak, opt_peak, true_peak, peak_phase}
```

### L3 扩展流程

```
Model → torch.compile(inductor) → capture post-grad FX + Scheduler hook
  → estimate_inductor_training_peak(capture_result, model)
  → L2 fields + {l3_fw_peak, l3_bw_peak, l3_true_peak, l3_peak_phase}
```

---

## 实验结果概要

### 12 策略对比 (ex3, LLaMA 6L/512H, B=8, SGD)

| 组 | 策略 | fwbw_peak | vs baseline |
|----|------|-----------|-------------|
| G1 | eager baseline | 895.8 MB | — |
| G1 | classic_ac | 620.2 MB | **-30.8%** |
| G2 | aot_eager+default | 1.0 GB | — |
| G2 | inductor(b=0.5) | 671.9 MB | **-34.4%** |
| G3 | ac+inductor | 620.1 MB | **-39.5%** |

### 仿真精度 (ex5/ex6)

| 后端 | 策略 | L2 MRE | L3 MRE |
|------|------|--------|--------|
| aot_eager | default | 2.4% | — |
| aot_eager | min_cut | 1.2% | — |
| inductor | ac+inductor | 38.4% | **6.8%** |
| inductor | sac+inductor | 30.4% | **5.4%** |

### Batch Scaling (ex4)

| batch | act_ratio | AC savings |
|-------|-----------|-----------|
| 1 | 1.45 | 0.0% |
| 4 | 2.82 | 24.9% |
| 8 | 4.55 | **30.8%** |

**关键结论**: AC 效果取决于 `activation_ratio = fw_peak / grad_bytes`，ratio >> 1 时才有效。

---

## 测试

```bash
# 全部测试
python -m pytest tests/ -x -q
```

---

## 文档

详见 [`docs/00-index.md`](docs/00-index.md)，按主题组织为 13 篇文档，涵盖架构、实验、开发日志、源码深度解析等。

---

## 历史模块说明

早期的 `aten_recompute/` (graph-level 重计算引擎) 和 `examples/` (Transformer 实验脚本) 已删除，关键设计要点存档于 [`docs/14-aten-recompute-key-ideas.md`](docs/14-aten-recompute-key-ideas.md)。
