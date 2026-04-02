# ATenIR Selective Recomputation

基于 PyTorch `torch.compile` / AOT Autograd 的 **ATen IR 级选择性激活重计算**项目。

项目当前有三条需要明确区分的路径：

1. **PyTorch 官方用户级重计算路径**
   - `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`
   - `create_selective_checkpoint_contexts(...)`
   - `CheckpointPolicy`
2. **PyTorch internal / diagnostic 路径**
   - 直接手工调用 internal partitioner 的实验性 backend
   - 只用于研究和诊断，**不作为官方主基线**
3. **ATenIR 自定义 graph-level 路径**
   - 本项目自己的 joint-graph selective recomputation
   - 当前主策略为 `strategy 7`

---

## 当前阶段目标

项目当前不是直接比较“谁更好”，而是按下面顺序推进：

1. **先正确复现 PyTorch built-in recomputation**
   - checkpoint / SAC 的官方用法要接对、跑通、验证对
2. **再验证 ATenIR 是否相对 no-recompute 有真实收益**
   - 先回答“有效没效”，再谈策略优劣
3. **再缩小 static/runtime gap**
   - 让静态估算能更接近真实运行
4. **最后再做完整策略比较**

这也是当前代码和实验组织的基本原则。

---

## 当前推荐入口

### Transformer 主轨道

- Benchmark：[`examples/transformer/benchmark.py`](examples/transformer/benchmark.py)
- Capture：[`examples/transformer/capture.py`](examples/transformer/capture.py)

### Transformer 统一入口

- [`examples/transformer/main.py`](examples/transformer/main.py)

可通过：

```bash
python examples/transformer/main.py benchmark
python examples/transformer/main.py capture
```

### 官方 built-in baseline 轨道

- [`examples/official_recompute_baseline/main.py`](examples/official_recompute_baseline/main.py)

这条轨道的作用是：
- 单独复现 PyTorch 官方推荐的 checkpoint / SAC 用法
- 避免把 internal partitioner 误当成官方 graph-level API

---

## 当前主方法矩阵

### A. region / 用户级方法

用于回答：**PyTorch built-in recomputation 在用户 API 层面是否有效**。

- `eager_baseline`
- `eager_checkpoint`
- `eager_sac`

语义：
- 用户显式定义 checkpoint region
- 反向时 replay region

### B. graph / 主轨道方法

用于回答：**ATenIR graph-level recomputation 相对 graph no-recompute 是否有效**。

- `compiled_no_recompute`
- `ATenIR_strat7_b1.0`

语义：
- joint graph 层面决定 save / recompute
- 当前主策略是 `strategy 7`

### C. diagnostic 方法

这些路径可以保留做研究或调试，但**不能直接当主结论基线**。

例如：
- 手工 backend + `min_cut_rematerialization_partition`
- 其它 strategy 家族的诊断性对比

---

## 当前已验证的主结论

截至当前代码状态，主轨道已经能稳定得到以下事实：

### region 路径
- `eager_checkpoint` 相对 `eager_baseline` 有显著显存收益
- `eager_sac` 相对 `eager_baseline` 也有显存收益
- region 组 correctness 已基本对齐

### graph 路径
- `ATenIR_strat7_b1.0` 相对 `compiled_no_recompute` 已复现出真实显存收益
- 当前 static 与 runtime 已经同向
- 但 static 仍然比 runtime 更乐观，存在可解释的 gap

### 当前对 gap 的理解
- static 保存激活减少量与 runtime 收益方向一致
- backward phase 的收益没有完全按 static 估计兑现
- runtime 里绝对峰值下降已经明显，但 backward 增量峰值仍受调度 / 临时量 / allocator 影响

---

## 项目结构（按当前真实用法）

```text
ATenIR-Selective-Recomputation/
├── aten_recompute/
│   ├── core/
│   │   ├── compiler.py
│   │   ├── partition.py
│   │   ├── min_cut.py
│   │   └── tag.py
│   ├── analysis/
│   │   ├── profiler.py
│   │   ├── static.py
│   │   ├── flops.py
│   │   └── comparison.py
│   └── utils/
│       ├── checkpoint.py
│       ├── graph_utils.py
│       └── save_ir.py
├── examples/
│   ├── transformer/
│   │   ├── benchmark.py
│   │   ├── capture.py
│   │   ├── main.py
│   │   ├── input.yaml
│   │   ├── model_config.yaml
│   │   ├── benchmark_flow/
│   │   ├── capture_flow/
│   │   ├── meta_pipeline.py
│   │   ├── model.py
│   │   └── model_loader.py
│   ├── official_recompute_baseline/
│   │   ├── main.py
│   │   └── model.py
│   └── simple_mlp/
│       └── ...
└── IR_artifacts/
```

### 目录职责

#### `aten_recompute/core/`
核心编译与 graph-level 重计算逻辑。

当前最关键的文件：
- `compiler.py`：`CompilerBackend`，`torch.compile` 的自定义后端入口
- `partition.py`：graph 切分主逻辑，决定 save / recompute
- `min_cut.py`：strategy 7 的 min-cut 求解与约束修复
- `tag.py`：层级标签传播所需的标记机制

#### `aten_recompute/analysis/`
分析与报告层。

当前最关键的文件：
- `static.py`：静态显存估算
- `profiler.py`：runtime 显存 / 时间采样、主报告、gap 结果持久化
- `flops.py`：FLOPs 与理论执行时间估算

#### `aten_recompute/utils/`
基础设施与共享工具。

当前最关键的文件：
- `checkpoint.py`：PyTorch checkpoint 包装
- `graph_utils.py`：FX 图结构 / boundary / graph compare 工具
- `save_ir.py`：IR / DOT / 报告输出目录与文件保存

#### `examples/transformer/`
主实验模型与主轨道。

承担：
- Transformer benchmark
- capture / meta-runtime 对照
- ATenIR 主轨道验证
- static/runtime gap 分析

内部当前分层为：
- `benchmark.py` / `capture.py`：推荐入口
- `main.py`：统一入口
- `benchmark_flow/`：benchmark 主流程分段实现
- `capture_flow/`：capture 主流程分段实现
- `meta_pipeline.py`：meta/runtime 图语义对照辅助
- `model.py` / `model_loader.py`：模型与构造逻辑

#### `examples/official_recompute_baseline/`
官方 built-in 重计算基线示例。

承担：
- checkpoint / SAC 官方用户 API 的清晰复现
- compile 下官方用户 API 行为观察
- 和 Transformer 主轨道分开，避免语义混淆

#### `examples/simple_mlp/`
更偏研究 / 诊断 / 早期原型参考。

不应直接当作 Transformer 主结论替代物。

---

## 调用流程（当前主轨道）

下面这张流程说明的是：**一次 Transformer benchmark 到底是如何穿过 examples、flow、core、analysis、utils 的。**

### 1. Benchmark 主流程

```text
examples/transformer/benchmark.py
    ↓
benchmark_flow/runner.py
    ↓
train_phase.py
    ├─ build_transformer_from_map(...)
    ├─ inject_transformer_layer_tags(...)
    ├─ CompilerBackend(strategy_config=...)
    └─ torch.compile(...)
    ↓
static_phase.py
    ├─ capture_static_with_retry(...)
    ├─ StaticEstimator.estimate_from_graphs(...)
    └─ FLOPsEstimator.estimate_from_graphs(...)
    ↓
runtime_phase.py
    ├─ instantiate methods
    ├─ correctness checks
    ├─ MemoryProfiler.profile_step(...)
    ├─ graph_gap_breakdown / bw_gap_breakdown
    └─ save_report(...)
```

### 2. Capture 主流程

```text
examples/transformer/capture.py
    ↓
capture_flow/runner.py
    ↓
capture_phase.py
    ├─ CompilerBackend(mode="static" or "runtime")
    ├─ torch.compile(...)
    └─ trigger forward/backward once
    ↓
analysis_phase.py
    ├─ compare_capture_semantics(...)
    └─ print meta/runtime graph comparison
```

### 3. `torch.compile` 进入 ATenIR 路径后的核心调用链

在主 graph 路径下，核心链路可以概括为：

```text
torch.compile(model, backend=CompilerBackend(...))
    ↓
CompilerBackend.__call__
    ↓
AOT Autograd joint graph capture
    ↓
partition.py
    ├─ baseline saved_values analysis
    ├─ layer rank propagation
    ├─ strategy dispatch
    ├─ strategy 7 → min_cut.py
    ├─ primal补充 / boundary调整
    └─ FW/BW graph split
    ↓
Inductor compile FW/BW
```

### 4. 主输出是如何产生的

```text
static_phase.py
    ├─ static_estimation.json
    └─ flops_estimation.json

runtime_phase.py
    ├─ memory_report.json
    ├─ runtime_memory.png
    ├─ phase_breakdown.png
    └─ graph_gap_breakdown / bw_gap_breakdown 写入 memory_report.json
```

### 5. 当前最推荐的阅读顺序

如果你第一次读这个项目，建议按下面顺序看：

1. `README.md`
2. `examples/transformer/benchmark.py`
3. `examples/transformer/benchmark_flow/runner.py`
4. `train_phase.py` / `static_phase.py` / `runtime_phase.py`
5. `aten_recompute/core/compiler.py`
6. `aten_recompute/core/partition.py`
7. `aten_recompute/core/min_cut.py`
8. `aten_recompute/analysis/static.py` / `profiler.py`

---

## 输入配置
### Transformer 主配置

主配置文件：
- [`examples/transformer/input.yaml`](examples/transformer/input.yaml)

模型结构配置：
- [`examples/transformer/model_config.yaml`](examples/transformer/model_config.yaml)

当前 `input.yaml` 主要包含：
- `run.task`
- `env.*`
- `capture.*`
- `benchmark.*`

其中：
- `env.recompute` 决定当前主策略
- `benchmark.run_correctness_checks` 决定是否进行 correctness 对照

### 统一入口的任务切换

`run.task` 可选：
- `benchmark`
- `capture`

也可以由 CLI 指定覆盖。

### 关于 `input_capture.yaml`

当前仓库里仍保留：
- [`examples/transformer/input_capture.yaml`](examples/transformer/input_capture.yaml)

它更接近历史 / capture-only 配置，后续会继续整理其是否保留以及如何和主配置收口。

---

## 运行方式

### 1. 运行 Transformer benchmark

```bash
python examples/transformer/benchmark.py
```

或：

```bash
python examples/transformer/main.py benchmark
```

### 2. 运行 Transformer capture

```bash
python examples/transformer/capture.py
```

或：

```bash
python examples/transformer/main.py capture
```

### 3. 运行官方 built-in baseline 示例

```bash
python examples/official_recompute_baseline/main.py
```

---

## 输出文件

当前最重要的输出目录是：

- [`IR_artifacts/Transformer/memory/`](IR_artifacts/Transformer/memory/)

### 主报告

- [`IR_artifacts/Transformer/memory/memory_report.json`](IR_artifacts/Transformer/memory/memory_report.json)

它当前已经是**主运行报告**，包含：
- runtime 结果
- phase 级峰值信息
- `graph_gap_breakdown`
- `bw_gap_breakdown`

### 静态估算

- [`IR_artifacts/Transformer/memory/static_estimation.json`](IR_artifacts/Transformer/memory/static_estimation.json)

包含：
- graph-based static peak estimate
- saved activation bytes
- FW/BW phase peak estimate

### FLOPs 估算

- [`IR_artifacts/Transformer/memory/flops_estimation.json`](IR_artifacts/Transformer/memory/flops_estimation.json)

### 图表

- [`IR_artifacts/Transformer/memory/runtime_memory.png`](IR_artifacts/Transformer/memory/runtime_memory.png)
- [`IR_artifacts/Transformer/memory/phase_breakdown.png`](IR_artifacts/Transformer/memory/phase_breakdown.png)
- [`IR_artifacts/Transformer/memory/static_estimation.png`](IR_artifacts/Transformer/memory/static_estimation.png)
- [`IR_artifacts/Transformer/memory/flops_estimation.png`](IR_artifacts/Transformer/memory/flops_estimation.png)

---

## 当前对 static/runtime 的解释方式

### 当前推荐口径

主轨道已经统一优先查看：
- **runtime 绝对峰值**
- **static estimated peak**

而不是把 runtime 增量峰值直接和 static 总峰值混在一起解释。

### 当前 graph gap 解释

对于：
- `compiled_no_recompute`
- `ATenIR_strat7_b1.0`

当前解释应结合：
- saved activation 减少量
- FW/BW phase peak
- runtime absolute peak
- runtime base memory
- gap breakdown

也就是说，不能只看一条总峰值线，而要看分项。

---

## 重要限制与边界

### 1. 不要把 internal partitioner 当成官方 graph API

`min_cut_rematerialization_partition` 是 PyTorch compile 内部机制的一部分，
但直接手工调用它的 backend 不能直接视为“官方完整 graph-level built-in 方法”。

### 2. region checkpoint 与 graph remat 不是同一种语义

- checkpoint / SAC：region replay
- ATenIR strategy 7：graph-level selective recomputation

因此不能直接用一句“谁更好”概括，必须先保证比较语义对齐。

### 3. static 数字不等于 runtime 真实峰值

当前 static 已经能较好反映方向，但仍会高估或低估某些 phase 收益。
因此所有主结论都应同时看：
- static
- runtime
- gap breakdown

---

## 当前最值得关注的文件

### 核心实现
- [`aten_recompute/core/partition.py`](aten_recompute/core/partition.py)
- [`aten_recompute/core/min_cut.py`](aten_recompute/core/min_cut.py)
- [`aten_recompute/core/compiler.py`](aten_recompute/core/compiler.py)

### 主轨道实验
- [`examples/transformer/benchmark_flow/runtime_phase.py`](examples/transformer/benchmark_flow/runtime_phase.py)
- [`examples/transformer/benchmark_flow/static_phase.py`](examples/transformer/benchmark_flow/static_phase.py)
- [`examples/transformer/benchmark_flow/strategy.py`](examples/transformer/benchmark_flow/strategy.py)

### 图语义与 capture 对照
- [`examples/transformer/meta_pipeline.py`](examples/transformer/meta_pipeline.py)
- [`aten_recompute/utils/graph_utils.py`](aten_recompute/utils/graph_utils.py)

### 主输出
- [`IR_artifacts/Transformer/memory/memory_report.json`](IR_artifacts/Transformer/memory/memory_report.json)

---

## 当前整理状态说明

当前仓库已经进入一轮结构整理阶段，后续还会继续处理：
- 胖文件拆分
- 重复定义清理
- 输入输出收口
- README 与真实入口彻底对齐
- diagnostic 路径和主路径的职责边界进一步清晰化

所以当前 README 已经比旧版本更接近真实项目状态，但后续仍会继续迭代。
