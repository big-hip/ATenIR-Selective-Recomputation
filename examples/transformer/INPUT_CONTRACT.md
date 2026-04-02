# Transformer 输入输出契约

当前 Transformer 主轨道已经不再是一个单纯的 `main.py` 演示脚本，而是由 benchmark / capture 两条主流程组成。

本文档只描述**当前推荐的真实用法**，并与 README 保持一致。

---

## 1. 当前推荐入口

### 推荐入口

- Benchmark：`examples/transformer/benchmark.py`
- Capture：`examples/transformer/capture.py`

### 统一入口

- `examples/transformer/main.py`

可通过：

```bash
python examples/transformer/main.py benchmark
python examples/transformer/main.py capture
```

说明：
- `main.py` 保留为统一 orchestrator；
- 日常使用时，`benchmark.py` / `capture.py` 更直接，也更清晰。

---

## 2. 主输入文件

### 主配置文件

- `examples/transformer/input.yaml`

### 模型配置文件

- `examples/transformer/model_config.yaml`

`input.yaml` 当前承担两类任务配置：
- `capture`
- `benchmark`

其基本结构为：

```yaml
version: 1

run:
  task: benchmark  # benchmark | capture

env:
  model_name: Transformer
  recompute:
    "7": 1.0
  recompute_log_level: INFO
  # project_root: /abs/project/root
  # run_id: exp_001
  # save_joint_graph: true

capture:
  mode: static
  batch_size: 2
  seq_len: 32
  dynamic: false
  static_profile: fast
  compare_runtime: true

benchmark:
  batch_size: 64
  n_steps: 10
  run_correctness_checks: true
```

---

## 3. 字段语义

### `run.*`
- `run.task`：主任务选择，`benchmark` 或 `capture`

### `env.*`
映射到运行时环境变量：
- `env.model_name -> MODEL_NAME`
- `env.recompute -> RECOMPUTE`
- `env.recompute_log_level -> RECOMPUTE_LOG_LEVEL`
- 可选：`project_root` / `run_id` / `save_joint_graph`

### `capture.*`
仅 capture 主流程使用：
- `mode`
- `batch_size`
- `seq_len`
- `dynamic`
- `static_profile`
- `compare_runtime`

### `benchmark.*`
仅 benchmark 主流程使用：
- `batch_size`
- `n_steps`
- `run_correctness_checks`

---

## 4. 当前主方法矩阵语义

### region / 用户级
- `eager_baseline`
- `eager_checkpoint`
- `eager_sac`

### graph / 主轨道
- `compiled_no_recompute`
- `ATenIR_strat7_b1.0`

### diagnostic
- internal partitioner-only 路径
- 其它 strategy 家族的诊断性方法

注意：
- `min_cut_rematerialization_partition` 的手工 backend 调用不能当作 PyTorch 官方完整 graph built-in 方法；
- 当前主输出只以主矩阵为准，diagnostic 不进入默认主结论。

---

## 5. 输出目录约定

### 主输出根目录

默认写入：

```text
<PROJECT_ROOT>/IR_artifacts/
```

如果未显式设置 `PROJECT_ROOT`，则由当前工作目录决定。

### Transformer 主输出目录

```text
IR_artifacts/Transformer/
```

其中最重要的是：

```text
IR_artifacts/Transformer/memory/
├── memory_report.json
├── static_estimation.json
├── flops_estimation.json
├── runtime_memory.png
├── phase_breakdown.png
├── static_estimation.png
└── flops_estimation.png
```

### 当前主报告

- `memory_report.json`

当前它已经包含：
- runtime 主结果
- absolute peak reporting
- `graph_gap_breakdown`
- `bw_gap_breakdown`

---

## 6. 关于 `input_capture.yaml`

当前仓库中仍保留：
- `examples/transformer/input_capture.yaml`

它更接近历史 / capture-only 配置，当前**不是主推荐配置入口**。

后续整理时将进一步决定：
- 是否保留为 capture-only 简化配置；
- 或者直接并入 `input.yaml` / 删除。

---

## 7. 关于 `examples/transformer/IR_artifacts/`

当前主流程的正式输出应以仓库根目录下的：
- `IR_artifacts/`
为准。

如果 `examples/transformer/IR_artifacts/` 中仍存在内容，它更可能是：
- 历史局部运行残留；
- 或旧阶段输出；
- 不应再被视为当前主输出根目录。

---

## 8. 当前推荐命令

### Benchmark

```bash
python examples/transformer/benchmark.py
```

### Capture

```bash
python examples/transformer/capture.py
```

### 统一入口模式

```bash
python examples/transformer/main.py benchmark
python examples/transformer/main.py capture
```

---

## 9. 契约边界

这个输入输出契约只描述：
- 当前主轨道推荐入口；
- 当前主配置文件；
- 当前主输出目录与主报告；
- 当前主矩阵与方法语义。

它不负责描述：
- diagnostic 细节；
- 历史实验残留目录的全部来源；
- 所有旧脚本兼容行为。

后续如果继续进行 I/O 收口，这份文档会和 README 一起继续更新。
