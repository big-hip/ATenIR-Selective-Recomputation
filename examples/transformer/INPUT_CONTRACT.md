# Transformer 主链输入契约（YAML 版）

目标：主链输入统一为 YAML 文件，CLI 仅用于选择 config 与临时覆盖。

## 1. 唯一入口

- 入口文件：`examples/transformer/main.py`
- 默认配置文件：`examples/transformer/input.yaml`
- 入口命令：`python examples/transformer/main.py [task] --config <yaml>`

说明：
- `task` 可省略，省略时使用 YAML 的 `run.task`。
- `task` 给定时优先于 YAML。

## 2. YAML 输入结构

```yaml
version: 1

run:
  task: capture  # capture | benchmark

env:
  model_name: Transformer
  recompute:
    "6": 0
  recompute_log_level: INFO
  # project_root: /abs/project/root
  # run_id: exp_001
  # save_joint_graph: true

capture:
  mode: static            # runtime | static
  batch_size: 2
  seq_len: 32
  dynamic: false
  static_profile: fast    # high | fast
  compare_runtime: false
```

## 3. 字段语义

- `run.task`：主任务选择，`capture` 或 `benchmark`。
- `env.*`：映射到运行时环境变量。
  - `env.recompute` 会被序列化为 `RECOMPUTE` JSON 字符串。
- `capture.*`：转换为 capture 子任务参数。
  - `mode -> --mode`
  - `batch_size -> --batch-size`
  - `seq_len -> --seq-len`
  - `dynamic -> --dynamic`
  - `static_profile -> --static-profile`
  - `compare_runtime -> --compare-runtime`

## 4. CLI 规则

- `--config`：指定 YAML 文件路径（默认 `examples/transformer/input.yaml`）。
- `task`：可选，若提供则覆盖 `run.task`。
- 其余 CLI 参数作为 capture 透传参数，优先级高于 YAML（用于临时覆盖）。

## 5. 运行示例

### 使用 YAML 默认任务

```bash
python examples/transformer/main.py
```

### 用 YAML 跑 benchmark

```bash
python examples/transformer/main.py benchmark
```

### 在 YAML 基础上临时覆盖 capture 参数

```bash
python examples/transformer/main.py capture --mode runtime --batch-size 4
```
