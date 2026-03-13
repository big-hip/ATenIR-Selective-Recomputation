# ATenIR Selective Recomputation

基于 PyTorch `torch.compile` + AOT Autograd 的**选择性激活重计算**框架。通过在 ATen IR 层面的 joint graph 切分阶段（partition_fn）控制哪些中间激活保存、哪些在反向中重计算，以换取显著的显存节省。

---

## 核心思路

深度学习训练时，前向传播产生的中间激活值必须保留至反向传播使用，这是显存压力的主要来源。**激活重计算**（Gradient Checkpointing）的策略是：不保存某些中间激活，而在反向传播时重新运行一遍对应的前向计算。

本项目通过自定义 `partition_fn` 嵌入 AOT Autograd 的 joint graph 切分流程，在切分阶段直接控制 `saved_values` 列表，决定哪些激活保存到前向输出、哪些在反向中自动重计算。整个过程与 `torch.compile` 完全集成，支持多种粒度的选择策略，无需手动捕获或包装计算图。

---

## 项目结构

```
ATenIR-Selective-Recomputation/
├── Dockerfile                             # Docker 镜像定义（PyTorch 2.6 + CUDA 12.4）
├── docker-compose.yml                     # 一键启动（含 GPU 挂载）
├── Makefile                               # 常用命令快捷入口
├── requirements.txt                       # Python 依赖
├── aten_recompute/                        # 核心库
│   ├── __init__.py                        # 包入口，导出核心公开 API
│   ├── core/                              # 核心编译与重计算逻辑
│   │   ├── partition.py                   # 核心 partition_fn：策略解析、层级传播、图切分
│   │   ├── tag.py                         # mark_layer 自定义算子 + inject_layer_tags
│   │   ├── min_cut.py                     # 策略 7：基于最小割算法的最优重计算
│   │   └── compiler.py                   # CompilerBackend：torch.compile 自定义后端
│   ├── analysis/                          # 内存分析与估算
│   │   ├── profiler.py                    # MemoryProfiler：运行时 GPU 内存采样 + 报告
│   │   ├── static.py                      # StaticEstimator：静态峰值显存估算
│   │   ├── comparison.py                  # SOTA 重计算方法理论对比表
│   │   └── _activations.py                # FW→BW 边界激活分析工具函数
│   └── utils/                             # 基础设施
│       ├── logger.py                      # 统一日志工具（含滚动文件 handler）
│       ├── save_ir.py                     # 保存计算图为 IR/DOT/Code 格式
│       ├── graph_utils.py                 # FX 图共享工具函数
│       └── checkpoint.py                  # PyTorch 原生 checkpoint 包装（用于对比）
└── examples/
    └── transformer/
        ├── model.py                       # 标准 Encoder-Decoder Transformer 实现
        └── main.py                        # 完整训练 + 显存对比流程入口
```

---

## 一键部署

### 方式一：Docker（推荐，适合论文复现/答辩演示）

前置要求：[Docker](https://docs.docker.com/get-docker/) + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)（GPU 支持）

```bash
# 一键构建并运行 Transformer 示例（默认策略 6：自动廉价重计算）
make run

# 指定其他策略
make run-strategy STRATEGY='{"1": null}'    # 全部重计算
make run-strategy STRATEGY='{"7": 1.0}'     # min-cut 最优

# 进入容器交互式终端
make shell

# 查看所有可用命令
make help
```

也可直接使用 docker compose：

```bash
docker compose build
docker compose up
```

IR 产物会自动挂载到宿主机的 `IR_artifacts/` 目录。

### 方式二：本地安装（pip / conda）

> 要求 PyTorch >= 2.1（需要 AOT Autograd、`torch.library.custom_op`、FX Graph 等特性）。

```bash
# 使用 Makefile 一键安装（需先激活 conda/venv）
make setup-local

# 或手动安装
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

可选依赖（已包含在 `requirements.txt` 中）：

```bash
pip install networkx    # 策略 7（min-cut 最优重计算）需要
pip install psutil      # CPU 环境下运行时内存分析需要
pip install matplotlib  # 生成内存分析图表需要
```

---

## 快速开始

以 Transformer 为例，完整流程分三步：

### 第一步：注入层级标签

使用 `inject_layer_tags` 为每一层注册前向钩子，打上 `mark_layer` 自定义算子标记。该函数接受通用的 `(nn.Module, rank)` 对列表，与模型架构无关：

```python
from aten_recompute.core import inject_layer_tags

# 构造 (层, rank) 对列表
enc_layers = [(layer, i) for i, layer in enumerate(model.encoder_layers)]
dec_layers = [
    (layer, len(model.encoder_layers) + i)
    for i, layer in enumerate(model.decoder_layers)
]
handles = inject_layer_tags(enc_layers + dec_layers)
```

### 第二步：使用 CompilerBackend 编译模型

`CompilerBackend` 是 `torch.compile` 的自定义后端，内部通过 `partition_fn` 在 AOT Autograd 的 joint graph 切分阶段完成：层级传播、策略筛选、mark_layer 清理、图切分。切分后的 FW/BW 图直接交给 Inductor 编译执行：

```python
from aten_recompute.core import CompilerBackend

backend = CompilerBackend(strategy_config={"6": 0}, save_ir=True)
compiled_model = torch.compile(model, backend=backend, dynamic=True)
```

### 第三步：标准训练循环

编译后的模型直接用于训练，无需额外包装：

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for step in range(n_steps):
    optimizer.zero_grad()
    output = compiled_model(src_data, tgt_data[:, :-1])
    loss = criterion(output.view(-1, vocab_size), tgt_data[:, 1:].view(-1))
    loss.backward()
    optimizer.step()
```

### 运行示例

```bash
cd examples/transformer
export RECOMPUTE='{"6": 0}'
export RECOMPUTE_LOG_LEVEL=INFO
python main.py
```

示例程序会依次运行五种配置进行对比：
1. **eager baseline**：无编译、无 checkpoint
2. **compiled no recompute**：编译但不重计算（隔离编译加速效果）
3. **PyTorch checkpoint**：原生 `torch.utils.checkpoint` 包装
4. **PyTorch SAC**：选择性激活检查点（Selective Activation Checkpointing）
5. **ATenIR recompute**：本框架的选择性重计算

优化后的图文件保存在：

```
IR_artifacts/<MODEL_NAME>/[runs/<RUN_ID>/]
├── partition/    # 切分后的 FW/BW 图（IR、DOT、Code）
└── memory/       # 内存分析报告（JSON + 图表）
```

---

## 重计算策略

通过环境变量 `RECOMPUTE` 以 JSON 格式指定，或通过 `CompilerBackend(strategy_config=...)` 传入。支持以下七种策略：

| 策略 ID | 含义 | 配置示例 |
|--------|------|---------|
| `"0"` | 不执行重计算（默认） | `'{}'` |
| `"1"` | 重计算所有层的激活 | `'{"1": null}'` |
| `"2"` | 按节点名称关键字选择 | `'{"2": ["%relu", "%norm"]}'` |
| `"3"` | 按层步长选择（start, stride） | `'{"3": [0, 2]}'`（第 0、2、4… 层） |
| `"4"` | 按比例选择前 N% 层 | `'{"4": 0.5}'`（前 50% 层） |
| `"5"` | 按 ATen 算子类型选择 | `'{"5": ["relu", "dropout"]}'` |
| `"6"` | 自动廉价重计算（按链深度） | `'{"6": 0}'`（深度 ≤ 0 的单算子重计算） |
| `"7"` | min-cut 最优重计算 | `'{"7": 1.0}'`（需安装 networkx） |

若未设置 `RECOMPUTE` 或配置无效，默认不执行任何重计算。

---

## 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `RECOMPUTE` | `{}` | 重计算策略配置（JSON） |
| `RECOMPUTE_LOG_LEVEL` | `INFO` | 日志级别（DEBUG/INFO/WARNING/ERROR） |
| `RECOMPUTE_LOG_FILE` | （无） | 日志文件名（设置后输出到 `IR_artifacts/logs/` 下） |
| `MODEL_NAME` | `default_model` | 模型名称（影响 IR 保存路径） |
| `PROJECT_ROOT` | 自动推导（`os.getcwd()`） | 项目根目录（IR 输出的基准路径） |
| `RUN_ID` | （无） | 实验批次 ID（设置后 IR 输出隔离到 `runs/<RUN_ID>/` 子目录） |

---

## 模块说明

### `aten_recompute/core/tag.py`

注册 `my_compiler::mark_layer` 自定义 ATen 算子（恒等映射，仅携带层号信息），并通过 `inject_layer_tags(layers)` 向指定层的前向钩子中插入标记调用。

`inject_layer_tags` 接受通用的 `Iterable[Tuple[nn.Module, int]]` 参数，调用方自行构造层与 rank 的对应关系，不依赖特定模型架构。`mark_layer` 在 `partition_fn` 内部被分析后自动清理，无需手动移除。

### `aten_recompute/core/partition.py` — `selective_recompute_partition`

核心 `partition_fn`，在 AOT Autograd 的 joint graph 切分阶段执行以下流程：

1. **Inductor 预处理**：`_recursive_joint_graph_passes`（replace_random、constant folding 等）
2. **节点分类**：收集 baseline `saved_values`（BW 需要的所有前向值）
3. **层级传播**：`_propagate_layer_ranks` 通过数据流传播为前向节点分配 `layer_Rank`
4. **策略筛选**：`_apply_strategy` 根据策略编号决定哪些 `saved_values` 改为重计算
5. **精确补充**：`_find_required_primals` 仅补充重计算链实际依赖的 primal，避免过度膨胀
6. **清理 mark_layer**：`_cleanup_mark_layer` 移除标记节点
7. **图切分**：调用 `_extract_fwd_bwd_modules` 生成独立的 FW/BW 图
8. **BW 重排序**：`reordering_to_mimic_autograd_engine` 将重计算节点推迟到真正需要时执行

### `aten_recompute/core/min_cut.py` — `solve_min_cut_recompute`

基于最小割算法的最优重计算策略（策略 7）。将 joint graph 的前向部分建模为流网络：

- 每个节点拆分为 `N_in` / `N_out`，内部边容量与节点内存成正比，距离反向越远的节点容量越高
- source 连接禁止重计算的节点（计算密集型 op、随机 op、primal）
- sink 连接反向需要的 saved_values
- 最小割将节点分为"保存"和"重计算"两组，最优平衡内存与计算
- 支持 `memory_budget` 参数控制重计算比例

### `aten_recompute/core/compiler.py` — `CompilerBackend`

`torch.compile` 自定义后端，在 `__call__` 中注入 `selective_recompute_partition` 作为 `partition_fn`。切分后的 FW/BW 图通过 `compile_fx_inner` 交给 Inductor 编译执行，可选保存切分后的图 IR 产物。

### `aten_recompute/analysis/profiler.py` — `MemoryProfiler`

运行时内存分析器，提供两类能力：

- **静态估算**：比较 Pass 前后 FX 图的 `meta['val']`，推导激活内存节省量，无需实际运行
- **运行时测量**：对训练步骤进行峰值显存与耗时采样，支持 CUDA 和 CPU（CPU 需 psutil）

支持多组配置对比（eager / compiled / checkpoint / SAC / ATenIR），输出 JSON 报告和可视化图表。

### `aten_recompute/analysis/static.py` — `StaticEstimator`

基于 FX 图的静态峰值显存估算器。通过分析 AOT Autograd 编译后的 FW/BW GraphModule 中的 FakeTensor 元信息，仿真 storage 生命周期，估算训练峰值显存。无需 GPU。

### `aten_recompute/analysis/comparison.py`

SOTA 重计算方法理论对比表，覆盖 PyTorch 内建方法（Checkpoint、SAC）、ATenIR 策略、以及学术界代表性方法（DTR、Checkmate、Rockmate）。

### `aten_recompute/utils/checkpoint.py`

封装 PyTorch 原生 `torch.utils.checkpoint`，提供 `apply_activation_checkpoint` / `remove_activation_checkpoint`，用于与本框架进行公平的内存/性能对比。

### `aten_recompute/utils/graph_utils.py`

共享的 FX 图工具函数（`get_output_node`、`get_saved_activations`），供分析模块复用。

### `aten_recompute/utils/save_ir.py`

保存 FX GraphModule 为多种格式（可读 IR、DOT 可视化拓扑、底层 Python Code），支持通过 `RUN_ID` 环境变量隔离不同批次的实验数据。

---

## 已知限制

- 策略 7（min-cut）需要额外安装 `networkx`
- `CompilerBackend` 目前仅通过 `fw_compiler` / `bw_compiler` 捕获图的副本用于调试，不支持直接访问 joint graph
- 静态估算中 checkpoint 公式准确度约 52%（无法捕获 cuBLAS workspace 和 CUDA allocator 开销）

---

## 技术背景

本项目利用了以下 PyTorch 内部机制：

- **AOT Autograd**：在执行前将 `nn.Module` 的前向与反向联合编译为 ATen IR joint graph，通过 `partition_fn` 控制切分策略
- **`torch.library.custom_op`**：注册自定义算子并提供 Fake Tensor 推导规则与 Autograd 规则，保证与 Dynamo/AOT 的兼容性
- **FX GraphModule**：PyTorch 的 IR 表示与变换框架，支持节点级别的图分析与变换
- **`torch.compile`**：PyTorch 2.x 的编译入口，通过自定义 backend 注入重计算逻辑，与 Inductor 优化管线无缝集成
