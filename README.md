# ATenIR Selective Recomputation

基于 PyTorch FX + AOT Autograd 的**选择性激活重计算**框架。通过在 ATen IR 层面分析并变换前向/反向计算图，在反向传播时按需重计算指定激活值，以换取显著的显存节省。

---

## 核心思路

深度学习训练时，前向传播产生的中间激活值必须保留至反向传播使用，这是显存压力的主要来源。**激活重计算**（Gradient Checkpointing）的策略是：不保存某些中间激活，而在反向传播时重新运行一遍对应的前向计算。

本项目在 AOT Autograd 捕获的 ATen IR 图上实现这一变换，支持多种粒度的选择策略，并与 PyTorch 自动微分系统完全兼容。

---

## 项目结构

```
ATenIR-Selective-Recomputation/
├── aten_recompute/                        # 核心库
│   ├── __init__.py                        # 包入口，统一 logger 初始化
│   ├── core/
│   │   ├── Tag.py                         # mark_layer 自定义算子 + inject_layer_tags
│   │   ├── get_activation_layer_ranks.py  # 数据流传播，推导激活所属层级
│   │   ├── recompute.py                   # ActivationRecomputation 图变换引擎
│   │   ├── Recom_pass.py                  # RecomputePass 策略调度器
│   │   └── train_recomputed.py            # 训练包装器（autograd.Function）
│   ├── get_Aten_IR/
│   │   └── Graph_compile_capture.py       # GraphCapture：捕获 FW/BW 计算图
│   └── utils/
│       ├── logger.py                      # 统一日志工具
│       └── save_ir.py                     # 保存计算图为 IR/DOT/Code 格式
└── example/
    └── Transformer/
        ├── Transformer.py                 # 标准 Transformer 实现
        └── main.py                        # 完整训练流程入口
```

---

## 安装依赖

```bash
pip install torch torchvision
```

> 要求 PyTorch >= 2.1（需要 AOT Autograd、`torch.library.custom_op`、FX Graph 等特性）。

---

## 快速开始

以 Transformer 为例，完整流程分五个阶段：

### 阶段一：注入层级标签

在模型被 Dynamo 捕获之前，用 `inject_layer_tags` 向每一层注册前向钩子，为该层输入张量打上 `mark_layer` 自定义算子标记：

```python
from aten_recompute.core import inject_layer_tags
inject_layer_tags(transformer)
```

### 阶段二：捕获 FW/BW 图

`GraphCapture` 内部通过 AOT Autograd 的 `fw_compiler` / `bw_compiler` 回调捕获 ATen IR 级别的前向和反向图：

```python
from aten_recompute.get_Aten_IR import GraphCapture

graph_capture = GraphCapture(transformer, src_data, tgt_data[:, :-1])
compiled_transformer = graph_capture.compile()

# 预热：触发真正的 AOT Autograd 捕获
output = compiled_transformer(src_data, tgt_data[:, :-1])
loss = criterion(output.view(-1, tgt_vocab_size), tgt_data[:, 1:].view(-1))
loss.backward()
# 此时 graph_capture.FW_gm / BW_gm 已就绪
```

### 阶段三：运行重计算 Pass

通过环境变量 `RECOMPUTE` 指定策略，`RecomputePass` 自动分析层级并变换图：

```bash
export RECOMPUTE='{"1": null}'   # 全部重计算
```

```python
from aten_recompute.core import RecomputePass

recompute_pass = RecomputePass({"FW": graph_capture.FW_gm, "BW": graph_capture.BW_gm})
recompute_pass.run()

fw_gm_opt = recompute_pass.fw_gm
bw_gm_opt = recompute_pass.bw_gm
```

### 阶段四：清理并保存图

训练前须移除 `mark_layer` 节点和 Dynamo 注入的 `layer_Rank` kwargs（`cleanup_for_training`），然后将优化图保存为可视化文件：

```python
from aten_recompute.utils import save_ir_and_dot

fw_gm_opt = cleanup_for_training(fw_gm_opt)
bw_gm_opt = cleanup_for_training(bw_gm_opt)

save_ir_and_dot(fw_gm_opt, model_name="Transformer", subfolder="recompute", graph_name="FW_recomputed")
save_ir_and_dot(bw_gm_opt, model_name="Transformer", subfolder="recompute", graph_name="BW_recomputed")
```

### 阶段五：训练

`run_training` 将优化后的 FW/BW 图包装为 `torch.autograd.Function`，接入标准 PyTorch 训练循环：

```python
from aten_recompute.core import run_training

run_training(
    fw_gm=fw_gm_opt,
    bw_gm=bw_gm_opt,
    model=transformer,
    src_data=src_data,
    tgt_data=tgt_data,
    tgt_vocab_size=tgt_vocab_size,
    criterion=criterion,
    graph_capture=graph_capture,
)
```

### 运行示例

```bash
cd example/Transformer
export RECOMPUTE='{"1": null}'
export RECOMPUTE_LOG_LEVEL=INFO
python main.py
```

优化后的图文件保存在：

```
IR_artifacts/Transformer/runs/<RUN_ID>/
├── capture/    # 原始捕获的 FW/BW 图
└── recompute/  # 重计算优化后的 FW/BW 图
```

---

## 重计算策略

通过环境变量 `RECOMPUTE` 以 JSON 格式指定，支持以下五种策略：

| 策略 ID | 含义 | 配置示例 |
|--------|------|---------|
| `"1"` | 重计算所有层的激活 | `'{"1": null}'` |
| `"2"` | 按节点名称关键字选择 | `'{"2": ["%relu", "%norm"]}'` |
| `"3"` | 按层步长选择（start, stride） | `'{"3": [0, 2]}'`（第 0、2、4… 层） |
| `"4"` | 按比例选择前 N% 层 | `'{"4": 0.5}'`（前 50% 层） |
| `"5"` | 按 ATen 算子类型选择 | `'{"5": ["relu", "dropout"]}'` |

若未设置 `RECOMPUTE` 或配置无效，默认不执行任何重计算。

---

## 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `RECOMPUTE` | `{}` | 重计算策略配置（JSON） |
| `RECOMPUTE_LOG_LEVEL` | `INFO` | 日志级别（DEBUG/INFO/WARNING/ERROR） |
| `RECOMPUTE_LOG_FILE` | `aten_recompute.log` | 日志文件名 |
| `MODEL_NAME` | `Transformer` | 模型名称（影响 IR 保存路径） |
| `PROJECT_ROOT` | 自动推导 | 项目根目录（IR 输出的基准路径） |

---

## 模块说明

### `aten_recompute/core/Tag.py`

注册 `my_compiler::mark_layer` 自定义 ATen 算子（恒等映射，仅携带层号信息），并通过 `inject_layer_tags(model)` 向 `encoder_layers` / `decoder_layers` 的前向钩子中插入标记调用。

**注意**：`mark_layer` 在分析阶段使用，训练前需通过 `cleanup_for_training` 移除（Dynamo 序列化会产生参数冲突）。

### `aten_recompute/core/get_activation_layer_ranks.py`

通过**数据流传播**（Dataflow Propagation）在 FW 图上为所有节点分配 `layer_Rank`：

- 遇到 `mark_layer` 节点时提取层号作为源头
- 普通节点继承所有输入节点中的最大层号（Max Rank，兼容残差/多分支）
- 最终收集 FW 图输出中被 BW 图消费的激活，提取它们的层号列表

### `aten_recompute/core/recompute.py` — `ActivationRecomputation`

核心图变换引擎，执行四步操作：

1. **过滤候选**：取用户指定节点与（FW 输出 ∩ BW 占位符）的交集
2. **BFS 上溯**：从目标激活向上寻找完整的可重计算子图
3. **修改 BW 图**：将子图节点复制进 BW 图，替换旧占位符
4. **修改 FW 图**：调整输出列表，消除死代码

### `aten_recompute/core/Recom_pass.py` — `RecomputePass`

策略调度器，解析 `RECOMPUTE` 环境变量后调用 `ActivationRecomputation`。

### `aten_recompute/core/train_recomputed.py`

将优化后的 FW/BW `GraphModule` 包装为 `torch.autograd.Function`：

- 自动推断 FW 图的调用约定（SymInt 位置、参数顺序）
- 张量型 saved 用 `save_for_backward`，标量型（SymInt）单独记录
- 兼容 `dynamic=True` 的动态形状模式

### `aten_recompute/get_Aten_IR/Graph_compile_capture.py` — `GraphCapture`

通过 AOT Autograd backend 回调捕获 FW/BW 图，同时用 `data_ptr()` 匹配参数顺序，确保 `fw_params_flat` 与 FW 图 primal 顺序严格一致。

---

## 已知限制

- `inject_layer_tags` 目前硬编码依赖模型具有 `encoder_layers` / `decoder_layers` 属性，用于其他架构时需自行修改
- `train_recomputed.py` 中 `make_recomputed_fn` 目前仅支持单一模型输出（`assert len(model_out_fw_indices) == 1`）
- GPT-2 示例（`example/Gpt2/`）依赖外部分析模块，当前无法独立运行

---

## 技术背景

本项目利用了以下 PyTorch 内部机制：

- **AOT Autograd**：在执行前将 `nn.Module` 的前向与反向联合编译为 ATen IR 图
- **`torch.library.custom_op`**：注册自定义算子并提供 Fake Tensor 推导规则，保证与 Dynamo/AOT 的兼容性
- **FX GraphModule**：PyTorch 的 IR 表示与变换框架，支持节点级别的图变换与重新编译
