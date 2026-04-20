# 11 — L2 仿真精度优化：MRE 21% → 7%

> **文档定位**: 记录 L2 FX 图仿真引擎的两个核心缺陷的发现过程、根因分析、
> 修复方案及跨配置验证结果。修改文件为 `toolkit/simulation/graph_estimator.py`。
>
> **前置阅读**: `08-static-simulation-deep-dive.md`（L2 仿真引擎全链路解析）

---

## 一、问题背景

修复前的 L2 仿真平均 MRE（Mean Relative Error）约 **21%**，主要表现为：

- **BW 阶段系统性高估 21-28%**（对所有配置普遍存在）
- **FW 阶段方向不一致**：部分配置高估，部分配置低估达 50%

该误差在多模型（LLaMA）、多配置（不同 layers/hidden/batch/seq 组合）下均存在，
属于仿真引擎的**系统性缺陷**，而非单一配置的偶发问题。

### 修复前基线数据

| 配置 | RT 峰值 | L2 峰值 | MRE | FW 误差 | BW 误差 |
|------|---------|---------|-----|---------|---------|
| 6L/512H b=8 s=128 | 1.0 GB | 1.2 GB | 21.6% | -50.9% | +21.6% |
| 2L/512H b=4 s=128 | 494 MB | 631 MB | 27.7% | -26.3% | +27.7% |
| 6L/512H b=4 s=64 | 550 MB | 705 MB | 28.1% | +3.4% | +28.1% |
| 4L/512H b=4 s=128 | 593 MB | 759 MB | 28.0% | -27.7% | +28.0% |
| 6L/256H b=8 s=128 | 771 MB | 769 MB | 0.3% | -57.0% | -0.3% |

**平均 MRE = 21.1%**

---

## 二、根因分析

### 2.1 诊断方法

通过以下步骤定位误差来源：

1. **分离图级峰值与训练步峰值**：分别检查 `fw_graph_peak`、`bw_graph_peak`、`grad_bytes` 的贡献
2. **对比 RT 分阶段峰值**：`measure_phased` 提供 fw_peak/bw_peak/opt_peak 三阶段独立基准
3. **逐节点审计 FW 图输出**：统计 output 节点的 view/non-view 分类及其 base tensor 关系

### 2.2 发现一：FW 图 view 输出未钉住 base tensor（严重低估 FW 峰值）

**现象**：FW 图的 `fw_graph_peak` 远小于运行时 FW 激活增量。

**诊断数据**（6L/512H b=8 s=128 配置）：

```
FW output inputs: 221 个节点
  其中 view 节点: 144 个（65.2%）
  非 view 节点:   77 个

view 输出中 base 未被 pin 的: 118 个独立 base 节点
  未钉住的 base 总字节: 447.9 MB  ← 被错误释放！
```

**根因**：

`pin_output_inputs=True` 仅钉住 output 节点的**直接输入**。当这些输入是 view 节点时：

```
mm → _unsafe_view → transpose → [保存为 FW output]
│        │             │
│        │             └── 被 pin（但 view 节点 size=0，不占内存）
│        └── 被 pin（同上，size=0）
└── 未被 pin！→ 在 last_use 到达后被释放 → 实际 storage 被丢弃
```

view 节点在仿真中大小为 0（因为 `is_view_node` 返回 True 后被跳过），
其内存来自 base tensor。但 base tensor 不在 `output_inputs` 集合中，
因此在 `last_use` 到达后被释放——**尽管 view 的 storage 仍然需要保留**。

**影响**：447.9 MB 的 base 内存被错误释放，导致 FW 峰值严重低估。

### 2.3 发现二：BW 峰值公式 `grad_bytes` 双重计算（系统性高估 BW 峰值）

**旧公式**：

```python
bw_peak = static_base + grad_bytes + bw_graph_peak   # ← grad_bytes 重复计算
```

**根因**：BW 图的输出节点就是梯度 tensor。在 BW 图中：

```
BW 输出节点: 57 个
  其中 view:     56 个
  非 view:        1 个
  总字节:       197.0 MB = param_bytes（完全等于参数量）
```

这些梯度节点在 BW 图执行过程中被分配，其 `last_use` 是 output 节点（图的最后一个节点），
因此它们在整个 BW 仿真过程中**一直存活**。也就是说 `bw_graph_peak` 已经自然包含了梯度内存。

再单独加 `grad_bytes`（= `param_bytes` = 197.0 MB）等于**双重计算了梯度内存**。

**影响**：BW 峰值被系统性高估约 `param_bytes` 的量，对 6L/512H 模型为 197 MB（~20% 峰值）。

---

## 三、修复方案

### 3.1 修复 A：钉住 view 输出的 base tensor

**新增辅助函数** `_find_view_base()`（第 9-43 行）：

```python
def _find_view_base(node: fx.Node) -> fx.Node:
    """沿 view 链回溯，找到真正拥有 storage 的 base 节点。

    view -> 共享 storage 的 input -> ... -> 非 view 节点（base）
    """
    visited = set()
    current = node
    while current.op in ("call_function", "call_method") and is_view_node(current):
        if id(current) in visited:
            break
        visited.add(id(current))
        val = current.meta.get("val")
        if not isinstance(val, torch.Tensor):
            break
        val_cdata = val.untyped_storage()._cdata
        # 找到与当前节点共享 storage 的输入节点
        for inp in current.all_input_nodes:
            inp_val = inp.meta.get("val")
            if isinstance(inp_val, torch.Tensor):
                if inp_val.untyped_storage()._cdata == val_cdata:
                    current = inp       # 向上回溯
                    break
    return current  # 非 view 节点 = base allocation
```

**在 `estimate_graph_peak()` 中应用**（第 54-64 行）：

```python
if pin_output_inputs:
    for node in nodes:
        if node.op == "output":
            output_inputs.update(node.all_input_nodes)
    # ★ 新增：为每个被 pin 的 view 节点，回溯找到 base 并一并 pin
    for n in list(output_inputs):
        if n.op in ("call_function", "call_method") and is_view_node(n):
            base = _find_view_base(n)
            output_inputs.add(base)
```

**效果**：FW 图的 118 个未钉住的 base 节点（447.9 MB）现在被正确保留，
FW 峰值从 243.1 MB 提升到接近运行时实测值。

### 3.2 修复 B：移除 BW 公式中的 `grad_bytes`，改用 `pin_output_inputs=True`

**旧代码**：

```python
bw_result = estimate_graph_peak(bw_gm, pin_output_inputs=False)
bw_peak = static_base + grad_bytes + bw_graph_peak
```

**新代码**（第 159-184 行）：

```python
bw_result = estimate_graph_peak(bw_gm, pin_output_inputs=True)   # ★ 改为 True
# ...
# grad_bytes 不再加入 bw_peak：梯度已作为 BW 图输出节点被建模
bw_peak = static_base + bw_graph_peak                             # ★ 无 + grad_bytes
```

**原理**：

1. BW 图的输出节点是梯度 tensor（56/57 为 view），它们在图执行期间一直存活
2. `pin_output_inputs=True` + view-base 回溯确保梯度的底层 storage 不被错误释放
3. 不再需要外部添加 `grad_bytes`，因为梯度内存已在图级仿真中被完整建模

### 3.3 公式对比

```
旧公式:
  fw_peak  = static_base + fw_graph_peak(pin=True, 无 view-base 回溯)
  bw_peak  = static_base + grad_bytes + bw_graph_peak(pin=False)

新公式:
  fw_peak  = static_base + fw_graph_peak(pin=True, 含 view-base 回溯)  ← FW 更准确
  bw_peak  = static_base + bw_graph_peak(pin=True, 含 view-base 回溯)  ← 不重复计梯度
```

---

## 四、验证结果

### 4.1 7 配置综合测试

模型：LLaMA，优化器：SGD，运行时基准：`measure_phased` 3 次取 IQR mean。

| 配置 | RT 峰值 | L2 峰值 | **MRE** | FW 误差 | BW 误差 | 峰值阶段 |
|------|---------|---------|---------|---------|---------|---------|
| 6L/512H b=8 s=128 | 1.0 GB | 1.0 GB | **2.4%** | -1.1% | +2.4% | BW |
| 2L/512H b=4 s=128 | 494 MB | 482 MB | **2.5%** | +1.7% | -2.5% | BW |
| 6L/512H b=4 s=64 | 550 MB | 508 MB | **7.7%** | +22.0% | -7.7% | BW |
| 4L/512H b=4 s=128 | 593 MB | 586 MB | **1.2%** | +6.2% | -1.2% | BW |
| 6L/256H b=8 s=128 | 771 MB | 688 MB | **10.8%** | -15.4% | -10.8% | BW |
| 6L/512H b=2 s=128 | 554 MB | 511 MB | **7.7%** | +21.7% | -7.7% | BW |
| 2L/256H b=8 s=128 | 646 MB | 544 MB | **15.8%** | -20.5% | -15.8% | BW |

### 4.2 汇总指标

| 指标 | 修复前 | 修复后 | 改善 |
|------|--------|--------|------|
| **平均 MRE** | 21.1% | **6.9%** | **↓ 14.2pp** |
| **最大 MRE** | 28.1% | 15.8% | ↓ 12.3pp |
| **最小 MRE** | 0.3% | 1.2% | — |
| BW 平均误差 | +21.0% (高估) | -6.1% (略低) | 方向修正 |
| FW 平均误差 | -31.7% (严重低估) | +2.1% (接近) | **方向修正** |

### 4.3 测试回归

修复后运行全量测试：

```
======================= 106 passed, 0 failed =======================
```

涉及更新的测试用例：
- `test_peak_formula`：`bw_total` 不再包含 `grad_bytes`
- `test_l2_absolute_peaks`：断言 `bw_peak == base + bw_graph_peak`

---

## 五、残余误差分析

修复后仍有 ~7% 平均 MRE，主要来自以下无法在 L2 层级建模的因素：

### 5.1 CUDA "dark memory"（~18 MB 固定偏移）

```
runtime_base - formula_base ≈ 16-19 MB
```

包含：CUDA context 初始化、cuBLAS workspace、模型内部 buffer（如 RoPE 的 cos/sin cache）。
对小模型（hidden=256）影响较大（~2-3% 峰值），大模型可忽略。

### 5.2 梯度 view-base 覆盖不完全

BW 图 57 个梯度输出中 56 个是 view，对应 56 个独立 base 节点：

| hidden | grad_bytes | base_bytes（pin 实际捕获） | 覆盖率 |
|--------|-----------|--------------------------|--------|
| 512 | 197.0 MB | 134.5 MB | 68.3% |
| 256 | 80.5 MB | 49.3 MB | 61.2% |

覆盖率 < 100% 是因为多个梯度 view 可能共享同一个 base tensor，
而 `val_bytes` 按逻辑元素数（`numel * element_size`）计算，
view 的逻辑大小可能超过 base 的逻辑大小（如 expand 后 clone 再 view 的情况）。

### 5.3 BW 峰值时间点的梯度积累

BW 图按逆序执行层（Layer N → Layer 1）。峰值通常出现在 BW 初期
（此时所有 saved activations 尚存 + 第一层的中间结果）。
此时仅有少量梯度已计算完毕，pin 带来的额外内存较小。

```
BW 峰值时刻的梯度占比 ≈ 1/N_layers × param_bytes  （仅当前层梯度已分配）
```

而旧公式直接加整个 `param_bytes`，对多层模型严重高估。
新公式通过图级仿真自然反映了这一时序特征。

### 5.4 Allocator 对齐与碎片化

CUDA Caching Allocator 的对齐粒度并非固定 512B：

| 分配大小 | 实际粒度 |
|---------|---------|
| ≤ 512B | 512B block |
| 512B - 1MB | 512B 对齐 |
| 1MB - 10MB | 2MB roundup |
| > 10MB | 20MB roundup |

大 tensor 的对齐浪费我们未建模。此外 block splitting/coalescing
行为导致 `allocated_bytes.all.peak` 与纯 live-range 仿真有偏差。

这些因素需要 **L3 (BFC 仿真)** 或 **L4 (Allocator Replay)** 层级才能覆盖。

---

## 六、技术细节：FX 图生成代码中的 `del` 语句

PyTorch FX `Graph.python_code()` 在代码生成时会插入 `= None`（等效于 `del`）语句：

```python
# torch/fx/graph.py 第 564-604 行
node_to_last_use: Dict[Node, Node] = {}
# ... 逆序遍历找到每个节点的最后使用位置 ...

def delete_unused_values(user: Node):
    nodes_to_delete = user_to_last_uses.get(user, [])
    if len(user.users.keys()) == 0:
        nodes_to_delete.append(user)        # side-effect 节点也释放
    to_delete_str = " = ".join([repr(n) for n in nodes_to_delete] + ["None"])
    body.append(f";  {to_delete_str}\n")
```

生成代码示例：

```python
mul_49 = torch.ops.aten.mul.Tensor(embedding, rsqrt)
mul_53 = torch.ops.aten.mul.Tensor(primals_5, mul_49)
t = torch.ops.aten.t.default(primals_6);  primals_6 = None     # ← primals_6 被释放
view_1 = torch.ops.aten.view.default(mul_53, [mul_57, 512])
mm = torch.ops.aten.mm.default(view_1, t)
_unsafe_view = torch.ops.aten._unsafe_view.default(mm, [...]);  mm = None  # ← mm 被释放
```

这证实了我们的 live-range 仿真与实际执行的内存释放时序**基本一致**——
都在最后使用后立即释放。仿真的误差主要来自**view aliasing** 和 **allocator 行为**，
而非释放时序。

---

## 七、与已有研究的对比

| 方法 | MRE | 特点 |
|------|-----|------|
| **DNNMem** (MSR, 2020) | 10-30% | 静态公式 + op 级别估算，不处理 view |
| **xMem** (2024) | <5% | BFC allocator 完整仿真（L3 级别） |
| **Checkmate/MONeT** | N/A | 关注优化目标（再物化调度），非精度 |
| **本项目 L2（修复后）** | **6.9%** | FX 图 live-range + view-base pin，无 allocator 仿真 |

本项目 L2 的精度**优于 DNNMem**，接近 xMem 的水平，
且不需要模拟 CUDA Caching Allocator 的 block 管理——
仅依赖 FX 图 meta 信息和 storage aliasing 检测。

---

## 八、修改文件清单

| 文件 | 改动 |
|------|------|
| `toolkit/simulation/graph_estimator.py` | 新增 `_find_view_base()`；`estimate_graph_peak` 增加 view-base pin；`estimate_training_peak` BW 改用 `pin=True`、移除 `grad_bytes` |
| `tests/test_simulation.py` | `test_peak_formula` 和 `test_l2_absolute_peaks` 断言更新 |

核心改动仅 **~40 行新增代码 + 2 行公式修改**，无 API 变更，完全向后兼容。

---

## 参考

- `08-static-simulation-deep-dive.md`：L2 仿真引擎全链路解析（修复前版本）
- `04-dev-log.md`：Bug B2（saved_act 双重计算）历史修复记录
- PyTorch: `torch/fx/graph.py` — `delete_unused_values()` 生成 `= None` 释放语句
- PyTorch: `torch/_functorch/partitioners.py` — `_extract_fwd_bwd_modules()` FW/BW 图分割
- xMem (2024): BFC allocator simulation for DNN memory estimation
- DNNMem (MSR, 2020): Static DNN memory estimation framework
