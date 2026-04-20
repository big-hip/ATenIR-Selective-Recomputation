# 08 — 静态仿真引擎源码深度解析

> **文档定位**: 对本项目静态仿真引擎的全链路解析，
> 涵盖 L1 配置公式法、L2 FX 图事件驱动法、L2.5 融合感知仿真、L3 Scheduler 仿真、
> view 检测、512B 对齐、四峰值体系、运行时验证机制，以及各层精度对比与误差来源分析。
>
> **前置阅读**: `01-architecture.md` §七（四层仿真概述）
> **论文对应**: 第 2 章 §2.3-2.5

---

## 一、概述：为什么需要静态仿真

核心价值：**不需要真正跑完训练，就能估算不同重计算策略下的峰值显存**。

传统做法：实际执行 → 观察 `torch.cuda.max_memory_allocated()` → 得到峰值。
问题：每种策略都要跑一次，大模型耗时数小时。

静态仿真：分析编译产物（FX 图的 FakeTensor 元信息）→ 模拟 tensor 分配/释放 → 估算峰值。
数秒内完成，支持批量策略对比。

| 层级 | 名称 | 输入 | 精度 (aot_eager) | 精度 (inductor) | 用途 |
|------|------|------|-----------------|----------------|------|
| **L1** | Config 公式法 | 模型配置 | MRE ~15-25% | — | 快速粗估、选型 |
| **L2** | FX 图事件驱动 | fw_gm / bw_gm | **MRE 1-7%** | MRE 28-38% | 精确仿真 (aot_eager) |
| **L2.5** | 融合感知图遍历 | fw_gm / bw_gm + fusion groups | — | **MRE 8-12%** | 近似建模 fusion |
| **L3** | Inductor Scheduler | Scheduler 编译产物 | — | **MRE 5-7%** | 精确建模 fusion |

---

## 二、四峰值体系

训练一步包含三个阶段（Forward / Backward / Optimizer），每个阶段有不同的内存组成：

```
训练步内存时间线：

       ┌── FW ──┐┌── BW ──────┐┌── OPT ──┐
       │        ││            ││         │
 peak→ │ ▓▓▓▓▓▓ ││ ▓▓▓▓▓▓▓▓▓ ││ ▓▓▓▓▓  │
       │ ▓▓▓▓▓▓ ││ ▓▓▓▓▓▓▓▓▓ ││ ▓▓▓▓▓  │
       │ ░░░░░░ ││ ░░░░░░░░░ ││ ░░░░░  │ ← static_base (param + optim_states)
       └────────┘└───────────┘└────────┘

 ▓ = 动态部分（激活/梯度/临时内存）
 ░ = 固定部分（参数 + 优化器状态）
```

四峰值公式：

```python
static_base = param_bytes + optim_bytes           # 参数 + 优化器状态（SGD=0倍，Adam=2倍）

fw_peak   = static_base + fw_activation_peak      # 前向：基座 + 激活内存峰值
bw_peak   = static_base + bw_activation_peak       # 反向：基座 + 激活（梯度已含在图内）
opt_peak  = static_base + grad_bytes + opt_temp    # 优化器：基座 + 梯度 + 临时内存
fwbw_peak = max(fw_peak, bw_peak)                  # FW/BW 联合峰值
true_peak = max(fw_peak, bw_peak, opt_peak)        # 训练步总峰值
```

关键设计决策：
- **`bw_peak` 天然包含 saved activations**：BW 图的 placeholder 就是 FW 保存的激活，
  不需要额外加 `saved_act`（否则会双重计算）
- **`grad_bytes` 只在 OPT 阶段额外计入**：BW 图的输出节点就是梯度 tensor，已含在 `bw_activation_peak` 中（详见 `11-l2-accuracy-improvement.md`）
- **`opt_temp`**：foreach Adam 对所有参数执行 `_foreach_sqrt` → 临时分配 `param_bytes`；
  fused Adam/SGD 无此开销

---

## 三、L1 仿真：Config 公式法

**源文件**: `toolkit/simulation/config_estimator.py`

### 3.1 设计思想

L1 不需要任何图信息，仅根据模型超参数（hidden_size, num_layers, ...）通过公式估算每个组件的内存占用。

### 3.2 参数量估算（第 27-48 行）

```python
# 嵌入层
embed_params = vocab * hidden
if has_position_embedding(config):
    embed_params += config.n_positions * hidden    # GPT-2: learned pos emb
# lm_head
tied = getattr(config, "tie_word_embeddings", True)
lm_head_params = 0 if tied else vocab * hidden

# Attention: Q, K, V, O 四个投影矩阵
kv_proj_scale = n_kv_heads / n_heads               # GQA: K/V 头数可能小于 Q
attn_params = int((2 + 2 * kv_proj_scale) * hidden * hidden)

# MLP
has_gate_proj = not is_gpt2                         # LLaMA: gate+up+down = 3*H*I
mlp_params = 3 * hidden * inter if has_gate_proj else 2 * hidden * inter

# LayerNorm / RMSNorm
ln_params = 4 * hidden if is_gpt2 else 2 * hidden  # GPT-2 LN 有 bias
```

### 3.3 激活量估算（第 53-75 行）

```python
bsh = batch * seq * hidden * elem                   # 基本单位

q_proj_act = bsh                                    # Q 投影输出
attn_weights_act = batch * n_heads * seq * seq * elem  # attention score 矩阵
kv_act = batch * n_kv_heads * seq * head_dim * elem * 2  # K + V
attn_out_act = bsh                                  # attention 输出
mlp_act = batch * seq * inter * elem                # MLP 中间层
if has_gate_proj:
    mlp_act *= 2                                    # gate_proj + up_proj 都被保存
ln_act = 2 * bsh                                    # 2 个 LN/RMSNorm 的输入
residual_act = 2 * bsh                              # 2 个残差连接的输入

per_layer_act = q_proj + attn_weights + kv + attn_out + mlp + ln + residual
total_act = per_layer_act * n_layers + embed_act + logits_act
```

### 3.4 峰值拼装（第 77-96 行）

```python
static_base = param_bytes + optim_bytes

fw_peak  = static_base + activation_bytes
bw_peak  = static_base + grad_bytes + activation_bytes   # 保守：假设所有激活在 BW 开始时存活
opt_peak = static_base + grad_bytes + opt_temp
fwbw_peak = max(fw_peak, bw_peak)
true_peak = max(fw_peak, bw_peak, opt_peak)
```

### 3.5 L1 的局限性

| 局限 | 原因 |
|------|------|
| 不区分重计算策略 | 不知道哪些激活被 min-cut 丢弃 |
| 不追踪生命周期 | 假设"所有激活同时存活"（上界估算） |
| 不处理 view aliasing | 不知道哪些 tensor 共享 storage |
| 不考虑 allocator 对齐 | 无 512B 粒度对齐 |
| 忽略临时计算 buffer | 如 softmax 临时分配、BN running stats 等 |

MRE 约 15-25%，适合快速粗估，不是论文的主要精度数据来源。

---

## 四、L2 仿真：FX 图事件驱动

**源文件**: `toolkit/simulation/graph_estimator.py`

### 4.1 设计思想

L2 使用编译后的 FX 图（带有 FakeTensor meta 信息），按执行顺序模拟每个 op 的
tensor 分配和释放，追踪当前内存和峰值。

核心优势：**能感知不同重计算策略**（因为不同 partition_fn 产生不同的 fw_gm/bw_gm）。

### 4.2 `estimate_graph_peak()` — 单图仿真引擎（第 9-94 行）

这是整个 L2 仿真的核心函数，对**单个** FX 图（FW 或 BW）进行事件驱动内存模拟。

#### 阶段 1：节点分类与大小计算（第 14-33 行）

```python
def estimate_graph_peak(gm, pin_output_inputs=False, align=512):
    nodes = list(gm.graph.nodes)

    # 标记 output 节点的输入（用于 FW 图的 saved activations）
    output_inputs = set()
    if pin_output_inputs:
        for node in nodes:
            if node.op == "output":
                output_inputs.update(node.all_input_nodes)

    node_size = {}
    view_count = 0
    for node in nodes:
        if node.op == "output":
            continue
        # view 节点：不分配新内存，跳过
        if node.op in ("call_function", "call_method") and is_view_node(node):
            view_count += 1
            continue
        # 计算节点字节数并 512B 对齐
        val = node.meta.get("val")
        nb = val_bytes(val) if val is not None else 0
        if nb > 0:
            node_size[node] = (nb + align - 1) & ~(align - 1)   # 512B 对齐
```

三种节点分类：
- **output 节点**: 跳过（不分配内存）
- **view 节点**: 共享 storage，不分配新内存（只增加 view_count）
- **alloc 节点**: 分配新内存，大小向上对齐到 512B

#### 阶段 2：计算最后使用位置（第 35-39 行）

```python
last_use = {}
for index, node in enumerate(nodes):
    for input_node in node.all_input_nodes:
        if input_node in node_size:
            last_use[input_node] = index           # 记录每个 tensor 被最后使用的位置
```

`last_use[X] = i` 表示节点 X 的输出 tensor 在第 i 个节点之后不再被使用，可以释放。

#### 阶段 3：事件驱动模拟（第 49-84 行）

```python
current = 0; peak = 0; live = {}

for index, node in enumerate(nodes):
    # ── 分配事件 ──
    if node in node_size:
        size = node_size[node]
        live[node] = size
        current += size
        peak = max(peak, current)
        timeline.append({"event": "alloc", "bytes": size, "current": current})

    # ── 释放事件 ──
    to_free = [
        live_node for live_node in tuple(live)
        if live_node is not node
        and last_use.get(live_node, -1) <= index
        and live_node not in output_inputs          # ← FW: saved acts 不释放
    ]
    for live_node in to_free:
        size = live.pop(live_node)
        current -= size
        timeline.append({"event": "free", "bytes": size, "current": current})
```

核心机制：
1. **每个 alloc 节点执行时**：`current += size`，更新 `peak`
2. **检查释放条件**：如果某个 live tensor 的所有 consumer 都已执行完（`last_use <= index`），释放它
3. **pin_output_inputs**：FW 图中 output 的输入是 saved activations，不能在 FW 内释放（要穿越到 BW 使用）

#### 关键参数 `pin_output_inputs`

```python
fw_result = estimate_graph_peak(fw_gm, pin_output_inputs=True)   # FW: 保留 saved acts + view-base
bw_result = estimate_graph_peak(bw_gm, pin_output_inputs=True)   # BW: 保留梯度输出 + view-base
```

- **FW 图**: `pin_output_inputs=True` → saved activations 不释放 → 反映"FW 结束时所有 saved acts 仍然活着"
- **BW 图**: `pin_output_inputs=True` → 梯度输出（及其 view-base）不释放 → 反映梯度在 BW 中逐步积累

> **view-base pin 机制**：被 pin 的 view 节点会自动回溯到拥有 storage 的 base 节点并一并 pin。详见 `11-l2-accuracy-improvement.md` §三.1。

### 4.3 `estimate_training_peak()` — 训练步峰值拼装（第 97-179 行）

```python
def estimate_training_peak(fw_gm, bw_gm, model, optimizer_cls=Adam, fused_optimizer=False):
    param_bytes = count_unique_params(model)       # 遍历 model._parameters 去重
    grad_bytes = param_bytes
    optim_bytes = param_bytes * optim_mul           # Adam=2x, SGD=0x

    fw_result = estimate_graph_peak(fw_gm, pin_output_inputs=True)
    bw_result = estimate_graph_peak(bw_gm, pin_output_inputs=True)

    fw_graph_peak = fw_result["peak_bytes"]         # FW 图内激活峰值
    bw_graph_peak = bw_result["peak_bytes"]         # BW 图内激活峰值（含梯度）

    static_base = param_bytes + optim_bytes

    fw_peak  = static_base + fw_graph_peak
    bw_peak  = static_base + bw_graph_peak           # 梯度已含在 bw_graph_peak 中
    opt_peak = static_base + grad_bytes + opt_temp
    fwbw_peak = max(fw_peak, bw_peak)
    true_peak = max(fw_peak, bw_peak, opt_peak)
```

**为什么 `bw_peak` 不需要再加 `saved_act`？**

BW 图的 `placeholder` 节点就是 saved activations。在 `estimate_graph_peak(bw_gm)` 遍历
BW 图时，这些 placeholder 在图开头就被"分配"（作为输入），它们的内存已经包含在 `bw_graph_peak` 中。
如果再加一次 `saved_act`，就会双重计算。

这是本项目早期的 Bug B2，修复后 MRE 从 ~50% 降到 6.9%。

---

## 五、View 检测：Storage Aliasing

**源文件**: `toolkit/utils/view_ops.py`

### 5.1 问题

PyTorch 的 view 操作（`view`, `reshape`, `permute`, `slice`, `t`, `unsqueeze`, ...）
不分配新 storage，只改变 tensor 的元信息（shape, stride, offset）。
如果仿真把 view 的输出当作新分配，会严重高估内存。

### 5.2 方案：FakeTensor `_cdata` 比较

```python
def _storage_cdata(tensor):
    storage = tensor.untyped_storage()
    return getattr(storage, "_cdata", None)

def is_view_node(node):
    val = node.meta.get("val")
    val_cdata = _storage_cdata(val)                  # 输出 tensor 的 storage id

    for input_node in node.all_input_nodes:
        input_val = input_node.meta.get("val")
        for tensor in _iter_tensor_like_values(input_val):
            if _storage_cdata(tensor) == val_cdata:  # 和任一输入共享 storage
                return True                          # → 是 view
    return False
```

**原理**：在 FakeTensorMode 下，`untyped_storage()._cdata` 正确反映 storage 共享关系。
如果输出 tensor 和某个输入 tensor 的 `_cdata` 相同，说明它们共享底层 storage → 这是一个 view op。

**为什么不用硬编码 op 名单？**
- `operator.getitem` 在 FX 图中也是 view（解包 tuple），但它不是 ATen op → 名单法遗漏
- GPT-2 实测：22.2 MB 被遗漏（8.3% 误差），改用 `_cdata` 后修复

### 5.3 在仿真中的作用

```python
# graph_estimator.py 第 27-29 行
if node.op in ("call_function", "call_method") and is_view_node(node):
    view_count += 1
    continue                     # 跳过 → 不计入 node_size → 不分配内存
```

View 节点被完全跳过，既不分配也不释放，因为它们的 storage 已被原始 tensor 计入。

---

## 六、512B 对齐

**源文件**: `toolkit/simulation/graph_estimator.py` 第 33 行

```python
node_size[node] = (nb + align - 1) & ~(align - 1)    # align = 512
```

**原理**: PyTorch 的 CUDA Caching Allocator 默认按 512B 对齐分配。一个 64 字节的 tensor
实际占用 512B。不对齐会系统性低估小 tensor 的内存占用。

实测影响：对 LLaMA-6L 约 2-3% 的额外内存，小模型更显著。

---

## 七、`val_bytes()` — 节点大小估算

**源文件**: `toolkit/utils/tensor_utils.py`

```python
def val_bytes(val, fallback_numel=4096):
    if isinstance(val, torch.Tensor):
        numel = hint_int(val.numel(), fallback=fallback_numel)
        return int(numel) * val.element_size()

    if isinstance(val, (tuple, list)):
        return sum(val_bytes(v) for v in val)            # 递归处理 tuple/list

    if isinstance(val, (torch.SymInt, torch.SymFloat, torch.SymBool)):
        return 0                                         # 标量不占 GPU 内存

    return 0
```

与 PyTorch 官方 `_size_of()` 保持一致的设计：
- **`hint_int(fallback=4096)`**：处理 SymInt（动态 shape 场景），无法求值时用 4096 作为 fallback
- **tuple/list 递归**：处理如 `torch.split` 返回多个 tensor 的情况
- **SymInt/Float/Bool = 0**：标量不占 GPU tensor 内存

---

## 八、`count_unique_params()` — 参数量精确统计

**源文件**: `toolkit/utils/tensor_utils.py`

```python
def count_unique_params(model):
    seen_ptrs = set()
    total_bytes = 0
    for module in model.modules():
        for param in module._parameters.values():
            if param is None:
                continue
            ptr = param.data_ptr()
            if ptr in seen_ptrs:
                continue                              # 去重：tied embeddings
            seen_ptrs.add(ptr)
            total_bytes += param.numel() * param.element_size()
    return total_bytes
```

关键：用 `data_ptr()` 去重，处理 weight tying（如 LLaMA 的 `lm_head` 和 `embed_tokens` 共享权重）。

---

## 九、图捕获：仿真的输入从哪来

**源文件**: `toolkit/capture/aot_capture.py`

```python
def capture_graphs(model, sample_input_ids, loss_fn, partition_fn=default_partition, dynamic=True):
    captured = {}

    def _backend(gm, example_inputs):
        def fw_compiler(fw_gm, _inputs):
            captured["fw"] = copy.deepcopy(fw_gm)     # 深拷贝保存 FW 图
            return make_boxed_func(fw_gm.forward)
        def bw_compiler(bw_gm, _inputs):
            captured["bw"] = copy.deepcopy(bw_gm)     # 深拷贝保存 BW 图
            return make_boxed_func(bw_gm.forward)
        return aot_module_simplified(gm, example_inputs,
                                     fw_compiler=fw_compiler,
                                     bw_compiler=bw_compiler,
                                     partition_fn=partition_fn)

    compiled = torch.compile(model, backend=_backend, dynamic=dynamic)
    output = compiled(sample_input_ids, **model_kwargs)
    loss = loss_fn(output)
    loss.backward()                                    # 触发 BW 图生成

    return captured["fw"], captured["bw"]
```

**流程**：
1. `torch.compile` + 自定义 backend
2. backend 内部使用 `aot_module_simplified` → AOT Autograd 生成 joint graph
3. `partition_fn` 决定 FW/BW 的分割方式（`default_partition` 保存全部 / `min_cut` 优化）
4. fw_compiler / bw_compiler 截获编译后的 FW/BW GraphModule
5. 深拷贝保存（因为原始 gm 可能在后续编译中被修改）

**不同策略的图差异**：

| partition_fn | FW output | BW placeholder | saved_act |
|--------------|-----------|----------------|-----------|
| `default_partition` | 用户输出 + 所有中间激活 | 所有中间激活 + tangents | 最大 |
| `min_cut_partition` | 用户输出 + min-cut 选中的激活 | min-cut 选中的激活 + tangents | 更小 |

这就是 L2 仿真能感知重计算策略的原因：不同策略产生不同的 fw_gm/bw_gm。

---

## 十、运行时验证：Ground Truth

**源文件**: `toolkit/profiler/step_profiler.py`

### 10.1 `measure_phased()` — 分阶段内存测量（第 116-200 行）

```python
def measure_phased(name, forward_fn, optimizer, *, repeats=5, warmup=3, device="cuda"):
    for _ in range(warmup):                            # warmup: JIT 编译、CUDA 初始化
        ...

    for _ in range(repeats):
        torch.cuda.empty_cache()
        optimizer.zero_grad(set_to_none=True)
        base = torch.cuda.memory_allocated(device)

        # ── FW 阶段 ──
        torch.cuda.reset_peak_memory_stats(device)
        loss = forward_fn()
        torch.cuda.synchronize()
        fw_peak = torch.cuda.max_memory_allocated(device)
        after_fw = torch.cuda.memory_allocated(device)

        # ── BW 阶段 ──
        torch.cuda.reset_peak_memory_stats(device)
        loss.backward()
        torch.cuda.synchronize()
        bw_peak = torch.cuda.max_memory_allocated(device)
        after_bw = torch.cuda.memory_allocated(device)

        # ── OPT 阶段 ──
        torch.cuda.reset_peak_memory_stats(device)
        optimizer.step()
        torch.cuda.synchronize()
        opt_peak = torch.cuda.max_memory_allocated(device)

    # IQR mean 聚合
    return PhaseResult(
        fw_peak=iqr_mean([r.fw_peak for r in results]),
        bw_peak=iqr_mean([r.bw_peak for r in results]),
        opt_peak=iqr_mean([r.opt_peak for r in results]),
        ...
    )
```

关键设计：
- **每个阶段前 `reset_peak_memory_stats`**：独立测量三阶段峰值
- **`zero_grad(set_to_none=True)`**：释放梯度，base = param + optim states
- **IQR mean**：去掉最高/最低 25% 后取均值，消除异常值

### 10.2 `PhaseResult` 数据结构

```python
@dataclass
class PhaseResult:
    fw_peak: int          # FW 阶段绝对峰值（含 base）
    bw_peak: int          # BW 阶段绝对峰值（含 base + grad）
    opt_peak: int         # OPT 阶段绝对峰值
    after_fw: int         # FW 结束后的当前内存
    after_bw: int         # BW 结束后的当前内存
    after_opt: int        # OPT 结束后的当前内存
    base_allocated: int   # 基座内存（param + optim states）
    overall_peak: int     # max(fw, bw, opt)
    fwbw_peak: int        # max(fw, bw)
    peak_phase: str       # "FW" / "BW" / "OPT"
    step_ms: float        # 整步时间
    grad_bytes: int       # after_bw - after_fw（梯度占用）
```

### 10.3 Validator — 仿真精度验证

**源文件**: `toolkit/profiler/validator.py`

```python
def validate(static_result, runtime_result, run_mode="compiled"):
    static_peak = static_result["true_peak"]
    runtime_peak = runtime_result.overall_peak
    mre = abs(static_peak - runtime_peak) / runtime_peak
    direction = "over" if static_peak > runtime_peak else "under"
    return ValidationResult(mre_allocated=mre, direction=direction, ...)
```

MRE (Mean Relative Error) = `|static - runtime| / runtime`，
还支持分阶段 MRE 对比（fw_peak, bw_peak, opt_peak 各自的误差）。

---

## 十一、L2.5 融合感知仿真

**源文件**: `toolkit/simulation/fusion_groups.py`, `toolkit/simulation/fusion_ops.py`, `toolkit/simulation/graph_estimator.py`

### 11.1 背景

Inductor 后端将连续的 pointwise/reduction 算子融合成单个 Triton kernel，内部中间张量不在 GPU 全局内存中物化。L2 不建模 fusion → inductor MRE 28-38%。L2.5 通过近似融合组识别来降低这一误差。

### 11.2 算子分类 (`fusion_ops.py`)

```python
EXTERN_OPS = {aten.mm, aten.bmm, aten.addmm, ...}  # 调用 CUBLAS/cuDNN，不参与融合

def is_fusable_op(node) -> bool:
    """非 extern 的 call_function 节点视为可融合。"""
```

### 11.3 融合组识别 (`fusion_groups.py`)

```python
def identify_fusion_groups(gm) -> list[FusionGroup]:
    """贪心拓扑扫描：连续的 fusable ops 合并为一组。
    组的边界在 extern op 或 graph 边界处切断。"""
```

每个 FusionGroup 包含：
- `nodes`: 组内所有节点
- `internal_nodes`: 仅组内消费的中间节点（分配设为 0）
- `boundary_nodes`: 组的输入/输出节点（保留正常分配）

### 11.4 融合感知估算

```python
estimate_graph_peak(gm, fusion_aware=True, optimize_order=True)
```

- `fusion_aware=True`: 识别融合组，internal 节点分配为 0
- `optimize_order=True`: 贪心调度 — 优先执行释放内存最多的节点，最小化峰值

L2.5 将 inductor MRE 从 28-38% 降低到 8-12%。

---

## 十二、L3 Scheduler 仿真

**源文件**: `toolkit/capture/inductor_capture.py`, `toolkit/simulation/graph_estimator.py`

### 12.1 背景

L2.5 仍是近似融合。L3 直接复用 Inductor 编译器的 `Scheduler.estimate_peak_memory()` 方法，获得编译器视角的精确峰值估算。

### 12.2 捕获机制

```python
def capture_inductor_graphs(model, input_ids, loss_fn, *, budget=None):
    # 1. monkey-patch Scheduler.__init__，在其中调用 self.estimate_peak_memory()
    # 2. 使用 compile_fx(inner_compile=hook) 截获 post-grad FW/BW GraphModule
    # 3. 返回 {fw_gm, bw_gm, sched_fw_peak, sched_bw_peak}
```

### 12.3 三层封装

```python
def estimate_inductor_training_peak(capture_result, model, ...):
    # L2: estimate_training_peak(fw_gm, bw_gm, model)
    # L2.5: estimate_graph_peak(fw_gm, fusion_aware=True) + min(sched_bw, fusion_bw)
    # L3: static_base + max(sched_fw_peak, sched_bw_peak + grad_bytes)
    # 返回包含三层结果的 dict
```

L3 将 inductor MRE 从 8-12% 进一步降低到 5-7%。

---

## 十三、误差来源分析

### 13.1 已建模（各层覆盖）

| 因素 | L2 | L2.5 | L3 |
|------|----|----- |----|
| Op 级激活生命周期 | ✅ 事件驱动 | ✅ 事件驱动 | ✅ Scheduler |
| View aliasing | ✅ `_cdata` | ✅ `_cdata` | ✅ Scheduler |
| SymInt 动态 shape | ✅ `hint_int` | ✅ `hint_int` | ✅ |
| 512B 对齐 | ✅ | ✅ | ✅ |
| Saved activations | ✅ pin | ✅ pin | ✅ |
| Optimizer 临时内存 | ✅ foreach Adam | ✅ | ✅ |
| Weight tying | ✅ `data_ptr()` | ✅ | ✅ |
| Kernel fusion | ❌ | ✅ 近似 | ✅ 精确 |
| 执行顺序优化 | ❌ | ✅ 贪心 | ✅ Scheduler |

### 13.2 未建模（贡献残余 MRE）

| 因素 | 影响 | 备注 |
|------|------|------|
| CUDA context / driver 内存 | 16-19 MB 固定偏移 | 放大模型后 <0.1% |
| Allocator 碎片 | reserved >> allocated | L3 不建模 reserved |
| Kernel 临时 buffer | cuBLAS workspace 等 | 通常 <1% |
| dark memory | ~16-19 MB | 见下 |

### 13.3 dark memory 量化

```
runtime_base - formula_base = 16-19 MB
```

CUDA context + allocator metadata + 模型内部 buffer。
小模型（~30M params）约 7-10% 峰值；放大版模型（~870M params）<0.1%，可忽略。

---

## 十四、完整调用链

```
# ====== AOT 图捕获 (L2) ======
capture_graphs(model, input_ids, loss_fn, partition_fn=min_cut)
    → (fw_gm, bw_gm)

# ====== Inductor 图捕获 (L2.5 + L3) ======
capture_inductor_graphs(model, input_ids, loss_fn, budget=1.0)
    → {fw_gm, bw_gm, sched_fw_peak, sched_bw_peak}

# ====== L2 仿真 ======
estimate_training_peak(fw_gm, bw_gm, model)
    ├── estimate_graph_peak(fw_gm, pin_output_inputs=True)  → fw_graph_peak
    ├── estimate_graph_peak(bw_gm, pin_output_inputs=True)  → bw_graph_peak
    └── 四峰值拼装 → {fw_peak, bw_peak, opt_peak, true_peak, peak_phase}

# ====== L2 + L2.5 + L3 三层仿真 ======
estimate_inductor_training_peak(capture_result, model)
    ├── L2: estimate_training_peak(fw_gm, bw_gm, model)
    ├── L2.5: estimate_graph_peak(fw_gm, fusion_aware=True) + min(sched_bw, fusion_bw)
    ├── L3: static_base + sched_peaks
    └── → {l2_*, l25_*, l3_*}

# ====== 运行时验证 ======
measure_phased(forward_fn, optimizer)
    → PhaseResult {fw_peak, bw_peak, opt_peak, peak_phase, ...}

validate(static_result, runtime_result)
    → ValidationResult {mre, direction, ...}
```

---

## 十五、源码文件索引

| 文件 | 内容 |
|------|------|
| `toolkit/simulation/config_estimator.py` | L1 配置公式法 (`estimate_from_config`) |
| `toolkit/simulation/graph_estimator.py` | L2/L2.5/L3 仿真引擎 (`estimate_graph_peak`, `estimate_training_peak`, `estimate_inductor_training_peak`) |
| `toolkit/simulation/fusion_groups.py` | L2.5 融合组识别 (`identify_fusion_groups`) |
| `toolkit/simulation/fusion_ops.py` | L2.5 算子分类 (`EXTERN_OPS`, `is_fusable_op`) |
| `toolkit/utils/view_ops.py` | View 检测（`_cdata` storage aliasing） |
| `toolkit/utils/tensor_utils.py` | `val_bytes()` + `count_unique_params()` |
| `toolkit/capture/aot_capture.py` | AOT FW/BW 图捕获 |
| `toolkit/capture/inductor_capture.py` | Inductor 双层捕获 + L3 Scheduler hook |
| `toolkit/capture/analysis.py` | 图分析（node stats, output bytes） |
| `toolkit/profiler/step_profiler.py` | `measure_step()` + `measure_phased()` |
| `toolkit/profiler/validator.py` | `validate()` + `analyze_error_sources()` |

---

## 参考

- `01-architecture.md` §七：四层仿真概述
- `13-l2.5-fusion-aware-design.md`：L2.5 融合感知设计详解
- `12-inductor-memory-analysis.md`：Inductor 后端内存分析 + L3 可行性
- `15-experiment-outputs.md`：实验数据说明（ex_sim_accuracy.csv 含 L2/L2.5/L3 MRE）
- `04-dev-log.md`：Bug B2（saved_act 双重计算）修复记录
- PyTorch: `torch.cuda.memory_stats()` — CUDACachingAllocator 统计接口
