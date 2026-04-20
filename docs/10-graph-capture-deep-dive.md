# 10 — 图捕获机制源码深度解析

> **文档定位**: 对本项目图捕获层（Pillar 2）的全链路源码解析，
> 涵盖 `torch.compile` 编译流水线、Dynamo 字节码追踪、AOTAutograd 联合图生成、
> FakeTensor 元信息传播、partition_fn 的 FW/BW 拆分机制、以及本项目
> `capture_graphs()` 如何截获编译产物。
>
> **前置阅读**: `01-architecture.md` §四（图捕获概述）

---

## 一、概述：图捕获在项目中的角色

```
┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
│ Pillar 1   │    │ Pillar 2   │    │ Pillar 3   │    │ Pillar 4   │
│ 策略注入    │ →  │ 图捕获     │ →  │ 静态仿真    │ →  │ 运行时验证  │
└────────────┘    └────────────┘    └────────────┘    └────────────┘
                   ↑ 本文
```

图捕获的目标：把 `model.forward()` + `loss.backward()` 的完整训练步
变成两张带有 FakeTensor 元信息的 FX 图（`fw_gm` / `bw_gm`），
供下游静态仿真在不执行真实 GPU 计算的情况下分析内存。

关键产出：
- **`fw_gm`**: 前向图，`output` = 用户输出 + saved activations
- **`bw_gm`**: 反向图，`placeholder` = saved activations + tangent inputs
- **每个节点的 `meta['val']`**: FakeTensor，携带 shape/dtype/stride/storage 信息

---

## 二、`torch.compile` 编译流水线总览

```
用户代码: compiled = torch.compile(model, backend=my_backend)
         output = compiled(input_ids)

                    ┌───────────────────────────┐
                    │     ① TorchDynamo          │
                    │  Python 字节码 → FX Graph   │
                    │  (torch._dynamo)            │
                    └───────────┬───────────────┘
                                │ FX GraphModule (高层 op)
                                ▼
                    ┌───────────────────────────┐
                    │     ② Backend Dispatch     │
                    │  用户自定义 backend 函数     │
                    │  backend(gm, example_inputs)│
                    └───────────┬───────────────┘
                                │
                                ▼
                    ┌───────────────────────────┐
                    │     ③ AOTAutograd          │
                    │  aot_module_simplified()    │
                    │  - 联合追踪 FW+BW           │
                    │  - partition_fn 拆分        │
                    │  - fw_compiler / bw_compiler│
                    └───────────┬───────────────┘
                                │ fw_gm, bw_gm
                                ▼
                    ┌───────────────────────────┐
                    │     ④ 后端编译              │
                    │  Inductor / aot_eager /     │
                    │  用户自定义 compiler          │
                    └───────────────────────────┘
```

### 2.1 阶段 ① TorchDynamo

**源文件**: `torch/_dynamo/`

Dynamo 是一个 Python 字节码级别的追踪器：
1. 拦截 Python frame 的执行（通过 `_TorchDynamoContext`）
2. 模拟执行字节码，将 PyTorch op 调用记录到 FX Graph
3. 遇到无法追踪的 Python 语句时产生 "graph break"
4. 输出：一个 `fx.GraphModule`，包含高层 op（如 `torch.nn.functional.linear`）

### 2.2 阶段 ② Backend Dispatch

```python
# torch.compile 内部
compiled = torch.compile(model, backend=my_backend)
# ↓ 等价于
# Dynamo 追踪后调用: my_backend(gm, example_inputs)
```

backend 是一个 `Callable[[fx.GraphModule, List[Tensor]], Callable]`。
Dynamo 将追踪得到的 `gm` 和示例输入传给 backend，backend 返回一个可调用对象。

### 2.3 阶段 ③ AOTAutograd

**源文件**: `torch/_functorch/aot_autograd.py`

AOTAutograd 是在 **编译时** 执行自动微分的核心模块：
1. 在 FakeTensorMode 下执行 forward，收集元信息
2. 通过 `create_joint()` + `make_fx()` 追踪 FW+BW 联合图
3. 通过 `partition_fn` 将联合图拆分为独立的 FW/BW 两张图
4. 分别调用 `fw_compiler` / `bw_compiler` 编译两张图

---

## 三、AOTAutograd 详解

### 3.1 `aot_module_simplified()` — 入口函数

**源文件**: `torch/_functorch/aot_autograd.py` 第 1007-1191 行

```python
def aot_module_simplified(
    mod: nn.Module,
    args,
    fw_compiler,          # 前向图编译器
    bw_compiler=None,     # 反向图编译器（默认同 fw_compiler）
    partition_fn=default_partition,   # 联合图拆分策略
    decompositions=None,
    ...
) -> nn.Module:
```

**流程**：
1. 展平参数 `params_flat = [*named_parameters, *named_buffers]`
2. 拼接完整输入 `full_args = params_flat + runtime_args`
3. 创建 FakeTensor：`fake_flat_args = process_inputs(full_args, ...)`
4. 构建 AOTConfig：封装所有配置（fw/bw compiler, partition_fn, decompositions）
5. 调用 `create_aot_dispatcher_function()` 执行核心编译
6. 返回包装后的 `forward` 函数

### 3.2 `_create_aot_dispatcher_function()` — 核心调度

**源文件**: `torch/_functorch/aot_autograd.py` 第 585-836 行

```python
def _create_aot_dispatcher_function(flat_fn, fake_flat_args, aot_config, fake_mode, shape_env):
    # ① 在 FakeTensorMode 下执行 forward，收集元信息
    fw_metadata = run_functionalized_fw_and_collect_metadata(flat_fn, ...)(*fake_flat_args)

    # ② 根据是否需要 autograd 选择调度器
    if needs_autograd:
        compiler_fn = aot_dispatch_autograd      # 训练模式 → 生成联合图
    else:
        compiler_fn = aot_dispatch_base          # 推理模式 → 只有 FW

    # ③ 执行编译
    compiled_fn, fw_metadata = compiler_fn(flat_fn, fake_flat_args, aot_config, ...)
    return compiled_fn, fw_metadata
```

### 3.3 `aot_dispatch_autograd_graph()` — 联合图生成

**源文件**: `torch/_functorch/_aot_autograd/dispatch_and_compile_graph.py` 第 251-338 行

这是生成联合 FW+BW 图的核心函数：

```python
def aot_dispatch_autograd_graph(flat_fn, flat_args, aot_config, *, fw_metadata):
    # ① 构造联合输入 = (primals, tangents)
    joint_inputs = (flat_args, fw_metadata.traced_tangents)

    # ② 准备联合函数（forward + torch.autograd.grad）
    fn_prepared = fn_prepped_for_autograd(flat_fn, fw_metadata)
    joint_fn = create_joint(fn_prepared, aot_config=aot_config)

    # ③ 函数化（消除 mutation）
    joint_fn, updated_inputs = create_functionalized_fn(joint_fn, joint_inputs, ...)

    # ④ 通过 make_fx 追踪为 FX 图
    fx_g = _create_graph(joint_fn, updated_inputs, aot_config=aot_config)

    # ⑤ 清理：消除死代码，拷贝 FW 元信息到 BW 节点
    fx_g.graph.eliminate_dead_code()
    copy_fwd_metadata_to_bw_nodes(fx_g)
    fx_g.recompile()

    return fx_g, saved_inputs, maybe_subclass_meta
```

### 3.4 `create_joint()` — 联合 FW+BW 函数

**源文件**: `torch/_functorch/_aot_autograd/traced_function_transforms.py` 第 192-282 行

```python
def create_joint(fn, *, aot_config):
    def inner_fn(primals, tangents):
        # ① 执行 forward
        outs, tangent_mask = fn(*primals)

        # ② 找出需要梯度的输入和输出
        grad_primals = [p for p in primals if p.requires_grad]
        needed_outs = [o for o in outs if o.requires_grad]

        # ③ 调用 torch.autograd.grad 计算反向
        backward_out = torch.autograd.grad(
            needed_outs,
            grad_primals,
            grad_outputs=needed_tangents,
            allow_unused=True,
        )

        # ④ 返回 (forward_outputs, backward_outputs)
        return outs, backward_out

    return inner_fn
```

**关键**：`create_joint` 在一个函数中同时包含了 FW 和 BW，
当被 `make_fx` 追踪时，会生成一个包含完整 FW+BW 的联合 FX 图。

### 3.5 `_create_graph()` — FX 追踪

**源文件**: `torch/_functorch/_aot_autograd/dispatch_and_compile_graph.py` 第 46-62 行

```python
def _create_graph(f, args, *, aot_config):
    with enable_python_dispatcher(), FunctionalTensorMode(...):
        fx_g = make_fx(
            f,
            decomposition_table=aot_config.decompositions,
            record_module_stack=True,
        )(*args)
    return fx_g
```

`make_fx` 通过 `torch.fx.Proxy` 机制追踪函数 `f` 的执行过程，
将所有 op 调用记录为 FX 图节点。在 FakeTensorMode 下执行，
**不进行真实计算**，只传播 shape/dtype 元信息。

---

## 四、FakeTensor 与 `meta['val']`

### 4.1 FakeTensor 是什么

FakeTensor 是不持有真实数据的 tensor "壳"，只保留元信息：
- **shape** (可以是 SymInt → 支持动态 shape)
- **dtype**
- **device**
- **stride**
- **storage 信息**（`_cdata`，用于 view 检测）

### 4.2 FakeTensorMode 的工作原理

```python
class FakeTensorMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args, kwargs):
        # 不执行真实 CUDA 计算
        # 只根据 op 的 shape 规则推导输出的元信息
        return self.dispatch(func, types, args, kwargs)
```

当 `make_fx` 在 FakeTensorMode 下追踪时：
1. 每个 op 被调用 → FakeTensorMode 拦截
2. 根据 op 的 meta kernel 推导输出 shape/dtype
3. 生成 FakeTensor 作为输出
4. 这个 FakeTensor 被存储在 FX 节点的 `meta['val']` 中

### 4.3 `meta['val']` 对仿真的意义

```python
node.meta['val']  # → FakeTensor, 携带:
                  #   .shape  → 用于 val_bytes() 计算内存
                  #   .numel() → 可能是 SymInt
                  #   .element_size() → 由 dtype 决定
                  #   .untyped_storage()._cdata → 用于 view 检测
```

这就是 L2 仿真的信息来源——不需要真实执行，只需要 FakeTensor 的元信息。

### 4.4 dynamic=True 的影响

```python
torch.compile(model, backend=_backend, dynamic=True)
```

- `dynamic=True`：Dynamo 使用 `ShapeEnv` 追踪符号化 shape（SymInt）
  → 图更通用，但 `val.numel()` 可能返回 SymInt
  → 需要 `hint_int(fallback=4096)` 处理
- `dynamic=False`：shape 被具体化（concrete int）
  → 图只对特定输入 shape 有效

本项目始终使用 `dynamic=True`，配合 `hint_int` 处理 SymInt。

---

## 五、partition_fn — 联合图拆分

### 5.1 联合图的结构

```
Joint Graph:
  placeholder: primals (params + inputs)
  placeholder: tangents (grad outputs)

  ... forward ops ...
  ... backward ops (autograd.grad 展开) ...

  output: (forward_outputs..., backward_outputs...)
             ↑ num_fwd_outputs ↑
```

`output` 的前 `num_fwd_outputs` 个是 FW 输出，其余是 BW 输出（梯度）。

### 5.2 `default_partition()` — 默认分割

**源文件**: `torch/_functorch/partitioners.py` 第 389-472 行

```python
def default_partition(joint_module, _joint_inputs, *, num_fwd_outputs):
    # 如果图中有可重计算的 op（AC/SAC 标记），自动使用 min-cut
    if has_recomputable_ops(joint_module):
        return min_cut_rematerialization_partition(...)

    # ① 提取 FW-only 子图
    forward_only_graph = _extract_graph_with_inputs_outputs(
        joint_module.graph, inputs=primal_inputs, outputs=fwd_outputs, "forward"
    )

    # ② 找出需要跨越 FW→BW 边界保存的 tensor（saved activations）
    for node in joint_module.graph.nodes:
        if node in forward_only_graph:
            backward_usages = [n for n in node.users if n not in forward_only_graph]
            if backward_usages:
                saved_values.append(node)  # 这个 FW 节点被 BW 使用 → 必须保存

    # ③ 生成最终的 FW/BW 模块
    return _extract_fwd_bwd_modules(joint_module, saved_values, ...)
```

**`default_partition` 的策略**：保存所有被 BW 使用的 FW 中间值 → saved activations 最大。

### 5.3 `_extract_fwd_bwd_modules()` — FW/BW 拆分

**源文件**: `torch/_functorch/partitioners.py` 第 290-386 行

```python
def _extract_fwd_bwd_modules(joint_module, saved_values, saved_sym_nodes, *, num_fwd_outputs):
    fwd_outputs, bwd_outputs = _extract_fwd_bwd_outputs(joint_module, num_fwd_outputs)

    # FW 图: inputs = primals, outputs = fwd_outputs + saved_values
    fwd_graph = _extract_graph_with_inputs_outputs(
        joint_module.graph,
        inputs  = primal_inputs + fwd_seed_offset_inputs,
        outputs = fwd_outputs + saved_values + saved_sym_nodes,
        "forward",
    )

    # BW 图: inputs = saved_values + tangents, outputs = bwd_outputs
    bwd_graph = _extract_graph_with_inputs_outputs(
        joint_module.graph,
        inputs  = saved_sym_nodes + saved_values + tangent_inputs + ...,
        outputs = bwd_outputs,
        "backward",
    )

    return fx.GraphModule(fwd_graph), fx.GraphModule(bwd_graph)
```

**核心逻辑**：
- FW 图的 **output** 被扩展为 `用户输出 + saved activations`
- BW 图的 **placeholder** 被扩展为 `saved activations + tangent inputs`
- saved activations 从 FW output 流向 BW placeholder — 这就是 saved tensors

### 5.4 `_extract_graph_with_inputs_outputs()` — 子图提取

**源文件**: `torch/_functorch/partitioners.py` 第 158-226 行

```python
def _extract_graph_with_inputs_outputs(joint_graph, inputs, outputs, subgraph):
    new_graph = fx.Graph()
    env = {}

    # ① 指定的 inputs 变为新图的 placeholder
    for node in inputs:
        new_node = new_graph.placeholder(node.name)
        new_node.meta = node.meta          # 保留 FakeTensor 元信息！
        env[node] = new_node

    # ② 遍历原图，拷贝合法节点
    for node in joint_graph.nodes:
        if node in env:
            continue
        elif node.op == "placeholder":
            env[node] = InvalidNode        # 原始 placeholder 不再需要
        elif node.op == "call_function":
            # 如果任一输入是 InvalidNode，此节点也无效
            if any(isinstance(env[x], InvalidNodeBase) for x in node.args if isinstance(x, fx.Node)):
                env[node] = InvalidNode
                continue
            env[node] = new_graph.node_copy(node, lambda x: env[x])  # 拷贝（保留 meta）

    # ③ 设置 output
    new_graph.output(tuple(env[x] for x in outputs))
    new_graph.eliminate_dead_code()
    return new_graph
```

**关键**: `node_copy` 会保留 `node.meta`，包括 `meta['val']`（FakeTensor）。
这就是为什么拆分后的 fw_gm/bw_gm 中每个节点仍然有完整的形状信息。

### 5.5 `min_cut_rematerialization_partition` — 最优拆分

**源文件**: `torch/_functorch/partitioners.py` 第 1703-1878 行

当使用 `min_cut` 作为 partition_fn 时：
- 使用 networkx 的 `minimum_cut` 算法在流网络上求解最小割
- 目标：**最小化 saved activations 的内存**（在内存与重计算之间取最优）
- AC/SAC 的 `recompute` 标签影响边的 capacity → 控制哪些节点可以被重计算

详细原理参见 `06-activation-memory-budget.md` 和 `07-ac-sac-deep-dive.md`。

---

## 六、本项目的 `capture_graphs()` 实现

**源文件**: `toolkit/capture/aot_capture.py`

### 6.1 核心代码

```python
def capture_graphs(model, sample_input_ids, loss_fn,
                   partition_fn=default_partition, dynamic=True):
    captured = {}

    def _backend(gm, example_inputs):
        def fw_compiler(fw_gm, _inputs):
            captured["fw"] = copy.deepcopy(fw_gm)       # ← 截获 FW 图
            return make_boxed_func(fw_gm.forward)

        def bw_compiler(bw_gm, _inputs):
            captured["bw"] = copy.deepcopy(bw_gm)       # ← 截获 BW 图
            return make_boxed_func(bw_gm.forward)

        return aot_module_simplified(
            gm, example_inputs,
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=partition_fn,                    # ← 策略注入点
        )

    torch._dynamo.reset()                                # 清理缓存
    try:
        compiled = torch.compile(model, backend=_backend, dynamic=dynamic)
        output = compiled(sample_input_ids, **model_kwargs)
        loss = loss_fn(output)
        loss.backward()                                  # ← 触发 BW 图生成

        return captured["fw"], captured["bw"]
    finally:
        torch._dynamo.reset()                            # 确保清理
```

### 6.2 设计决策详解

| 决策 | 原因 |
|------|------|
| **自定义 backend** | 在 AOTAutograd 内部截获 FW/BW 图，而非编译后产物 |
| **`copy.deepcopy(fw_gm)`** | 后续编译可能修改 gm，深拷贝保存快照 |
| **`make_boxed_func`** | AOTAutograd 要求返回 boxed calling convention 函数 |
| **`dynamic=True`** | 使用符号化 shape，图更通用 |
| **`torch._dynamo.reset()`** | 清理 Dynamo 缓存，避免与后续编译冲突 |
| **`finally` 块** | B15 修复：确保异常时也清理 Dynamo 状态 |
| **必须执行 `backward()`** | BW 图是**惰性生成**的，只有 backward 时才触发 bw_compiler |

### 6.3 不同策略产生不同的图

```python
# 策略 A: default_partition — 保存所有中间值
fw_gm_default, bw_gm_default = capture_graphs(model, ids, loss_fn,
    partition_fn=default_partition)

# 策略 B: min_cut — 优化 saved activations
fw_gm_mincut, bw_gm_mincut = capture_graphs(model, ids, loss_fn,
    partition_fn=min_cut_rematerialization_partition)
```

| partition_fn | FW output 数量 | saved activations | BW placeholder 数量 |
|--------------|---------------|-------------------|-------------------|
| `default_partition` | 多（所有被 BW 使用的中间值） | 最大 | 多 |
| `min_cut_partition` | 少（只保存 min-cut 选中的） | 较小 | 少 |

这就是 L2 仿真能感知重计算策略的根本原因：**不同 partition_fn 产生结构不同的 fw_gm/bw_gm**。

---

## 七、图分析工具

**源文件**: `toolkit/capture/analysis.py`

### 7.1 `graph_stats()` — 图统计

```python
def graph_stats(gm):
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            n_placeholder += 1
        elif node.op in ("call_function", "call_method"):
            if is_view_node(node):
                n_view += 1
                total_view_bytes += val_bytes(val)
            else:
                n_alloc += 1
                total_alloc_bytes += val_bytes(val)

    return {"n_total", "n_placeholder", "n_view", "n_alloc",
            "total_alloc_bytes", "total_view_bytes",
            "symint_ok", "symint_fail"}
```

统计每张图的节点分类：placeholder（输入）、view（不分配）、alloc（分配新内存）。

### 7.2 `count_fw_output_bytes()` — saved activations 大小

```python
def count_fw_output_bytes(fw_gm):
    for node in fw_gm.graph.nodes:
        if node.op == "output":
            for input_node in node.all_input_nodes:
                total += val_bytes(input_node.meta.get("val"))
    return total
```

FW 图的 output 节点的输入 = 用户输出 + saved activations。
这个值用于 L2 仿真结果中的 `saved_act_bytes` 字段。

### 7.3 `analyze_graph()` — 逐节点分析

```python
def analyze_graph(gm):
    nodes = []
    for node in gm.graph.nodes:
        nodes.append({
            "name": node.name,
            "op": node.op,
            "target": str(node.target),
            "is_view": is_view_node(node),
            "bytes": val_bytes(val),
        })
    return {"num_nodes": len(nodes), "nodes": nodes}
```

返回每个节点的详细信息，用于人工分析和调试。

---

## 八、IR 导出

> **注意**: `ir_saver.py` 已在 v6.1 代码清理中删除。以下保留作为设计参考。

**原源文件**: `toolkit/capture/ir_saver.py`

```python
def save_ir(gm, path):
    rendered = gm.print_readable(print_output=False)
    Path(path).write_text(rendered, encoding="utf-8")
```

`print_readable()` 是 FX GraphModule 内置方法，输出人类可读的 Python-like IR，
包括每个节点的 op、target、args，方便论文中展示或人工检查。

---

## 九、BW 图的惰性生成机制

### 9.1 为什么需要 `loss.backward()`

```python
compiled = torch.compile(model, backend=_backend)
output = compiled(input_ids)      # ← 此时只触发 FW 编译
loss = loss_fn(output)
loss.backward()                   # ← 此时才触发 BW 编译
```

AOTAutograd 的设计是**惰性**的：
- FW 编译在首次 forward 时完成
- BW 编译在首次 backward 时完成

原因：BW 图依赖于 FW 的 output 信息（哪些需要梯度、tangent shapes 等），
而这些信息在 FW 执行完之后才完全确定。

### 9.2 在 `capture_graphs()` 中的体现

```python
output = compiled(sample_input_ids)  # ← fw_compiler 被调用，captured["fw"] 填充
loss = loss_fn(output)
loss.backward()                       # ← bw_compiler 被调用，captured["bw"] 填充
```

如果不执行 `backward()`，`captured["bw"]` 将为 None → 抛出 RuntimeError。

---

## 十、完整调用链

```
capture_graphs(model, input_ids, loss_fn, partition_fn=default_partition)
    │
    ├── torch._dynamo.reset()
    │
    ├── torch.compile(model, backend=_backend, dynamic=True)
    │     │
    │     └── TorchDynamo 字节码追踪
    │           └── 输出: gm (高层 FX Graph)
    │
    ├── _backend(gm, example_inputs)
    │     │
    │     └── aot_module_simplified(gm, example_inputs,
    │           │                    fw_compiler, bw_compiler, partition_fn)
    │           │
    │           ├── params_flat = [*named_parameters, *named_buffers]
    │           ├── full_args = params_flat + runtime_args
    │           ├── fake_flat_args = process_inputs(full_args)    ← FakeTensor 化
    │           │
    │           └── _create_aot_dispatcher_function(...)
    │                 │
    │                 ├── fw_metadata = run_functionalized_fw_and_collect_metadata(...)
    │                 │                  ← FakeTensorMode 下执行 FW，收集输入/输出元信息
    │                 │
    │                 └── aot_dispatch_autograd(...)
    │                       │
    │                       ├── aot_dispatch_autograd_graph(...)
    │                       │     │
    │                       │     ├── create_joint(fn)
    │                       │     │     └── inner_fn(primals, tangents):
    │                       │     │           outs = fn(*primals)
    │                       │     │           grads = torch.autograd.grad(outs, primals, tangents)
    │                       │     │           return (outs, grads)
    │                       │     │
    │                       │     ├── _create_graph(joint_fn, joint_inputs)
    │                       │     │     └── make_fx(joint_fn)(*joint_inputs)
    │                       │     │           ← FakeTensorMode + FunctionalTensorMode
    │                       │     │           ← 生成联合 FX 图 (joint graph)
    │                       │     │
    │                       │     └── eliminate_dead_code() + copy_fwd_metadata_to_bw_nodes()
    │                       │
    │                       └── partition_fn(joint_graph, num_fwd_outputs=N)
    │                             │
    │                             ├── default_partition:
    │                             │     ├── _extract_graph_with_inputs_outputs(→ fwd_only)
    │                             │     ├── 收集 saved_values (FW 节点被 BW 使用)
    │                             │     └── _extract_fwd_bwd_modules()
    │                             │           ├── FW output = user_out + saved_values
    │                             │           └── BW input  = saved_values + tangents
    │                             │
    │                             └── min_cut_partition:
    │                                   ├── solve_min_cut() → 最优 saved_values
    │                                   └── _extract_fwd_bwd_modules()
    │
    ├── compiled(input_ids)          ← 触发 FW 编译
    │     └── fw_compiler(fw_gm)    ← captured["fw"] = deepcopy(fw_gm)
    │
    ├── loss = loss_fn(output)
    │
    ├── loss.backward()              ← 触发 BW 编译
    │     └── bw_compiler(bw_gm)    ← captured["bw"] = deepcopy(bw_gm)
    │
    └── return captured["fw"], captured["bw"]
```

---

## 十一、`fw_gm` / `bw_gm` 的结构

### 11.1 fw_gm (前向图)

```
placeholder: primals_0  (param: embedding.weight)      meta['val'] = FakeTensor(V, H)
placeholder: primals_1  (param: attn.q_proj.weight)    meta['val'] = FakeTensor(H, H)
...
placeholder: primals_N  (input: input_ids)              meta['val'] = FakeTensor(B, S)

call_function: embedding = aten.embedding(primals_0, primals_N)  meta['val'] = FakeTensor(B, S, H)
call_function: q = aten.linear(embedding, primals_1)             meta['val'] = FakeTensor(B, S, H)
...
call_function: logits = aten.linear(hidden, primals_M)           meta['val'] = FakeTensor(B, S, V)

output: (logits,                   ← 用户输出
         embedding, q, k, v, ...,  ← saved activations
         sym_size_0, sym_size_1)   ← 符号化 shape
```

### 11.2 bw_gm (反向图)

```
placeholder: saved_sym_0, saved_sym_1         ← 符号化 shape
placeholder: embedding, q, k, v, ...          ← saved activations (= FW output 的后半部分)
placeholder: tangent_0                         ← grad_output (loss 对 logits 的梯度)

call_function: grad_hidden = aten.mm(tangent_0, primals_M.T)
call_function: grad_q = ...
...
call_function: grad_embedding_weight = ...

output: (grad_primal_0, grad_primal_1, ..., grad_primal_N)  ← 对每个 requires_grad 参数的梯度
```

---

## 十二、源码文件索引

| 文件 | 内容 | 行数 |
|------|------|------|
| `toolkit/capture/__init__.py` | 模块导出 | 4 行 |
| `toolkit/capture/aot_capture.py` | `capture_graphs()` — 图捕获主入口 | 65 行 |
| `toolkit/capture/analysis.py` | `graph_stats` + `analyze_graph` + `count_fw_output_bytes` | 95 行 |
| ~~`toolkit/capture/ir_saver.py`~~ | `save_ir()` — 已删除 | - |
| `torch/_functorch/aot_autograd.py:1007-1191` | `aot_module_simplified` — AOTAutograd 入口 |
| `torch/_functorch/aot_autograd.py:585-836` | `_create_aot_dispatcher_function` — 核心调度 |
| `torch/_functorch/_aot_autograd/dispatch_and_compile_graph.py:251-338` | `aot_dispatch_autograd_graph` — 联合图生成 |
| `torch/_functorch/_aot_autograd/traced_function_transforms.py:192-282` | `create_joint` — FW+BW 联合函数 |
| `torch/_functorch/_aot_autograd/dispatch_and_compile_graph.py:46-62` | `_create_graph` — make_fx 追踪 |
| `torch/_functorch/partitioners.py:389-472` | `default_partition` — 默认分割 |
| `torch/_functorch/partitioners.py:290-386` | `_extract_fwd_bwd_modules` — FW/BW 拆分 |
| `torch/_functorch/partitioners.py:158-226` | `_extract_graph_with_inputs_outputs` — 子图提取 |

---

## 参考

- `01-architecture.md` §四：图捕获概述
- `06-activation-memory-budget.md`：min-cut partition 详解
- `07-ac-sac-deep-dive.md`：AC/SAC 如何影响 partition
- `08-static-simulation-deep-dive.md`：L2 仿真如何消费 fw_gm/bw_gm
- PyTorch docs: [torch.compile](https://pytorch.org/docs/stable/torch.compiler.html)
- PyTorch docs: [AOTAutograd](https://pytorch.org/functorch/stable/notebooks/aot_autograd_optimizations.html)
