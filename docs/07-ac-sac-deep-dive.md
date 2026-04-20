# 07 — Activation Checkpointing (AC) 与 Selective AC (SAC) 源码深度解析

> **文档定位**: 对 PyTorch 2.6 中经典 AC 和 SAC 的全链路源码解析，
> 涵盖 eager 路径原理、torch.compile 路径的 HigherOrderOp 机制、
> SAC policy 如何转化为 FX 图 `recompute` tag、以及 AC/SAC 与 min-cut 的组合原理。
>
> **源码版本**: PyTorch 2.6.0 (torch2.6-gpu conda env)
>
> **前置阅读**: `06-activation-memory-budget.md`（min-cut 与 budget 机制）

---

## 一、概述：三层重计算策略

PyTorch 提供三个层次的重计算策略，粒度从粗到细：

| 层次 | 技术 | 粒度 | 控制方式 |
|------|------|------|---------|
| **Classic AC** | `torch.utils.checkpoint.checkpoint` | 模块级 | 包裹整个 TransformerBlock |
| **SAC** | `create_selective_checkpoint_contexts` + policy_fn | 算子级 | 按 op 类型决定 save/recompute |
| **Min-cut + budget** | `min_cut_rematerialization_partition` | 图级（自动） | `activation_memory_budget` 参数 |

三者可以**叠加使用**——AC/SAC 先标记哪些区域/算子需要重算，min-cut 再在此基础上做进一步优化。

---

## 二、Classic AC：Activation Checkpointing

### 2.1 核心思想

AC 的思想非常简单：

```
前向：正常执行，但不保存中间激活 → 只保存输入
反向：需要梯度时，重新执行前向来恢复中间激活
```

以一个 Transformer Block 为例：

```
正常训练：         AC 训练：
FW: x → [block] → y     FW: x → [block] → y      ← 只保存 x，不保存中间值
     保存所有中间值            丢弃中间值
BW: 直接用保存的中间值   BW: 先重跑 block(x) → 恢复中间值
     计算梯度                  再计算梯度
```

**内存节省**: 不需要保存 block 内部的所有中间激活（如 attention scores, FFN 中间层等），
只需保存 block 的输入。代价是反向时额外执行一次前向。

### 2.2 Eager 路径实现

**源文件**: `torch/utils/checkpoint.py`

#### `checkpoint()` 入口（第 343-501 行）

```python
@torch._disable_dynamo     # Dynamo 不 trace 此函数内部
def checkpoint(
    function,
    *args,
    use_reentrant: Optional[bool] = None,
    context_fn: Callable = noop_context_fn,   # SAC 通过此参数注入
    determinism_check: str = _DEFAULT_DETERMINISM_MODE,
    debug: bool = False,
    **kwargs
):
```

有两种实现模式：

| 模式 | 参数 | 实现类 | 特点 |
|------|------|-------|------|
| **Reentrant** | `use_reentrant=True` | `CheckpointFunction` (autograd.Function) | 旧版，FW 在 `torch.no_grad()` 下执行 |
| **Non-reentrant** | `use_reentrant=False` | `_checkpoint_without_reentrant_generator` | 推荐，支持 `context_fn`（SAC 依赖此参数）|

#### Reentrant 模式（`CheckpointFunction`，第 225-327 行）

```python
class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args):
        ctx.save_for_backward(*tensor_inputs)      # 只保存 tensor 输入
        with torch.no_grad():                       # 关键：FW 在 no_grad 下执行
            outputs = run_function(*args)            # 中间激活不被 autograd 追踪
        return outputs

    @staticmethod
    def backward(ctx, *args):
        inputs = ...                                 # 从 ctx 恢复输入
        with torch.enable_grad():
            outputs = ctx.run_function(*detached_inputs)  # 重新执行前向
        torch.autograd.backward(outputs_with_grad, args_with_grad)  # 再计算梯度
        return grads
```

**原理**：
1. 前向在 `torch.no_grad()` 下执行 → autograd 不记录任何中间 op → 不保存中间激活
2. 反向时重新执行 `run_function` → 恢复中间激活 → 再用 `torch.autograd.backward` 计算梯度
3. RNG 状态被保存并恢复 → 保证 dropout 等随机 op 的确定性

#### Non-reentrant 模式（第 1422-1549 行）

Non-reentrant 模式使用 `saved_tensors_hooks` 机制：

```python
class _checkpoint_hook(torch.autograd.graph.saved_tensors_hooks):
    def __init__(self, frame):
        def pack_hook(x):
            holder = _Holder()                       # 前向：不保存 tensor，只保存空 holder
            frame.weak_holders.append(weakref.ref(holder))
            return holder

        def unpack_hook(holder):
            if not frame.is_recomputed[gid]:
                frame.recompute_fn(*args)            # 反向首次 unpack 时触发重算
                frame.is_recomputed[gid] = True
            return frame.recomputed[gid][holder]     # 返回重算出的 tensor
```

**原理**：
1. `pack_hook`：前向时，每个 autograd 要保存的 tensor 被替换为空 `_Holder` → 内存不实际保存
2. `unpack_hook`：反向时首次需要 tensor 时，触发一次完整重算（`recompute_fn`）
3. `early_stop`：一旦所有需要的 tensor 都已重算出来，停止重算 → 避免不必要的计算
4. 支持 `context_fn` 参数 → SAC 通过此参数注入

### 2.3 本项目的 AC 封装

**源文件**: `toolkit/strategy/classic_ac.py`

```python
def wrap_with_checkpoint(model, block_class_name, use_reentrant=False):
    for module in model.modules():
        if module.__class__.__name__ != block_class_name:
            continue
        original_forward = module.forward
        module._original_forward = original_forward
        module.forward = _make_checkpoint_forward(original_forward, use_reentrant)
    return model

def _make_checkpoint_forward(original_forward, use_reentrant):
    def checkpointed_forward(*args, **kwargs):
        return checkpoint(original_forward, *args, use_reentrant=use_reentrant, **kwargs)
    return checkpointed_forward
```

遍历模型，将所有匹配 `block_class_name`（如 `"LlamaDecoderLayer"`）的模块的 `forward`
替换为 `checkpoint(original_forward, ...)`。

---

## 三、SAC：Selective Activation Checkpointing

### 3.1 核心思想

Classic AC 将整个 block 的所有中间值都丢弃再重算，过于粗暴。
SAC 允许**按算子类型选择**哪些保存、哪些重算：

```
Classic AC:  block 内所有 op 的输出都不保存 → 全部重算
SAC:         matmul 输出 → 保存（计算昂贵）
             add/relu/gelu 输出 → 不保存（重算便宜）
```

### 3.2 CheckpointPolicy 枚举（第 1226-1252 行）

```python
class CheckpointPolicy(enum.Enum):
    MUST_SAVE = 0          # 必须保存，compile 也不能覆盖
    PREFER_SAVE = 1        # 倾向保存，compile 可以覆盖
    MUST_RECOMPUTE = 2     # 必须重算，compile 也不能覆盖
    PREFER_RECOMPUTE = 3   # 倾向重算，compile 可以覆盖
```

`MUST_*` vs `PREFER_*` 的区别：当与 `torch.compile` 结合时，`PREFER_*`
允许编译器（min-cut partitioner）覆盖用户的选择，而 `MUST_*` 是强制约束。

### 3.3 SAC 的两个 TorchDispatchMode（第 1270-1331 行）

SAC 通过一对 context manager 实现——一个用于前向，一个用于反向重算：

#### `_CachingTorchDispatchMode`（前向时激活）

```python
class _CachingTorchDispatchMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        policy = self.policy_fn(ctx, func, *args, **kwargs)   # 用户 policy 决策

        is_compiling = _is_compiling(func, args, kwargs)
        if is_compiling:
            fx_traceback.current_meta["recompute"] = policy    # 编译模式：写入 FX node meta

        out = func(*args, **kwargs)

        if policy in (MUST_SAVE, PREFER_SAVE) or is_compiling:
            self.storage[func].append(detach(out))             # eager 模式：缓存输出
        return out
```

#### `_CachedTorchDispatchMode`（反向重算时激活）

```python
class _CachedTorchDispatchMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        policy = self.policy_fn(ctx, func, *args, **kwargs)

        if policy in (MUST_SAVE, PREFER_SAVE) or is_compiling:
            out = self.storage[func].pop(0)                    # 从缓存取出（不重算）
        else:
            out = func(*args, **kwargs)                        # 重新执行（重算）
        return out
```

**Eager 路径流程**：
1. 前向：`_CachingTorchDispatchMode` 截获每个 op，按 policy 决定是否缓存其输出
2. 反向：`_CachedTorchDispatchMode` 截获每个 op，`SAVE` 的直接从缓存取，`RECOMPUTE` 的重新执行

### 3.4 `create_selective_checkpoint_contexts`（第 1334-1417 行）

```python
def create_selective_checkpoint_contexts(policy_fn_or_list, allow_cache_entry_mutation=False):
    storage = defaultdict(list)
    return (
        _CachingTorchDispatchMode(policy_fn, storage),      # 前向 context
        _CachedTorchDispatchMode(policy_fn, storage, ...),  # 反向 context
    )
```

返回一对 context manager，通过 `checkpoint(..., context_fn=partial(create_selective_checkpoint_contexts, policy_fn))`
传入。

### 3.5 本项目的 SAC 封装

**源文件**: `toolkit/strategy/sac.py`

```python
SAC_POLICIES = {
    "save_matmuls": lambda ctx, op, *args, **kwargs: (
        CheckpointPolicy.MUST_SAVE
        if op in {aten.mm.default, aten.addmm.default, aten.bmm.default}
        else CheckpointPolicy.PREFER_RECOMPUTE
    ),
    "save_attention": lambda ctx, op, *args, **kwargs: (
        CheckpointPolicy.MUST_SAVE
        if op in {aten.mm.default, aten.addmm.default, aten.bmm.default,
                  aten._scaled_dot_product_flash_attention.default,
                  aten._scaled_dot_product_efficient_attention.default}
        else CheckpointPolicy.PREFER_RECOMPUTE
    ),
    "recompute_all": lambda ctx, op, *args, **kwargs: CheckpointPolicy.PREFER_RECOMPUTE,
}

def wrap_with_sac(model, block_class_name, policy_name="save_matmuls"):
    policy_fn = SAC_POLICIES[policy_name]
    for module in model.modules():
        if module.__class__.__name__ != block_class_name:
            continue
        module.forward = _make_sac_forward(module.forward, policy_fn)
    return model

def _make_sac_forward(original_forward, policy_fn):
    context_fn = partial(create_selective_checkpoint_contexts, policy_fn)
    def checkpointed_forward(*args, **kwargs):
        return checkpoint(original_forward, *args,
                          use_reentrant=False, context_fn=context_fn, **kwargs)
    return checkpointed_forward
```

三种预定义 policy：

| Policy 名 | 保存什么 | 重算什么 | 效果 |
|-----------|---------|---------|------|
| `save_matmuls` | mm/addmm/bmm | 其余全部 | 保留最贵的矩阵乘输出 |
| `save_attention` | matmul + flash_attention | 其余全部 | 保留 attention 相关 |
| `recompute_all` | 无 | 全部 | 等效于 Classic AC |

---

## 四、AC/SAC 在 torch.compile 下的工作原理

这是最关键的部分——解释为什么 AC/SAC 可以和 min-cut 叠加。

### 4.1 Dynamo 对 checkpoint 的处理

`checkpoint()` 函数被 `@torch._disable_dynamo` 装饰，Dynamo 不会 trace 进去。
但 Dynamo 会**特殊处理**它：

```
Dynamo trace 遇到 checkpoint()
  ↓
识别为 HigherOrderOp → TagActivationCheckpoint
  ↓
将 checkpointed function 捕获为子图 (gmod)
  ↓
在子图节点上打 tag:
  - 无 SAC: node.meta["recompute"] = PREFER_RECOMPUTE     (全部重算)
  - 有 SAC: node.meta["recompute"] = None (先置空，待 SAC policy 填充)
```

**源文件**: `torch/_higher_order_ops/wrap.py` 第 129-241 行

### 4.2 TagActivationCheckpoint.tag_nodes（第 188-199 行）

```python
def tag_nodes(self, gmod, is_sac):
    unique_graph_id = next(uid)
    for node in gmod.graph.nodes:
        if node.op in ("call_function", "call_method", "call_module"):
            node.meta["ac_graph_id"] = unique_graph_id     # 标记属于哪个 checkpoint 区域
            if is_sac:
                node.meta["recompute"] = None              # SAC: 待 policy 填充
            else:
                node.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE  # AC: 全部重算
    return gmod
```

关键点：
- `ac_graph_id`：标识不同 checkpoint 区域（每个 TransformerBlock 有不同的 id）
- `recompute`：AC 模式下直接设为 `PREFER_RECOMPUTE`；SAC 模式下先置空

### 4.3 SAC 在编译路径的 policy 填充

当 `context_fn` 存在时，Dynamo 会在 AOT Autograd tracing 时执行 SAC 的
`_CachingTorchDispatchMode`（第 1286-1290 行）：

```python
is_compiling = _is_compiling(func, args, kwargs)
if is_compiling:
    # 编译模式下：将 policy 结果写入 FX 图节点的 meta
    fx_traceback.current_meta["recompute"] = policy
```

这样，经过 AOT Autograd tracing 后，joint graph 中的每个节点都有了 `recompute` tag：
- **AC 区域内的节点**: `PREFER_RECOMPUTE`（建议全部重算）
- **SAC 区域内的节点**: 根据 policy_fn 可能是 `MUST_SAVE` 或 `PREFER_RECOMPUTE`
- **非 checkpoint 区域的节点**: 没有 `recompute` tag

### 4.4 cleanup_recompute_tags（partitioners.py 第 777-812 行）

在 partitioner 开始前，先做一次清理：

```python
def cleanup_recompute_tags(joint_module):
    """如果两个连续的 checkpoint block 之间没有 op，
    需要在边界处保存 tensor（标记为 MUST_SAVE）。"""
    for node in joint_module.graph.nodes:
        if must_recompute(node):
            for user in node.users:
                if must_recompute(user) and user.meta["ac_graph_id"] > node.meta["ac_graph_id"]:
                    node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
```

**原因**: 如果 Block A 的输出直接是 Block B 的输入，这个 tensor 必须在两个 block 之间保存，
不能被重算（否则两个 block 形成循环依赖）。

---

## 五、AC/SAC 与 min-cut 的组合：为什么可以叠加

### 5.1 组合机制

**核心**: AC/SAC 通过 `recompute` tag 告诉 min-cut "这些节点应该被重算"，
min-cut 将此作为**约束**融入图分割算法。

在 `solve_min_cut()` 中（partitioners.py 第 815-1200 行）：

```python
# 第 992-993 行：如果节点被标记为 must_recompute，不 ban 它
def ban_recomputation_if_allowed(node):
    if must_recompute(node):       # 有 MUST_RECOMPUTE 或 PREFER_RECOMPUTE tag
        return False               # 不 ban → 允许被重算
    banned_nodes.add(node)
    return True

# 第 1024-1030 行：must_recompute 的节点，强制添加 inf 边到 sink
if must_recompute(node):
    # 添加 inf 容量边: X_in → sink
    # 保证 X 在 min-cut 后属于 sink 侧 → 必定被重算
    nx_graph.add_edge(node.name + "_in", "sink", capacity=math.inf)
    continue

# 第 906 行：MUST_SAVE 的节点被 ban（禁止重算）
if node.meta.get("recompute", None) == CheckpointPolicy.MUST_SAVE:
    return True   # ban → 必须保存
```

### 5.2 三种 tag 在 min-cut 中的效果

| Tag | 在 min-cut 流网络中 | 效果 |
|-----|---------------------|------|
| `MUST_RECOMPUTE` / `PREFER_RECOMPUTE` | X_in → sink (inf) | 强制在 sink 侧 → **必定被重算** |
| `MUST_SAVE` | source → X_in (inf) | 强制在 source 侧 → **必定被保存** |
| `PREFER_SAVE` | source → X_in (inf) | 同 MUST_SAVE |
| 无 tag | X_in → X_out (weight) | min-cut 自由决定 |

### 5.3 组合流程图

```
模型代码:
    wrap_with_checkpoint(model, "LlamaDecoderLayer")   # 或 wrap_with_sac
    compiled = torch.compile(model, backend="inductor")

              ↓

Dynamo trace:
    遇到 checkpoint() → TagActivationCheckpoint
    AC: 所有 block 内节点 → recompute = PREFER_RECOMPUTE
    SAC: mm/bmm → recompute = MUST_SAVE, 其余 → recompute = PREFER_RECOMPUTE

              ↓

AOT Autograd:
    生成 joint fw+bw graph
    每个节点带有 recompute tag + ac_graph_id

              ↓

cleanup_recompute_tags():
    block 边界节点 → MUST_SAVE（防止循环依赖）

              ↓

min_cut_rematerialization_partition():
    1. 读取 memory_budget = config.activation_memory_budget
    2. choose_saved_values_set(budget)
       - MUST_RECOMPUTE 节点 → min-cut 流网络中被强制重算
       - MUST_SAVE 节点 → min-cut 流网络中被强制保存
       - 无 tag 节点 → min-cut 自由优化
       - budget < 1.0 → 在 AC/SAC 标注基础上进一步调整

              ↓

fw_module / bw_module:
    - AC 标记的节点出现在 bw_module 中（重算）
    - SAC MUST_SAVE 的节点留在 fw_module output（保存）
    - min-cut 在剩余自由节点上做最优分割
```

### 5.4 为什么组合有效

1. **AC/SAC 提供"建议"**: 用户通过 AC 说"整个 block 都可以重算"，或通过 SAC 说"matmul 必须保存、其余可以重算"
2. **min-cut 做"精细优化"**: 在用户建议的基础上，min-cut 还可以进一步优化——
   - `PREFER_RECOMPUTE` 节点只是"建议"重算，min-cut 可以决定其中某些实际上保存更优（当 budget=1.0 时）
   - 对于 AC/SAC 区域之外的节点，min-cut 仍然独立做最优决策
3. **budget 提供"力度控制"**: 通过 `activation_memory_budget` 可以在 AC/SAC 的基础上进一步控制整体内存

实际效果（ex3 实验数据, LLaMA 6L/512H）：

| 策略 | 组合 | fwbw_peak |
|------|------|-----------|
| S05 aot_eager+default | 无重计算 baseline | 1033 MB |
| S10 ac+aot_eager(default) | AC only | 672 MB (↓35%) |
| S11 ac+inductor | AC + min-cut + fusion | 596 MB (↓42%) |
| S12 sac_mm+inductor | SAC + min-cut + fusion | 628 MB (↓39%) |

S10→S11 说明：AC 已经让 block 内节点可重算，Inductor 的 min-cut 在此基础上进一步优化了
非 AC 区域的节点分割 + kernel fusion 减少了中间临时 tensor。

---

## 六、Eager 路径 vs Compile 路径对比

| 维度 | Eager AC/SAC | Compiled AC/SAC |
|------|-------------|-----------------|
| 前向执行 | `saved_tensors_hooks` 拦截保存 | Dynamo trace → FX 图 + `recompute` tag |
| 反向重算 | `unpack_hook` 触发 `recompute_fn` | 编译器生成的 bw_module 包含重算 op |
| SAC 缓存 | `TorchDispatchMode` 运行时截获 | policy 结果编码为静态 `recompute` tag |
| 优化空间 | 仅 op 级选择 | + min-cut 全局优化 + kernel fusion |
| 开销 | hook dispatch 开销 | 编译一次，后续零开销 |
| 限制 | SAC+compile 在 PyTorch 2.6 可能崩溃 | 不支持 `use_reentrant=True` |

关键区别：**Eager 路径是运行时动态决策（每次前向都执行 hook），Compile 路径是编译时静态决策
（tag 固化在图中，后续执行无额外开销）。**

---

## 七、MUST vs PREFER 的实际意义

```
AC 模式:
  所有 block 内节点 → PREFER_RECOMPUTE
  ↓
  min-cut with budget=1.0:
    PREFER 只是建议，min-cut 可以选择保存某些节点
    → 比纯 AC 更智能：昂贵节点可能被保留
  min-cut with budget=0.0:
    强制全部重算（与纯 AC 一致）

SAC 模式 (save_matmuls):
  mm/addmm/bmm → MUST_SAVE (强制保存)
  其余 → PREFER_RECOMPUTE (建议重算)
  ↓
  min-cut:
    MUST_SAVE 节点必定被保存（不受 budget 影响）
    PREFER_RECOMPUTE 节点由 min-cut 自由决定
    → 用户保证了关键 op 不被重算
```

如果用户希望某个 op 无论如何不被重算，应使用 `MUST_SAVE`（不会被 compile 覆盖）。
如果只是倾向但允许编译器优化，使用 `PREFER_SAVE`。

---

## 八、完整调用链（以 S11 ac+inductor 为例）

```
wrap_with_checkpoint(model, "LlamaDecoderLayer")
  └── module.forward = checkpoint(original_forward, use_reentrant=False)

torch.compile(model, backend="inductor")
  │
  ├── Dynamo trace
  │     遇到 checkpoint() → HigherOrderOp
  │     TagActivationCheckpoint.tag_nodes(gmod, is_sac=False)
  │       └── 每个 call_function 节点:
  │             node.meta["ac_graph_id"] = uid
  │             node.meta["recompute"] = PREFER_RECOMPUTE
  │
  ├── AOT Autograd
  │     joint fw+bw graph 生成
  │     block 内节点带 recompute=PREFER_RECOMPUTE
  │     block 外节点无 recompute tag
  │
  ├── cleanup_recompute_tags()
  │     block 边界: node.meta["recompute"] = MUST_SAVE
  │
  ├── min_cut_rematerialization_partition()              [partitioners.py:1703]
  │     memory_budget = config.activation_memory_budget  [:1819]
  │     │
  │     └── choose_saved_values_set(budget=1.0)          [:1466]
  │           │
  │           └── solve_min_cut()                        [:815]
  │                 │
  │                 ├── must_recompute(node)?             [:992]
  │                 │   → True: X_in→sink (inf)          [:1029]
  │                 │          该节点必定被重算
  │                 │
  │                 ├── MUST_SAVE?                        [:906]
  │                 │   → True: source→X_in (inf)
  │                 │          该节点必定被保存
  │                 │
  │                 ├── 无 tag?
  │                 │   → should_ban_recomputation()      [:901]
  │                 │     min-cut 自由决定
  │                 │
  │                 └── nx.minimum_cut()                  [:1176]
  │                       → saved_values
  │
  └── _extract_fwd_bwd_modules()                         [:1834]
        fw_module: 输出 = 用户输出 + saved_values
        bw_module: 输入 = saved_values + tangents
                   包含 AC 标记节点的重算 ops
```

---

## 九、源码文件索引

以下所有路径位于 `torch2.6-gpu` conda env 中：

| 文件 | 关键内容 | 行号 |
|------|---------|------|
| `torch/utils/checkpoint.py` | `checkpoint()` 入口 | 343-501 |
| 同上 | `CheckpointFunction` (reentrant) | 225-327 |
| 同上 | `_checkpoint_without_reentrant_generator` (non-reentrant) | 1422-1549 |
| 同上 | `_checkpoint_hook` (saved_tensors_hooks) | 1099-1152 |
| 同上 | `CheckpointPolicy` 枚举 | 1226-1252 |
| 同上 | `_CachingTorchDispatchMode` (SAC 前向) | 1270-1298 |
| 同上 | `_CachedTorchDispatchMode` (SAC 反向) | 1300-1331 |
| 同上 | `create_selective_checkpoint_contexts` | 1334-1417 |
| `torch/_higher_order_ops/wrap.py` | `TagActivationCheckpoint` | 129-241 |
| 同上 | `tag_nodes()` — 设置 recompute tag | 188-199 |
| `torch/_functorch/partitioners.py` | `must_recompute()` | 117-121 |
| 同上 | `cleanup_recompute_tags()` | 777-812 |
| 同上 | `solve_min_cut()` — recompute tag 如何影响流网络 | 815-1200 |
| 同上 | `should_ban_recomputation()` — MUST_SAVE 处理 | 901-949 |

本项目封装：

| 文件 | 内容 |
|------|------|
| `toolkit/strategy/classic_ac.py` | `wrap_with_checkpoint` / `unwrap_checkpoint` |
| `toolkit/strategy/sac.py` | `SAC_POLICIES` / `wrap_with_sac` |
| `toolkit_examples/ex3_simulation_accuracy.py` | Group 1 (eager AC/SAC) + Group 3 (AC/SAC + compiled) |

---

## 参考

- PyTorch 官方文档: [Activation Checkpointing](https://pytorch.org/docs/stable/checkpoint.html)
- TorchTitan: SAC 在大规模训练中的应用
- `06-activation-memory-budget.md`: min-cut + budget 机制详解
