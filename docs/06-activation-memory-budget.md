# 06 — `activation_memory_budget` 源码深度解析

> **文档定位**: 对 PyTorch 2.6 中 `activation_memory_budget` 参数的全链路源码解析，
> 涵盖配置定义、Inductor 调用入口、三级 min-cut 逐步放松、0-1 背包精确求解、
> Op 白名单/黑名单分类以及本项目的封装层。
>
> **源码版本**: PyTorch 2.6.0 (torch2.6-gpu conda env)

---

## 一、概述

`activation_memory_budget` 是 PyTorch 编译器 (Inductor / AOT Autograd) 中控制
**激活内存与重计算时间权衡**的核心参数。取值 0.0~1.0：

| budget 值 | 含义 | 行为 |
|-----------|------|------|
| **1.0** (默认) | 保留 100% 激活 | 标准 min-cut，只重算廉价 op → 最优运行时 |
| **0.5** | 保留约 50% | 背包求解，重算约一半大张量 → 中等平衡 |
| **0.0** | 保留 0% | 只保存模型输入，其余全部重算 → 类似全 AC |

> 注意：budget=0.0 并不一定比 0.5 更省内存。全重算导致 BW 重算压力过大，
> backward peak 可能反而更高。这正是背包优化的价值：找到最优平衡点。

---

## 二、配置定义

**源文件**: `torch/_functorch/config.py`

### 2.1 核心参数（第 104-128 行）

```python
# By default, the partitioner is purely trying to optimize for runtime (although
# it should always use less memory than eager)
# This knob controls the partitioner to make that tradeoff for you, choosing the
# fastest option that saves less activations than the memory budget.
# Specifically, 0.0 corresponds to the activation memory from applying
# activation checkpointing to the full compiled region, and 1.0 corresponds to
# the activation memory from the default runtime-optimized strategy.  So, 0.4
# would result in a strategy that saves 40% of the activations compared to the
# default strategy.
# It solves a 0-1 knapsack to find the minimum recompute necessary to stay below
# the activation memory budget.
activation_memory_budget = 1.0

# This controls how we estimate the runtime when deciding what the cheapest
# operators to recompute are. The 3 options are
# "flops": Bases it off of the flop count provided by torch.utils.flop_counter
# "profile": Benchmarks each operator to come up with a runtime
# "testing": Returns 1 for everything
activation_memory_budget_runtime_estimator = "flops"

# This controls the solver used for the 0-1 knapsack. By default we use a
# quantized DP solution ("dp"). The other approaches are a "greedy" and a "ilp"
# (which has a scipy dependency).
activation_memory_budget_solver = "dp"
```

三个相关配置项：
- **`activation_memory_budget`**: 核心参数，0.0~1.0
- **`activation_memory_budget_runtime_estimator`**: 估算重算代价的方式（默认用 flops 计数）
- **`activation_memory_budget_solver`**: 背包求解器（默认 DP 动态规划，也支持 greedy、ILP）

### 2.2 Ban 开关（第 82-102 行）

```python
ban_recompute_used_far_apart = True          # 禁止重算离当前节点读取位置很远的节点
ban_recompute_long_fusible_chains = True      # 禁止在 BW 形成任意长的重算融合链
ban_recompute_materialized_backward = True    # 禁止重算 BW 中需要物化的节点
ban_recompute_not_in_allowlist = True         # 仅允许白名单 op 重算
ban_recompute_reductions = True               # 禁止重算 reduction（结果小但重算贵）
recompute_views = False                       # view op 是否保存（False=总是重算 view）
```

以及一个"全部放松"的总开关：

```python
# Sets all of the ban_recompute heuristics to False except ban_recompute_reductions
aggressive_recomputation = False              # 第 141 行
```

这些 ban 开关在 `budget < 1.0` 时会被逐步放松（详见第四节）。

---

## 三、调用入口：`min_cut_rematerialization_partition()`

**源文件**: `torch/_functorch/partitioners.py` 第 1703-1878 行

当使用 `torch.compile(model, backend="inductor")` 编译模型后执行前向时，
触发以下链路：

```
torch.compile
  → Dynamo trace (Python → FX graph)
    → AOT Autograd (joint fw+bw graph)
      → min_cut_rematerialization_partition()      ← 入口
        → choose_saved_values_set(budget=b)        ← 核心决策
          → saved_values → fw/bw 分割
```

### 3.1 budget 读取（第 1819-1823 行）

```python
memory_budget = config.activation_memory_budget          # 从全局 config 读取
for node in joint_graph.nodes:
    if isinstance(node.meta.get("memory_budget", None), float):
        memory_budget = node.meta["memory_budget"]       # per-graph 覆盖优先
        break
```

先读全局 `config.activation_memory_budget`，但如果图中某节点的 `meta["memory_budget"]`
有 float 值，优先使用（支持 per-graph 粒度覆盖）。

### 3.2 dist_from_bw 计算（第 1809-1817 行）

```python
for node in reversed(joint_module.graph.nodes):
    if node.op == "output":
        node.dist_from_bw = int(1e9)
    elif not node_info.is_required_fw(node):
        node.dist_from_bw = 0                            # BW 节点距离=0
    else:
        node.dist_from_bw = int(1e9)
        for user in node.users:
            node.dist_from_bw = min(node.dist_from_bw, user.dist_from_bw + 1)
```

反向遍历 joint graph，计算每个前向节点到反向图的距离。距离越大 → 离反向越远
→ 保存代价越高（在 min-cut 流网络中权重乘以 `1.1^dist`）。

### 3.3 传入核心决策（第 1824-1828 行）

```python
saved_values = choose_saved_values_set(
    joint_graph,
    node_info,
    memory_budget=memory_budget,
)
```

---

## 四、核心决策：`choose_saved_values_set()`

**源文件**: `torch/_functorch/partitioners.py` 第 1466-1700 行

这是整个 budget 机制的核心，分为四个阶段。

### 4.1 阶段 A：边界快速返回（第 1491-1501 行）

```python
if memory_budget == 0:
    return node_info.inputs                    # 只保存模型输入，其余全部重算

runtime_optimized_saved_values, _ = solve_min_cut(
    joint_graph, node_info, min_cut_options,
)
if memory_budget == 1:
    return runtime_optimized_saved_values       # 标准 min-cut，最优运行时
```

- **budget=0** → 只保存 inputs，类似全 AC
- **budget=1** → 标准 min-cut，只重算廉价 op

### 4.2 阶段 B：归一化内存比例（第 1503-1518 行）

```python
min_act_size = estimate_activations_size(node_info.inputs)          # budget=0 的极端
max_act_size = estimate_activations_size(runtime_optimized_saved_values)  # budget=1 的结果

def get_mem_ratio(activations):
    return (estimate_activations_size(activations) - min_act_size) / (
        max_act_size - min_act_size
    )
```

budget 的物理含义在此定义：

```
mem_ratio = (actual_size - min_size) / (max_size - min_size)
```

- `min_size` = 只保存 inputs（budget=0 的极端情况）
- `max_size` = runtime_optimized（budget=1 的结果）
- **budget=0.4 意味着：激活内存只允许 min→max 区间的 40%**

### 4.3 阶段 C：三级 min-cut 逐步放松（第 1520-1541 行）

逐步放松 ban 开关，每次放松后重新跑 min-cut，如果 `mem_ratio < budget` 就提前返回：

**第一级 — more_aggressive**（第 1520-1530 行）

```python
more_aggressive_options = replace(
    min_cut_options,
    ban_if_used_far_apart=False,
    ban_if_long_fusible_chains=False,
    ban_if_materialized_backward=False,
)
more_aggressive_saved_values, _ = solve_min_cut(joint_graph, node_info, more_aggressive_options)
if get_mem_ratio(more_aggressive_saved_values) < memory_budget:
    return more_aggressive_saved_values
```

**第二级 — aggressive**（第 1532-1541 行）

```python
aggressive_options = replace(
    more_aggressive_options,
    ban_if_not_in_allowlist=False,            # 再放松 allowlist 限制
)
aggressive_recomputation_saved_values, banned_nodes = solve_min_cut(
    joint_graph, node_info, aggressive_options
)
if get_mem_ratio(aggressive_recomputation_saved_values) < memory_budget:
    return aggressive_recomputation_saved_values
```

三级汇总：

| 级别 | 放松的 ban | 效果 |
|------|-----------|------|
| **default** (budget=1) | 无 | 保守，只重算白名单中的廉价 op |
| **more_aggressive** | far_apart + chains + materialized | 允许更多中间节点被重算 |
| **aggressive** | 上面 + allowlist | 几乎不限制哪些 op 可重算 |

### 4.4 阶段 D：0-1 背包精确求解（第 1547-1700 行）

当三级 min-cut 都无法满足 budget 约束时，进入背包求解。

**准备阶段：收集可重算的 banned 节点**（第 1547-1576 行）

```python
def get_recomputable_banned_nodes(banned_nodes):
    return [
        i for i in banned_nodes
        if (
            i.dist_from_bw < int(1e9)                     # 反向确实需要该节点
            and get_node_storage(i) not in input_storages  # 不是模型输入
        )
    ]

all_recomputable_banned_nodes = sorted(recomputable_banned_nodes, key=_size_of, reverse=True)

memories_banned_nodes = [get_normalized_size(_size_of(i)) for i in all_recomputable_banned_nodes]
runtimes_banned_nodes = [estimate_runtime(node) for node in all_recomputable_banned_nodes]
```

从 aggressive min-cut 中收集"被 ban 的节点"（通常是 `mm`, `bmm`, `attention` 等
compute-intensive ops）。每个节点有两个属性：
- **memory**: 保存该节点占多少内存（归一化）
- **runtime**: 重算该节点的计算代价（用 flops 估算）

**背包求解**（第 1579-1647 行）

```python
def get_saved_values_knapsack(memory_budget, node_info, joint_graph):
    (
        expected_runtime,
        saved_node_idxs,
        recomputable_node_idxs,
    ) = _optimize_runtime_with_given_memory(
        joint_graph,
        memories_banned_nodes,     # 每个节点的内存代价
        runtimes_banned_nodes,     # 每个节点的计算代价
        max(memory_budget, 0),     # 内存约束
        node_info,
        all_recomputable_banned_nodes,
    )
    dont_ban = set()
    for idx in recomputable_node_idxs:
        dont_ban.add(all_recomputable_banned_nodes[idx])

    saved_values, _ = solve_min_cut(
        joint_graph, node_info,
        aggressive_options,
        dont_ban,                  # 背包选中的节点从 ban 名单中移除
    )
    return saved_values, expected_runtime
```

核心思想：**在 memory <= budget 约束下，最小化重算带来的额外计算开销**。
背包选出的节点通过 `dont_ban` 参数传给最终的 `solve_min_cut()`，
使这些 compute-intensive op 也被允许重算。

---

## 五、背包求解器

**源文件**: `torch/_functorch/partitioners.py` 第 1392-1413 行

```python
def _optimize_runtime_with_given_memory(
    joint_graph, memory, runtimes, max_memory, node_info, all_recomputable_banned_nodes,
):
    SOLVER = config.activation_memory_budget_solver
    if SOLVER == "greedy":
        return greedy_knapsack(memory, runtimes, max_memory)
    elif SOLVER == "ilp":
        return ilp_knapsack(memory, runtimes, max_memory)
    elif SOLVER == "dp":
        return dp_knapsack(memory, runtimes, max_memory)   # 默认
    elif callable(SOLVER):
        saved_node_idx, recomp_node_idx = SOLVER(
            memory, joint_graph, max_memory, node_info, all_recomputable_banned_nodes
        )
        return (0.0, saved_node_idx, recomp_node_idx)
    else:
        raise RuntimeError(f"Not aware of memory budget knapsack solver: {SOLVER}")
```

| 求解器 | 说明 |
|--------|------|
| `"dp"` (默认) | 量化 DP 动态规划，精确但离散化 |
| `"greedy"` | 贪心：按 runtime/memory 比排序 |
| `"ilp"` | 整数线性规划（需 scipy） |
| callable | 支持自定义求解器 |

### 5.1 运行时估算（第 1419-1463 行）

```python
def estimate_runtime(node):
    RUNTIME_MODE = config.activation_memory_budget_runtime_estimator
    if RUNTIME_MODE == "testing":
        return 1
    elif RUNTIME_MODE == "profile":
        ms = benchmarker.benchmark_gpu(lambda: node.target(*args, **kwargs))
        return ms
    elif RUNTIME_MODE == "flops":
        with FlopCounterMode(display=False) as mode:
            node.target(*args, **kwargs)
        return max(mode.get_total_flops(), 1)
```

默认 `"flops"` 模式：用 `torch.utils.flop_counter` 统计每个 op 的浮点运算数作为
重算代价权重。`mm` 的 flops 远高于 `add`，因此 budget 下优先保留 `mm` 不重算。

---

## 六、Op 白名单/黑名单

**源文件**: `torch/_functorch/partitioners.py` 第 1220-1382 行 (`get_default_op_list()`)

### 6.1 允许重算的 op（白名单，约 130 个）

```python
default_recomputable_ops = [
    # 算术
    aten.add, aten.sub, aten.mul, aten.div, aten.pow, ...
    # 激活函数
    aten.relu, aten.silu, aten.gelu, aten.sigmoid, aten.tanh, ...
    # 数学函数
    aten.exp, aten.log, aten.sqrt, aten.rsqrt, aten.cos, aten.sin, ...
    # 比较
    aten.eq, aten.ne, aten.ge, aten.gt, aten.le, aten.lt, ...
    # 规约
    aten.sum, aten.mean, aten.amax, aten.var, aten.std, ...
    # 其他
    aten.where, aten.clamp, aten.clone, aten.full_like, operator.getitem, ...
]
```

这些 op 计算廉价（pointwise / view），重算代价很低。

### 6.2 view op（总是重算）

```python
recomputable_view_ops = [
    aten.squeeze, aten.unsqueeze, aten.alias,
    aten.view, aten.slice, aten.t,
    prims.broadcast_in_dim, aten.expand, aten.as_strided, aten.permute, aten.select,
]
```

view op 的重算代价为 0（不涉及数据拷贝），因此总是重算不保存。

### 6.3 compute-intensive op（黑名单，默认禁止重算）

```python
compute_intensive_ops = [
    aten.mm,                                        # 矩阵乘
    aten.bmm,                                       # 批矩阵乘
    aten.addmm,                                     # 加偏置矩阵乘（Linear 的底层 op）
    aten.convolution,                               # 卷积
    aten.convolution_backward,                      # 卷积反向
    aten._scaled_dot_product_flash_attention,        # Flash Attention
    aten._scaled_dot_product_efficient_attention,    # Efficient Attention
    aten._flash_attention_forward,
    aten._efficient_attention_forward,
    aten.upsample_bilinear2d,                       # 双线性上采样
    aten._scaled_mm,                                # 量化矩阵乘
]
```

这些 op 计算代价极高（$O(n^3)$ 或 $O(n^2 d)$），默认禁止重算。
只有当 `budget` 压得很低、三级 min-cut 都无法满足时，背包求解器才会从中挑选
"性价比最高的"（memory 大但 runtime 相对低的）来重算。

### 6.4 random op（绝对禁止重算）

```python
random_ops = [aten.native_dropout, aten.rand_like, aten.randn_like]
```

随机 op 结果不可复现，无论 budget 取何值都不会被重算。

---

## 七、本项目的封装

**源文件**: `toolkit/strategy/memory_budget.py`

```python
_PREVIOUS_BUDGET = None
_HAS_ACTIVE_OVERRIDE = False

def set_memory_budget(budget: float = 0.5) -> bool:
    """设置 activation_memory_budget，自动保存旧值以便恢复。"""
    global _PREVIOUS_BUDGET, _HAS_ACTIVE_OVERRIDE
    import torch._functorch.config as cfg
    try:
        if not _HAS_ACTIVE_OVERRIDE:
            _PREVIOUS_BUDGET = cfg.activation_memory_budget
        cfg.activation_memory_budget = budget
        _HAS_ACTIVE_OVERRIDE = True
        return True
    except Exception:
        return False

def clear_memory_budget():
    """恢复 activation_memory_budget 的原始值。"""
    global _PREVIOUS_BUDGET, _HAS_ACTIVE_OVERRIDE
    if not _HAS_ACTIVE_OVERRIDE:
        return
    import torch._functorch.config as cfg
    cfg.activation_memory_budget = _PREVIOUS_BUDGET
    _PREVIOUS_BUDGET = None
    _HAS_ACTIVE_OVERRIDE = False
```

用法（在 `ex3_simulation_accuracy.py` 中）：
```python
set_memory_budget(0.5)
compiled = torch.compile(model, backend="inductor")
# ... 训练 ...
clear_memory_budget()
```

`set_memory_budget(0.5)` 就是把 `cfg.activation_memory_budget` 设为 0.5，
在 `torch.compile` 时 Inductor 自动读取。

---

## 八、完整调用链

```
ex3: set_memory_budget(0.5)
  |
  v
cfg.activation_memory_budget = 0.5               [config.py:116]
  |
  v
torch.compile(model, backend="inductor")
  |
  v
Inductor compile_fx
  |
  v
min_cut_rematerialization_partition()             [partitioners.py:1703]
  |
  +-- memory_budget = config.activation_memory_budget     [:1819]
  |   (可被 node.meta["memory_budget"] 覆盖)              [:1820-1822]
  |
  +-- choose_saved_values_set(budget=0.5)                 [:1466]
  |     |
  |     +-- budget==0? --> 只保存 inputs (全重算)           [:1491]
  |     +-- budget==1? --> 标准 min-cut (最优运行时)        [:1500]
  |     |
  |     +-- 第一级 more_aggressive min-cut                 [:1520-1530]
  |     |     放松: far_apart + chains + materialized
  |     |     if mem_ratio < budget --> 返回
  |     |
  |     +-- 第二级 aggressive min-cut                      [:1532-1541]
  |     |     再放松: allowlist
  |     |     if mem_ratio < budget --> 返回
  |     |
  |     +-- 第三级 背包精确求解                              [:1579-1647]
  |           收集 banned 的 compute-intensive 节点
  |           每个节点: (memory, runtime)
  |           dp_knapsack(max_memory=budget)                [:1392-1406]
  |           选中的节点从 ban 名单移除
  |           最终 solve_min_cut(dont_ban=选中节点)
  |
  v
saved_values --> fw/bw 分割                                [:1834-1838]
```

---

## 九、实验验证（ex3 数据）

以下数据来自 `toolkit_examples/ex3_simulation_accuracy.py`（LLaMA 6L/512H, batch=8, seq=128, SGD）：

| 策略 | fwbw_peak | vs S05 baseline | step_ms |
|------|-----------|----------------|---------|
| S07 inductor(b=1.0) | 838 MB | -18.9% | 20 ms |
| S08 inductor(b=0.5) | 672 MB | **-34.8%** | 22 ms |
| S09 inductor(b=0.0) | 845 MB | -18.1% | 28 ms |

关键观察：
- **b=0.5 是最优平衡点**：比 b=1.0 多省 16% 内存，仅多 2ms
- **b=0.0 反而不如 b=0.5**：全重算导致 BW 重算压力过大，backward peak 反而更高
- 这验证了背包求解的价值：不是"越激进越好"，而是在约束下找最优解

---

## 参考

- PyTorch PR: [#126320](https://github.com/pytorch/pytorch/pull/126320) — `activation_memory_budget` 原始实现
- 源文件路径（torch2.6-gpu conda env）：
  - 配置: `torch/_functorch/config.py` (第 82-141 行)
  - 核心: `torch/_functorch/partitioners.py` (第 1220-1700 行)
  - Inductor 入口: `torch/_inductor/compile_fx.py`
