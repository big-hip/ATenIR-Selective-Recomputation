# Inductor 后端的 L2 级内存分析：可行性调研

> **作者**: Cascade (AI 辅助)  
> **日期**: 2025-04-17  
> **环境**: PyTorch 2.6.0 + CUDA 12.4, LLaMA (2L/512H), batch=4, seq=128, SGD

---

## 一、问题背景

在 `ex5_simulation_vs_runtime.py` 中，inductor 编译的策略（S07-S09, S11-S12）被标注为
**"L2 不可仿真"**，原因是：

> inductor 对 FX 图进行了深度优化（算子融合、Triton codegen、内存规划），
> 与 `aot_eager` 捕获的 ATen 级 FX 图结构完全不同。

**核心问题**：对 inductor 编译的模型，能否进行 L2 级（图遍历）的内存分析？如何做？

---

## 二、Inductor 编译管线全景

```
用户模型 (nn.Module)
    │
    ▼
┌─────────────────────────────────────────────┐
│  TorchDynamo (torch._dynamo)                │
│  · Python 字节码分析 → 捕获 FX 图            │
│  · 输出: GraphModule (高层 ATen ops)         │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│  Pre-Grad Passes (_recursive_pre_grad_passes)│
│  · 模式匹配优化 (pattern_matcher)            │
│  · 算子分解 (decomposition)                  │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│  AOTAutograd (aot_module_simplified)         │
│  · 联合图 (joint graph) 构建                 │
│  · FW/BW 分区 (min_cut_rematerialization)    │
│  · 输出: fw_gm, bw_gm (ATen 级 FX 图)       │
│  ★ 此处可截取 → 即 "post-AOT" 图             │
└─────────────────┬───────────────────────────┘
                  │
         ┌───────┴───────┐
         │               │
    fw_compiler     bw_compiler
         │               │
         ▼               ▼
┌─────────────────────────────────────────────┐
│  compile_fx_inner (inductor 编译入口)         │  ← ★ 截取点 A
│  ┌─────────────────────────────────────────┐ │
│  │  Joint Graph Passes                     │ │
│  │  · _recursive_joint_graph_passes(gm)    │ │
│  │  · 联合图级别的模式匹配优化              │ │
│  └─────────────────┬───────────────────────┘ │
│                    ▼                         │
│  ┌─────────────────────────────────────────┐ │
│  │  Post-Grad Passes                       │ │  ← ★ 截取点 B
│  │  · view_to_reshape (view→reshape)       │ │
│  │  · FakeTensorProp (填充 meta['val'])     │ │
│  │  · post_grad_passes:                    │ │
│  │    - group_batch_fusion                 │ │
│  │    - pattern_matcher (算子融合模式)      │ │
│  │    - reinplace_inplaceable_ops          │ │
│  │    - stable_topological_sort            │ │
│  │  · 输出: 优化后的 ATen 级 FX 图          │ │
│  └─────────────────┬───────────────────────┘ │
│                    ▼                         │
│  ┌─────────────────────────────────────────┐ │
│  │  GraphLowering (FX → Inductor IR)       │ │
│  │  · 每个 FX 节点 → IR Buffer/Operation   │ │
│  │  · 循环分析、索引表达式                  │ │
│  └─────────────────┬───────────────────────┘ │
│                    ▼                         │
│  ┌─────────────────────────────────────────┐ │
│  │  Scheduler (调度 + 融合)                 │ │
│  │  · 创建 SchedulerNode / SchedulerBuffer │ │
│  │  · 算子融合 (fuse_nodes)                 │ │
│  │  · 内存排序 (reorder_for_peak_memory)    │ │
│  │  · 内存估算 (estimate_peak_memory)       │ │  ← ★ inductor 内置内存分析
│  └─────────────────┬───────────────────────┘ │
│                    ▼                         │
│  ┌─────────────────────────────────────────┐ │
│  │  Codegen (Triton / C++ 代码生成)         │ │
│  │  · Triton kernel 代码                    │ │
│  │  · Wrapper code (调用序列)               │ │
│  └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

**关键观察**：从 Dynamo 到最终的 Triton kernel，图经历了 **至少 5 层变换**。
每一层都可能改变内存行为。

---

## 三、实验：截取 Inductor Post-Grad FX 图

### 3.1 截取方法

通过替换 `compile_fx` 的 `inner_compile` 参数，在 `compile_fx_inner` 被调用时
（此时 FX 图已经过 joint_graph_passes + post_grad_passes）截取 FW/BW 图的深拷贝：

```python
from torch._inductor.compile_fx import compile_fx, compile_fx_inner as orig_inner

captured = {}
def my_inner(gm, example_inputs, **kwargs):
    key = 'bw' if kwargs.get('is_backward', False) else 'fw'
    captured[key] = copy.deepcopy(gm)
    return orig_inner(gm, example_inputs, **kwargs)

compiled = torch.compile(model,
    backend=lambda gm, ei: compile_fx(gm, ei, inner_compile=my_inner),
    dynamic=True)
```

### 3.2 截取结果

| 指标 | aot_eager FW | inductor FW | aot_eager BW | inductor BW |
|------|-------------|-------------|-------------|-------------|
| 总节点数 | 280 | 300 | 481 | 412 |
| view 节点数 | 141 | 137 | 187 | 113 |
| 分配节点数 | 109 | 133 | 208 | 227 |
| 总分配量 | 278.3 MB | 416.4 MB | 649.6 MB | 1.1 GB |
| output 参数数 | 85 | 71 | 21 | 21 |
| 有 FakeTensor meta | 279/280 | 299/300 | 480/481 | 411/412 |

### 3.3 算子差异

**Inductor FW 独有的算子**（aot_eager 没有）：

| 算子 | 数量 | 含义 |
|------|------|------|
| `permute` | 46 | 替代 `t` + `transpose`（inductor 统一用 permute） |
| `amax` / `sub` / `exp` / `sum` | 各 3-5 | `_softmax` 被分解为原子操作 |
| `sigmoid` / `log` | 2, 1 | `silu` 和 `_log_softmax` 被分解 |
| `where` / `ne` | 3, 1 | `masked_fill` 被分解 |
| `prims.iota` / `prims.convert_element_type` | 1, 1 | `arange` 被分解为 prims |

**aot_eager FW 独有的算子**（inductor 分解掉了）：

| 算子 | 被分解为 |
|------|---------|
| `_softmax` | `amax` → `sub` → `exp` → `sum` → `div` |
| `silu` | `sigmoid` → `mul` |
| `_log_softmax` + `nll_loss_forward` | `log` → `gather` → `ne` → `where` |
| `t` / `transpose` / `_unsafe_view` | 统一为 `permute` / `view` |

**关键发现**：inductor 的 **decomposition** 将高层复合算子分解为更细粒度的原子操作，
导致：
- 节点数增加（FW: 280→300, 虽然 BW 减少 481→412 因为去掉了 detach 等）
- **中间 tensor 数量增加**（FW 分配节点 109→133）
- **总分配量显著增大**（FW: 278.3→416.4 MB, BW: 649.6→1.1 GB）

### 3.4 L2 估算器应用于 inductor post-grad 图

直接将现有的 `estimate_graph_peak()` 应用于截取的 inductor post-grad FX 图：

| 阶段 | L2 估算 | 运行时实测 | MRE | 方向 |
|------|---------|-----------|-----|------|
| FW graph_peak | 377.2 MB | — | — | — |
| BW graph_peak | 377.2 MB | — | — | — |
| true_peak | 526.2 MB | 759.3 MB | **30.7%** | **低估** |

对比 aot_eager 的 L2 对 inductor 运行时的估算：MRE = 36.5%。

**结论**：在 inductor post-grad 图上跑 L2 估算，MRE 从 36.5% 降到了 30.7%，
有一定改善但仍然很大。**根本原因**是 post-grad 图和实际执行之间还有巨大差距。

---

## 四、为什么 Post-Grad FX 图的 L2 估算不准确？

### 4.1 算子融合（Kernel Fusion）—— 最大误差来源

Inductor 的 Scheduler 会将多个 ATen 算子融合为单个 Triton kernel：

```
Post-grad FX 图 (ATen 级):           Scheduler 融合后:
  amax(x)  →  alloc buf0              ┐
  sub(x, buf0) → alloc buf1           │  fused_kernel_0(x) → alloc buf3
  exp(buf1) → alloc buf2              │  (buf0, buf1, buf2 变成寄存器临时变量,
  sum(buf2) → alloc buf3              ┘   不需要实际 GPU 内存分配)
```

- post-grad 图认为 `buf0`, `buf1`, `buf2` 各需要独立分配 → **高估中间内存**
- 实际执行中，融合后只需要 `buf3` 的输出分配 → **中间值在寄存器/共享内存中计算**
- 但在某些情况下，融合 kernel 可能需要 **workspace buffer** → 额外分配

**量化影响**：softmax 分解产生 4 个中间 tensor（amax, sub, exp, sum），融合后只
需要 1 个输出。对于 shape `[4, 8, 128, 128]`（attention scores），每个中间 tensor
= 4×8×128×128×4 = 2 MB，融合节省约 6 MB × 每层 2 次 = 12 MB（2 层模型）。

### 4.2 In-place 优化（reinplace_inplaceable_ops）

Post-grad 的最后一步 `reinplace_inplaceable_ops` 将部分 out-of-place 操作转为
in-place，这意味着：
- FX 图中的某些 `call_function` 节点实际复用输入 tensor 的内存
- 但 `meta['val']` 仍然是独立的 FakeTensor → L2 估算器会误认为需要新分配

### 4.3 Buffer Reuse（内存规划）

Inductor 的 `memory_planning` 选项（默认关闭，但 `reorder_for_peak_memory` 默认开启）
允许 Scheduler 重排节点执行顺序以降低峰值内存：

```python
# torch._inductor.config
reorder_for_peak_memory = True   # 默认开启
memory_planning = False          # 默认关闭（实验性）
```

当 `reorder_for_peak_memory=True` 时，Scheduler 尝试三种拓扑排序策略
（LPMF、BFS、DFS），选择峰值最低的排列。这一排序在 post-grad FX 图层面不可见。

### 4.4 Triton Kernel Workspace

某些 Triton kernel（特别是 matmul/GEMM 相关）可能需要额外的 workspace buffer，
这些在 FX 图层面完全不可见，贡献额外的运行时内存开销。

### 4.5 CUDA Caching Allocator

inductor 生成的代码直接使用 `torch.empty()` 分配，经过 CUDA Caching Allocator。
Allocator 的 block 分裂/合并、碎片化行为在任何静态图分析中都无法精确建模。

---

## 五、Inductor 内置的内存估算机制

### 5.1 `torch._inductor.memory` 模块

Inductor 在 `memory.py` 中实现了自己的内存估算系统，**作用于 Scheduler 层级**
（融合后的节点），而非 FX 图层级：

```python
# torch/_inductor/memory.py

def estimate_peak_memory(
    nodes: List[BaseSchedulerNode],          # 融合后的调度节点
    name_to_freeable_input_buf: Dict[...],   # 可释放的输入 buffer
    graph_outputs: Set[str],                 # 图输出（不可释放）
) -> Tuple[int, List[int]]:
    """基于 SchedulerNode 的执行顺序，估算峰值内存"""
```

**核心机制**：
1. 为每个 `SchedulerBuffer` 计算 `(size_alloc, size_free)` —— 考虑了 `MultiOutputLayout`
2. 使用 `V.graph.sizevars.size_hint()` 计算 buffer 大小（非 FakeTensor）
3. 追踪每个 buffer 的 start_step 和 end_step
4. 增量扫描计算峰值

### 5.2 内存排序优化

```python
def reorder_for_peak_memory(nodes, ...):
    """尝试多种拓扑排序，选择峰值最低的"""
    methods = [
        topological_sort_lpmf,   # Least Peak Memory First (贪心)
        topological_sort_bfs,    # BFS (FIFO 释放)
        topological_sort_dfs,    # DFS (最小读写优先)
    ]
    # 对每种排序计算 estimate_peak_memory，选最优
```

LPMF 算法来自论文 *"Buffer memory optimization for video codec application
modeled in Simulink"* (DAC 2006)。

### 5.3 为什么不能直接使用 inductor 内置估算？

| 限制 | 说明 |
|------|------|
| **时机太晚** | 必须完成 GraphLowering + Scheduler 构建才能调用，此时已经过完整编译 |
| **无法脱离编译** | `estimate_peak_memory` 依赖 `SchedulerNode` / `SchedulerBuffer` 对象，这些只在编译过程中存在 |
| **只分析单个图** | 只分析 FW 或 BW 图的激活内存，不包含 param/grad/optim |
| **不含外部内存** | Triton workspace、cuBLAS handle 等不在其模型内 |

---

## 六、可行方案评估

### 方案 A：在 Post-Grad FX 图上改进 L2 估算

**思路**：截取 `compile_fx_inner` 入口处的 post-grad FX 图，用改进的 `estimate_graph_peak` 分析。

**优点**：
- FX 图仍然有 `meta['val']` FakeTensor，我们的工具链可以直接工作
- 不需要完成完整编译，截取成本低
- 与现有 L2 框架兼容

**缺点**：
- 不反映融合 → **系统性高估中间内存**（实验: 总分配量 416.4 MB vs aot_eager 278.3 MB）
- 但运行时峰值却更高（759.3 MB vs ~480 MB for aot_eager）→ 低估实际峰值
- 低估原因：inductor 运行时有 Triton workspace、codegen overhead、不同的 allocator 行为
- **MRE 约 30%**，难以进一步降低

**工程量**：低（1-2 天），但精度有限。

**适用场景**：粗略估算，误差容忍 >25%。

### 方案 B：Hook Inductor Scheduler 的 `estimate_peak_memory`

**思路**：在编译过程中 hook `GraphLowering.codegen()`，提取 Scheduler 的
`estimate_peak_memory` 结果。

**优点**：
- 利用 inductor 自身的融合感知内存估算
- 理论上更准确（考虑了 kernel fusion、buffer reuse）

**缺点**：
- 必须完成完整编译（包括 Triton codegen），耗时 15-25 秒
- 深度依赖 inductor 内部 API（`SchedulerNode`, `SchedulerBuffer`），极不稳定
- 只得到单图的激活峰值，仍需外部拼装 param/grad/optim
- inductor 内部 API 跨版本变化大，维护成本高

**工程量**：中（3-5 天），但维护成本高。

**适用场景**：需要精确分析 inductor 编译策略的内存行为。

### 方案 C：编译后运行时 Profiling（当前方案）

**思路**：不做静态分析，直接运行编译后的模型并用 `measure_phased` 测量。

**优点**：
- 100% 准确反映实际行为
- 无需理解 inductor 内部机制
- 与所有编译策略兼容

**缺点**：
- 需要 GPU 实际运行
- 需要足够的 GPU 内存（不能预测 OOM）
- 首次编译耗时长（15-25 秒）

**工程量**：零（已实现）。

**适用场景**：绝大多数实际使用场景。

### 方案 D：混合方案 — 分层估算

**思路**：
1. 在 post-grad FX 图上做 L2 估算（方案 A），得到**上界估计**
2. 应用**融合折扣因子**（fusion discount）修正中间 tensor 的过度计算
3. 加上 **dark memory 修正**（CUDA context + Triton workspace）

```python
def estimate_inductor_peak(fw_gm_post_grad, bw_gm_post_grad, model, **kwargs):
    # Step 1: 基础 L2 估算
    raw_fw = estimate_graph_peak(fw_gm_post_grad, pin_output_inputs=True)
    raw_bw = estimate_graph_peak(bw_gm_post_grad, pin_output_inputs=True)
    
    # Step 2: 融合折扣 — pointwise ops 被融合后中间 tensor 不实际分配
    # 经验系数: 约 0.6-0.8 (取决于模型中 pointwise 比例)
    fusion_discount = 0.7
    fw_adjusted = raw_fw['peak_bytes'] * fusion_discount
    bw_adjusted = raw_bw['peak_bytes'] * fusion_discount
    
    # Step 3: dark memory 修正
    dark_memory = 20 * 1024 * 1024  # ~20 MB (CUDA context + Triton workspace)
    
    return max(fw_adjusted, bw_adjusted) + static_base + dark_memory
```

**优点**：不需要完整编译，精度可调
**缺点**：融合折扣因子需要标定，不同模型/配置可能不同

**工程量**：中（3-5 天，含标定）。

---

## 七、与现有研究的对比

### 7.1 学术界方法

| 方法 | 年份 | 核心思路 | 是否处理编译器融合？ |
|------|------|---------|-------------------|
| **DNNMem** | 2020 | 框架无关的公式法 | ❌ 不考虑 |
| **SchedTune** | 2022 | 调度器级别分析 | ⚠️ 只考虑 TF/XLA 的融合 |
| **LLMem** | 2024 | LLM 特化公式 | ❌ 假设 eager 执行 |
| **xMem** | 2024 | CPU 端 FakeTensor 仿真 | ❌ 假设 eager 执行 |

**关键发现**：**没有现有工作**同时处理 PyTorch 2.x `torch.compile` + inductor 融合
场景下的静态内存估算。这是一个**开放问题**。

### 7.2 工业界实践

| 方案 | 使用者 | 方式 |
|------|--------|------|
| `activation_memory_budget` | PyTorch 官方 | 在 partitioner 层级估算，不考虑 inductor 融合 |
| `torch.cuda.memory._snapshot()` | 通用 | 运行时抓取，事后分析 |
| `torch._inductor.memory.estimate_peak_memory` | inductor 内部 | Scheduler 级别，仅用于排序优化 |
| `TORCH_TRACE` + profiling | Meta 内部 | 编译期 + 运行时联合分析 |

---

## 八、结论与建议

### 8.1 核心结论

1. **inductor post-grad FX 图可以被截取**，且保留了 `meta['val']` FakeTensor，
   我们的 `estimate_graph_peak` 可以直接运行。

2. **但精度很差**（MRE ~30%），根本原因是 **kernel fusion**：
   - post-grad 图中的 pointwise 中间 tensor 在融合后不实际分配
   - 但 Triton workspace / 编译开销又引入图中不可见的额外内存

3. **Inductor 自身有 Scheduler 级别的 `estimate_peak_memory`**，
   但只能在完整编译后才能使用，且 API 极不稳定。

4. **没有现有学术工作**解决了 `torch.compile` + inductor 场景下的
   静态内存估算问题。

### 8.2 当前建议

对于本项目（毕设）的定位：

| 策略类型 | 建议方案 | 理由 |
|---------|---------|------|
| `aot_eager` 系列 | L2 图遍历（现有方案） | MRE ~7%，精度足够 |
| `inductor` 系列 | **运行时实测**（方案 C） | 唯一可靠的方法 |
| 未来扩展 | 方案 D（混合）+ 融合系数标定 | 可作为后续研究课题 |

### 8.3 论文/毕设中的呈现建议

在论文中可以这样定位：

> 本工具的 L2 仿真引擎面向 **ATen 级 FX 图**（aot_eager 后端），
> MRE 约 7%。对于 inductor 后端，由于 kernel fusion 导致
> 图层级分析与实际执行之间存在语义鸿沟（semantic gap），
> 目前采用运行时分阶段测量作为补充。
> 这一语义鸿沟是 DL 编译器内存分析领域的**开放问题**，
> 现有学术工作（DNNMem, xMem, LLMem）均未涉及编译器融合场景。

---

## 九、附录：实验数据

### A. 图结构对比（LLaMA 2L/512H, batch=4, seq=128）

```
aot_eager FW: 280 nodes (view=141, alloc=109), total_alloc=278.3 MB
inductor FW: 300 nodes (view=137, alloc=133), total_alloc=416.4 MB
  → 分解导致分配节点 +22%, 总分配量 +49.6%

aot_eager BW: 481 nodes (view=187, alloc=208), total_alloc=649.6 MB
inductor BW: 412 nodes (view=113, alloc=227), total_alloc=1.1 GB
  → view 减少(detach 去除), 但分配节点 +9%, 总分配量 +69.3%
```

### B. L2 估算精度对比

```
目标: inductor 运行时 overall_peak = 759.3 MB

方法 1: aot_eager L2 估算    → 482.2 MB, MRE = 36.5% (低估)
方法 2: inductor post-grad L2 → 526.2 MB, MRE = 30.7% (低估)
方法 3: 运行时实测            → 759.3 MB, MRE = 0%    (精确)
```

### C. 算子分解映射表

| aot_eager 算子 | inductor 分解为 | 额外 tensor 数 |
|---------------|----------------|---------------|
| `_softmax` | `amax` → `sub` → `exp` → `sum` → `div` | +3 |
| `silu` | `sigmoid` → `mul` | +1 |
| `_log_softmax` | `amax` → `sub` → `exp` → `sum` → `log` | +3 |
| `nll_loss_forward` | `gather` → `ne` → `where` → `squeeze` | +3 |
| `t` / `transpose` | `permute` (统一) | 0 |
| `_unsafe_view` | `view` (统一) | 0 |
| `arange` | `prims.iota` | 0 |

### D. Inductor 内部内存分析 API

```python
# 核心文件: torch/_inductor/memory.py

# 1. buffer 大小计算
compute_size_for_scheduler_buffer(name_to_buf) → Dict[str, (alloc, free)]

# 2. 峰值估算 (基于执行顺序)
estimate_peak_memory(nodes, freeable_inputs, graph_outputs) → (peak, mem_trace)

# 3. 内存感知拓扑排序
reorder_for_peak_memory(nodes, ...) → reordered_nodes
  ├── topological_sort_lpmf()   # Least Peak Memory First
  ├── topological_sort_bfs()    # Breadth First (FIFO)
  └── topological_sort_dfs()    # Depth First (小优先)

# 配置开关:
torch._inductor.config.reorder_for_peak_memory = True   # 默认开启
torch._inductor.config.memory_planning = False           # 默认关闭
```

---

## 十、参考资料

1. PyTorch Inductor 源码: `torch/_inductor/compile_fx.py`, `memory.py`, `scheduler.py`
2. PyTorch Blog: "Current and New Activation Checkpointing Techniques in PyTorch" (2024)
3. Gao et al., "DNNMem: Estimating GPU Memory Consumption of Deep Learning Models" (ESEC/FSE 2020)
4. Kim et al., "LLMem: Estimating GPU Memory Usage for Fine-Tuning Pre-Trained LLMs" (IJCAI 2024)
5. xMem: "A CPU-Based Approach for Accurate Estimation of GPU Memory Usage" (2024)
6. DAC 2006: "Buffer memory optimization for video codec application modeled in Simulink"
7. PyTorch PR #142822: "[Inductor] Move peak memory pass and overlap pass to be run at the right place"
