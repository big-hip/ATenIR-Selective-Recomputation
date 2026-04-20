# 02 — 技术研究与工业调研

> **文档定位**: 项目技术基础——PyTorch 源码分析、FakeTensor 实验、文献综述、工业界调研。
> 合并自: `.windsurf/plans/04-research-notes-99269f.md` + `.windsurf/plans/08-industry-survey-e2f24d.md`

---

## 一、PyTorch `partitioners` 源码分析

**源文件**: `torch._functorch.partitioners`

### 1.1 `OpTypes` 分类

PyTorch 将 ATen ops 分为以下类别（影响重计算决策）：

```python
class OpTypes:
    fusible_ops: Set[OpOverload]           # 可融合的逐元素 ops
    compute_intensive_ops: Set[OpOverload]  # 计算密集 ops (mm, bmm, conv, ...)
    random_ops: Set[OpOverload]            # 随机 ops (dropout, ...)
    view_ops: Set[OpOverload]              # view ops (0 cost to recompute)
    recomputable_ops: Set[OpOverload]      # 可安全重计算 ops
```

view_ops 列表（来自 `get_default_op_list()`）:
```
squeeze, unsqueeze, alias, view, slice, t,
broadcast_in_dim, expand, as_strided, permute, select, split
```

> 注意：此列表不含 `getitem`（因为 getitem 不是 ATen op），但 getitem 解包也是 view 行为。

### 1.2 `_size_of(node)` — 节点大小估算

```python
def _size_of(node: fx.Node) -> int:
    def object_nbytes(x) -> int:
        if not isinstance(x, torch.Tensor):
            return 0
        return _tensor_nbytes(hint_int(x.numel(), fallback=4096), x.dtype)
    
    if "val" in node.meta:
        val = node.meta["val"]
        if isinstance(val, py_sym_types):
            return 1
        elif isinstance(val, (list, tuple)):
            return sum(object_nbytes(n) for n in val)
        elif isinstance(val, torch.Tensor):
            return object_nbytes(val)
    ...
```

关键点：
- 使用 `hint_int(x.numel(), fallback=4096)` 处理 SymInt
- 对 tuple/list 求和（不检查 storage aliasing，因为用途不同——这是 min-cut weight，不是内存仿真）

我们的 `val_bytes` 实现与 PyTorch 官方 `_size_of` 一致:
- 同样使用 `hint_int(fallback=4096)`
- 同样处理 SymInt/SymFloat/SymBool 返回 0
- 同样递归处理 tuple/list

### 1.3 `min_cut_rematerialization_partition`

核心流程：
1. 分类所有节点（`get_default_op_list()`）
2. 识别 recomputable 节点
3. 应用启发式过滤：
   - `ban_if_used_far_apart`: 远距离使用不重计算
   - `ban_if_long_fusible_chains`: 长融合链不重计算
   - `ban_if_materialized_backward`: BW 中已物化不重计算
   - `ban_if_not_in_allowlist`: 不在白名单不重计算
   - `ban_if_reduction`: reduction ops 不重计算
4. 构建 networkx 有向图
5. 调用 `nx.minimum_cut` 求解
6. 根据 cut 结果划分 FW/BW

### 1.4 `default_partition`

简单直通：FW 图原样返回，所有中间激活都保存。

### 1.5 `_extract_fwd_bwd_modules`

从 joint graph 中根据 partition 结果提取 FW 和 BW 两个独立 GraphModule。
FW 的 output 包含：用户输出 + saved activations。
BW 的 placeholder 包含：saved activations + tangent inputs。

### 1.6 关键配置项（`torch._functorch.config`）

| 配置 | 默认值 | 说明 |
|------|--------|------|
| `activation_memory_budget` | 1.0 | 0.0=最大内存节省, 1.0=最大运行时优化（详见 `06-activation-memory-budget.md`）|
| `aggressive_recomputation` | False | 放松所有 ban → 允许更多重算 |
| `ban_recompute_not_in_allowlist` | True | 仅允许白名单 op 重算 |
| `ban_recompute_reductions` | True | 禁止重算 reduction op |
| `recompute_views` | False | 是否允许重算 view op |
| `max_dist_from_bw` | — | 禁止重算离反向图太远的节点 |

> 实验验证见 `03-experiments.md` §四 和 §八.3

---

## 二、FakeTensor 与 Storage Aliasing

### 2.1 Storage Aliasing 检测

**实验**: 在 FakeTensorMode 下测试各种 view 操作的 storage 共享情况。

```python
from torch._subclasses.fake_tensor import FakeTensorMode
with FakeTensorMode() as fm:
    x = fm.from_tensor(torch.randn(4, 4))
    y = x.view(2, 8)
    z = x + 1
    print(x.untyped_storage()._cdata)  # e.g., 944330960
    print(y.untyped_storage()._cdata)  # 同上 (view)
    print(z.untyped_storage()._cdata)  # 不同 (alloc)
```

**结论**:
- `data_ptr()` 对 FakeTensor 全返回 0 → **不可用**
- `untyped_storage()._cdata` 正确反映 storage 共享 → **推荐方案**
- `_base` 属性在某些情况下不设置（如 `reshape` 对已 contiguous 的 tensor）

### 2.2 View 操作在 FX 图中的分布（GPT-2 实测）

配置: n_embd=64, n_layer=1, n_head=1, batch=1, seq=16

```
=== VIEW OPS (storage aliasing) ===
  aten.view.default: 21
  aten.permute.default: 4
  aten.expand.default: 4
  aten.slice.Tensor: 4
  aten.unsqueeze.default: 1
  aten.transpose.int: 1
  aten.t.default: 1
  aten._unsafe_view.default: 1
Total: 37

=== ALLOC OPS (new storage, 当前方法误判) ===
  <built-in function getitem>: 20  ← 实际是 view！
  aten.add.Tensor: 5
  aten.addmm.default: 4
  aten.mul.Tensor: 4
  aten.embedding.default: 2
  aten.bmm.default: 2
  ...
Total: 47
```

### 2.3 getitem 节点存储共享验证

所有 getitem 节点与其父节点（native_layer_norm, native_dropout, split 等的 tuple 输出）共享存储：

```
getitem: getitem[0] from native_dropout, shared=True
getitem_1: getitem[1] from native_dropout, shared=True
getitem_2: getitem[0] from native_layer_norm, shared=True
getitem_3: getitem[1] from native_layer_norm, shared=True
...（全部 shared=True）
```

### 2.4 定量影响

GPT-2 (n_embd=256, n_layer=4, batch=4, seq=128):
```
Total alloc bytes (current): 265.9 MB
Missed views (getitem):       22.2 MB (8.3%)
Correct alloc bytes:         243.7 MB
```

> 使用 `_cdata` storage aliasing 检测修正了 **4.8%** 的过估。详见 `04-dev-log.md` Phase 1。

---

## 三、`estimate_from_graphs` 公式验证

### 3.1 BW Placeholder 组成

GPT-2 (n_embd=256, n_layer=4, batch=4, seq=128):
```
BW placeholders: 46
  tangent inputs:    3   (107.1 MB)
  saved activations: 43  (134.6 MB)
FW output inputs:    (241.7 MB)  ← 包含 primals(model params) + saved acts
```

### 3.2 重复计算验证

旧公式: `act_peak = max(fw_peak, saved_act + bw_peak)`
- `bw_peak` 通过 `estimate_graph_peak(bw_gm)` 计算
- BW 图 placeholder 按序计入 `current`，包含 saved activations
- 因此 `bw_peak` 已包含 saved_act
- `saved_act + bw_peak` 重复计算了 ~134.6 MB

正确公式: `act_peak = max(fw_peak, bw_peak)`

### 3.3 三阶段峰值模型

```
训练步骤    | 常驻内存                  | 瞬态内存
FW 阶段     | params + optim_states    | FW activations (peak = fw_peak)
BW 阶段     | params + optim_states    | saved_act + BW temps (peak = bw_peak)
Opt 阶段    | params + optim_states    | gradients
```

v4.2 四峰值体系（Phase 0 已实现）：
```python
fw_peak  = static_base + fw_graph_peak
bw_peak  = static_base + grad_bytes + bw_graph_peak
opt_peak = static_base + grad_bytes + opt_temp
fwbw_peak = max(fw_peak, bw_peak)
true_peak = max(fw_peak, bw_peak, opt_peak)
```

---

## 四、CUDACachingAllocator 行为

> 基于 Zach DeVito 文档 + `CUDACachingAllocator.cpp` 源码 + xMem 论文。

### 4.1 核心行为

```
malloc(size):
  size = round_up_512(size)                    # 512B 对齐
  pool = size < 1MB ? small_pool : large_pool  # 双池
  block = best_fit(pool, size)                 # 最佳匹配空闲块
  if not found:
    cudaMalloc(segment_size)                   # 新分配 segment
    segment_size = 2MB (small) / 20MB (large)
  block = maybe_split(block, size)             # 大块拆分
  return block

free(block):
  mark_free(block)
  merge_with_neighbors(block)                  # 合并相邻空闲块
  # 不调用 cudaFree，缓存复用
```

### 4.2 精度差距来源

| 来源 | 影响 | L2 是否覆盖 |
|------|------|------------|
| 512B 对齐浪费 | 小 tensor 影响大 | ✅ `(nb + 511) & ~511` |
| Segment 粒度浪费 | 2MB/20MB 最小分配 | ❌ 需 L3 BFC |
| 碎片化 | free 后空洞 | ❌ 需 L3 BFC |
| 执行顺序差异 | 拓扑排序 vs 实际调度 | ❌ 需 LPMF |
| Caching 效应 | free 不真释放 | ❌ peak_allocated vs peak_reserved |

> 实验验证（`03-experiments.md` §八.1）确认：warmup 后碎片化不是主要误差来源。

### 4.3 各级仿真预期 MRE

| Level | 方法 | 预期 MRE | 实测 MRE | 对标 |
|-------|------|---------|---------|------|
| **L1** | Config 公式推导 | ~20-30% | 44.7%→B12修后降低 | DNNMem |
| **L2** | FX 图事件驱动 + 512B align | ~10-15% | **6.9%** (B10修后) | Inductor |
| L3 | + BFC Simulator | ~5-8% | 待实现 | LLMem |
| L4 | CPU 执行 + BFC 重放 | ~3-5% | 待实现 | xMem |

---

## 五、文献调研摘要

### 5.1 xMem (Middleware 2025, SOTA)

- **MRE**: ~3-5%（vs DNNMem 降低 91%）
- **方法**: CPU profiler trace → Analyzer (alloc/free events) → Orchestrator (修正 CPU→GPU 语义: 参数 persistent, 梯度延伸到 zero_grad) → Simulator (双层 BFC 仿真)
- **关键创新**: Orchestrator 语义修正使 CPU profiling 可映射到 GPU 行为
- **核心洞察**（与本项目方法论一致）：CPU 和 GPU 执行相同训练逻辑，产生一致的 tensor 集合；差异仅在于 low-level 实现和 allocator 行为

### 5.2 DNNMem

- **方法**: 静态图分析 + 框架级 allocator 仿真
- **MRE**: ~10-30%（论文报告 19.12%）
- **限制**: 不支持动态行为、optimizer

### 5.3 LLMem (IJCAI 2024)

- **方法**: GPU 实际执行几步 + 外推，结构感知公式针对 Transformer 优化
- **MRE**: ~5-10%
- **特点**: **需要 GPU**，代码侵入式

### 5.4 Horus

- **方法**: 静态图分析
- **MRE**: ~10-30%
- **适用**: 中等规模模型

### 5.5 Inductor 内存规划

- **模块**: `torch._inductor.memory.estimate_peak_memory`
- **方法**: 与本项目 L2 几乎相同（事件驱动前缀和）
- **额外**: LPMF 拓扑排序优化执行顺序

### 5.6 PyTorch FakeTensor 定位

- 是**图捕获工具**（shape propagation），不是内存估算器
- PyTorch Issue #136446 正推进 Inductor + FakeTensor 内存估算
- 我们的方案：FakeTensor 提供 meta (shape/dtype/storage)，自己做内存仿真

---

## 六、工业界三大技术路线

### 路线 A：手写 Eager + 手动优化（Megatron-LM / DeepSpeed）

**代表**：NVIDIA Megatron-LM, Microsoft DeepSpeed

**做法**：
- **不使用 torch.compile**，完全在 eager 模式下手写分布式并行 + 内存优化
- 内存估算采用**公式法**：基于参数量 × 固定系数（如 `18 * params / n_gpus`）
- DeepSpeed 提供 `estimate_zero2/3_model_states_mem_needs` API：只估算 params + optimizer + gradients
- **activation memory 被完全忽略**或留给用户经验判断

**DeepSpeed 内存估算公式（ZeRO-2 为例）**：
```
GPU_memory = 4 * params + 16 * params / n_gpus  (无 offload)
GPU_memory = 2 * params                         (optimizer offload 到 CPU)
```
- 完全不考虑 activation、临时 tensor、allocator 行为

**优点**：极致性能、完全可控
**缺点**：支持新模型成本极高、activation 估算缺失、可扩展性差

### 路线 B：torch.compile + AOT 编译（TorchTitan / PyTorch 官方方向）

**代表**：Meta TorchTitan, PyTorch 2.x 官方生态

**做法**：
- 使用 `torch.compile` 将 eager 模型编译为 FX graph
- AOT Autograd 生成 forward/backward 联合图，min-cut partitioner 自动决定 recomputation
- PyTorch 2.4+ 提供 `activation_memory_budget` API 自动化 SAC
- 支持新模型 = **只要 Dynamo 能 trace 就行**

**TorchTitan 的关键实践**：
- 使用 **regional compilation**：只对 TransformerBlock 做 torch.compile，避免全模型 trace
- 好处：(1) 每个 region 得到完整图无 graph break (2) 相同 block 只编译一次
- 与 FSDP2、TP、AsyncTP、Float8 全部 compose
- **这正是我们 toolkit 的对标方向**

**ezyang (PyTorch 核心开发) 2025.08 总结**：
> "torch.compile 在训练上 speedup 1.5-2x 是典型值。它还支持全局内存优化如 automatic activation checkpointing。"

**优点**：模型无关、自动优化、生态完整
**缺点**：graph break 兼容性、编译时间、某些模型不可 fullgraph

### 路线 C：CPU 仿真 / 静态分析（学术研究方向）

**代表**：xMem (2025), LLMem (IJCAI 2024), DNNMem (2020)

| 方法 | 技术路径 | 精度 (MRE) | 局限 |
|------|---------|-----------|------|
| DNNMem | 静态图分析 + 框架级 allocator 仿真 | 19.12% | 不支持动态行为、optimizer |
| LLMem | GPU 实际执行几步 + 外推 | ~5% | **需要 GPU**，代码侵入式 |
| xMem | CPU 执行 trace + 双层 allocator 仿真 | **降低 91% MRE** vs DNNMem | 需要 CPU profiling 3 步 |

---

## 七、Graph Break 与模型兼容性

### 7.1 实测的 graph break 情况

| 模型 | transformers 4.35.2 | graph break | 根因 |
|------|-------------------|-------------|------|
| GPT-2 | ✅ fullgraph | 0 | — |
| LLaMA | ✅ fullgraph | 0 | — |
| Mistral | ✅ fullgraph | 0 | — |
| OPT | ❌ 4 graphs | 3 | 动态控制流 |
| Bloom | ❌ guard fail | — | OpaqueUnaryFn_log2 (ALiBi) |
| Falcon | ❌ 7 graphs | 6 | 多处控制流 |
| GPT-NeoX | ❌ deepcopy fail | — | meta/cpu 设备冲突 |

**结论**：不是所有模型都能做完整 AOT capture。这是整个行业共同面临的问题。

### 7.2 TorchTitan 的解决方案：Regional Compilation

```python
for layer in model.layers:
    layer = torch.compile(layer)  # 只编译单个 TransformerBlock
```

好处：TransformerBlock 内部通常无 graph break；相同结构只编译一次；与 FSDP2/TP 完美组合。

**启示**：我们的 capture 可以改为 block-level capture，全模型内存 = block 内存 × N_layers + embedding + lm_head。

---

## 八、本项目在工业界的定位

```
精度低                                              精度高
├── DeepSpeed 公式法 ──── DNNMem ──── 我们(ATen IR L2) ──── xMem ──── LLMem
│   (不算 activation)  (MRE ~19%)   (MRE 6.9%)        (CPU trace)  (需 GPU)
│
│   不需要 GPU ──────────────────────────────────────── 需要 GPU ───
```

### 独特价值

1. **零 GPU 成本**：capture 只需一次 torch.compile，之后纯 CPU 分析
2. **op 级粒度**：比 DeepSpeed/DNNMem 的粗公式更精确
3. **与 PyTorch 2.x 生态一致**：基于 AOT Autograd + FX Graph，直接复用 min-cut partitioner 基础设施
4. **selective recomputation 决策**：不只是估算内存，还能指导优化策略

### 现实挑战

1. **模型兼容性**：不是所有模型都能 fullgraph AOT capture → 解决方案：regional compilation
2. **allocator 仿真精度**：静态分析无法完美模拟 CUDACachingAllocator 碎片行为
3. **分布式训练**：FSDP/TP 下的内存模式更复杂 → 短期方案：单卡分析 + 公式缩放
4. **loss_fn 一致性**：不同 loss 导致 BW 图结构不同（已验证差异 30%）

---

## 九、PyTorch API 完整性验证

所有 toolkit 依赖的 PyTorch API 在 2.6.0 下验证通过：

| API | 参数 | 状态 |
|-----|------|------|
| `aot_module_simplified` | mod, args, fw_compiler, bw_compiler, partition_fn, ... | ✅ |
| `min_cut_rematerialization_partition` | joint_module, _joint_inputs, compiler, num_fwd_outputs | ✅ |
| `create_selective_checkpoint_contexts` | policy_fn_or_list, allow_cache_entry_mutation | ✅ |
| `hint_int(42, fallback=0)` | — | ✅ → 42 |
| Memory stats keys | `allocated_bytes.all.peak`, `reserved_bytes.all.peak` | ✅ |
| `_record_memory_history`, `_dump_snapshot` | — | ✅ |
| `torch.compile` | fullgraph, dynamic, backend, mode | ✅ |
| `checkpoint` with `context_fn` | SAC support | ✅ |

---

## 十、环境信息

```
conda env: torch2.6-gpu
Python: 3.x
PyTorch: 2.6.0
CUDA: 12.4
transformers: 4.35.2
matplotlib: 3.10.6
pandas: 2.3.3
tabulate: 0.10.0
networkx: installed (PyTorch dependency)
gradio: 3.24 (via conda-forge)
```

无外网访问，所有模型用 `from_config` 离线创建。

---

## 参考资料

1. [ezyang: State of torch.compile for training (Aug 2025)](https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025/)
2. [TorchTitan: One-stop PyTorch native solution for production ready LLM pre-training (2024)](https://arxiv.org/html/2410.06511v1)
3. [PyTorch Blog: Current and New Activation Checkpointing Techniques](https://pytorch.org/blog/activation-checkpointing-techniques/)
4. [xMem: CPU-Based Approach for Accurate GPU Memory Estimation (2025)](https://arxiv.org/html/2510.21048v1)
5. [LLMem: Estimating GPU Memory Usage for Fine-Tuning LLMs (IJCAI 2024)](https://arxiv.org/pdf/2404.10933)
6. [DeepSpeed Memory Requirements Documentation](https://deepspeed.readthedocs.io/en/latest/memory.html)
7. [dev-discuss: Min-cut optimal recomputation with AOTAutograd](https://dev-discuss.pytorch.org/t/min-cut-optimal-recomputation-i-e-activation-checkpointing-with-aotautograd/467)
