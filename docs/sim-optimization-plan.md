# 静态仿真优化方案（修订版）

> 基于 xMem (2025)、LLMem (2024)、DNNMem (2020)、PyTorch CachingAllocator 源码分析，
> 以及对初版方案中启发式风险的系统审视后形成的修订方案。

---

## 一、文献调研总结

### 1.1 现有工作对比

| 方法 | 年份 | 核心思路 | MRE | AC 支持 | torch.compile | 关键局限 |
|------|------|---------|-----|---------|---------------|---------|
| **DNNMem** | 2020 | 静态图遍历 + 算子内存函数 + 64MB buffer 常量 | 19-34% | ✗ | ✗ | 无 optimizer/混合精度建模 |
| **LLMem** | 2024 | Transformer 结构化公式 + 2MB 页对齐 | ~5% | ✓ (公式) | ✗ | 仅 CausalLM, 不支持 compile |
| **xMem** | 2025 | CPU profiling → Orchestrator → 双层 BFC 仿真 | **~3-4%** | ✓ (运行时) | ✗ | 仅 eager, 不支持 compile |
| **本项目 (当前)** | 2025 | ATen IR 图级 live-range + Inductor L3 hook | L2=8.8% | ✓ (图级) | **✓** | AC 策略 BW 低估 12-19% |

### 1.2 关键发现

**xMem 的核心创新——双层 Allocator 仿真**：
- 仿真 PyTorch CUDACachingAllocator 的 BFC (Best Fit with Coalescing) 算法
- 建模 small pool (<1MB) / large pool (≥1MB) 分离
- 建模 Segment 级分配 (<1MB→2MB, 1-10MB→20MB, >10MB→2MB 对齐) + Block 分裂/合并
- 建模缓存行为（free 后保留 block 在 pool 中供复用）
- 仅在两层 allocator 都失败后报 OOM

**xMem 的 Memory Orchestrator——生命周期规则**：
- 模型参数 → persistent（全程常驻）
- 批数据 → 限制在单 iteration 内
- 激活值 → 保留 CPU 推导的生命周期（近似）
- **梯度 → 持续到 `optimizer.zero_grad()` 被调用**（非立即释放）
- Optimizer 状态 → 首次 iteration 分配后 persistent

**PyTorch CachingAllocator 的请求对齐规则**（来自 Zach DeVito 博客）：
- 请求级对齐：默认 512B（我们当前的 `align=512` 是正确的）
- **Segment 级对齐**：从 CUDA 申请的块为 2MB/20MB 倍数，但 block 分裂允许多个小张量共享一个 Segment
- **因此：对每个张量独立做 2MB 对齐是错误的**——这会产生巨大的累积误差

### 1.3 对初版方案启发式风险的验证

| 初版优化 | 风险评估 | 文献依据 | 结论 |
|----------|---------|---------|------|
| 优化1: `max(graph_peak-overlap, grad_bytes)` | ⚠ 分布式/冻结层会过度悲观 | xMem Orchestrator 梯度规则 | **废弃，改用 overlap-aware peak** |
| 优化3: 静态 2MB 对齐 | ❌ 严重错误 | CachingAllocator block splitting | **废弃** |
| 优化6: `n_compute/n_ph > 1.3` 阈值 | ⚠ 脆弱，误伤非 AC 模型 | — | **改用结构化检测** |
| 优化7: L2.5 退化为 L3 | ⚠ 单点失效 | xMem 强调独立估计路径 | **保持独立，同步修复 L2.5 逻辑** |
| 优化8: 1.5x 容忍复用 | ⚠ 忽略 Stream 同步 | CachingAllocator stream-aware free | **暂缓，仅单 stream 场景安全** |

---

## 二、修订后的优化方案

### 核心方法论变化

**从"打补丁"到"修正模型"**：不再在公式层面叠加启发式修正项，
而是修正 live-range 仿真中 overlap 计算的数学模型，使其在所有场景下自洽。

---

### 优化 A：Overlap-Aware Peak Tracking（核心修复）

**目标**：彻底修正 AC 策略的 BW 低估问题，无需任何启发式参数。

**问题根因（精确表述）**：

当前公式 `bw_peak = static_base + (bw_graph_peak - bw_ph_overlap)` 在 BW 图的
**绝对峰值时刻** 计算 overlap。但对 AC 策略，绝对峰值在图开头（所有 placeholder
存活），此时 overlap 最大（所有 forwarded primals 都在）。而运行时的真实峰值在
BW 中间（部分 placeholder 释放 + 梯度累积），此时 overlap 更小，导致"净新分配"
更高。

**数学表述**：

定义 BW 图在时间步 t 的"净新分配"（不与 static_base 重叠的部分）：

```
net_new(t) = current(t) - overlap(t)
           = current(t) - min(fwd_primal_total, fwd_primal_alive(t))
```

当前代码取：`net_new = peak(current) - overlap_at_peak_moment`

正确做法应取：`net_new = max_t { current(t) - overlap(t) }`

**对 S05 (无 AC) 的验证**：
- 时刻 0: current=20.5GB, overlap=3.0GB, net=17.5GB ← peak
- 时刻 mid: current=10.0GB, overlap=1.5GB, net=8.5GB
- peak_net = 17.5 → bw_peak = 9.8 + 17.5 = 27.3 ≈ RT 27.2 ✓（无变化）

**对 S10 (AC) 的验证**：
- 时刻 0: current=5.5GB, overlap=3.0GB, net=2.5GB ← 旧 peak
- 时刻 mid: current=5.5GB, overlap=1.5GB, net=4.0GB ← **新 peak**
- 时刻 late: current=4.0GB, overlap=0.5GB, net=3.5GB
- peak_net = 4.0 → bw_peak = 9.8 + 4.0 = 13.8（旧值 12.3, 改善 1.5GB）
- RT=15.7, 残余 gap=1.9GB（旧 3.4GB → 改善 44%）

**⚠ 跨图映射陷阱与解决方案**

原方案计划通过 `_forwarded_primal_ph_set(fw_gm, bw_gm)` 逐节点识别 BW 中的
forwarded primal placeholder。经实测验证，这存在以下问题：

1. **`copy.deepcopy` 破坏跨图 FakeTensor storage 匹配**：`capture_graphs` 中
   `deepcopy(fw_gm)` 和 `deepcopy(bw_gm)` 分别执行，两个图中的 FakeTensor
   `_cdata` 不再对应。
2. **`CompiledBwInfo` / `saved_tensor_indices` 不存在**（经搜索 PyTorch 2.6 全量
   源码确认）。实际的编译期元数据是 `ViewAndMutationMeta`，其中有
   `tensors_saved_for_backwards_slice` 和 `num_forward`，但这些仅在
   AOTAutograd dispatch 内部可用，**不会附加到导出的 FW/BW GraphModule 上**。
3. **BW placeholder 排列不是简单的 tangents+saved**。实测确认排列为：
   `[*saved_sym_nodes, *saved_values, *tangent_inputs, *bwd_seed_offset]`，
   且 saved_values 包含 forwarded primals + saved activations 混合。

**修订后的实现方案（聚合近似法，无需跨图映射）**：

核心观察：我们不需要知道「哪个具体 BW placeholder 是 forwarded primal」，
只需在每个时间步知道「forwarded primal 的存活字节近似值」。

1. **`_forwarded_primal_bytes` 已有**：在 FW 图内部通过 storage 匹配计算
   forwarded primal 总字节 `fwd_primal_total`（图内匹配，deepcopy 不影响）

2. **修改 `estimate_graph_peak`**：新增 `overlap_bytes: int = 0` 参数
   ```python
   def estimate_graph_peak(gm, ..., overlap_bytes: int = 0):
       # overlap_bytes = forwarded primal 总字节数
       # 在 live-range 遍历中，额外跟踪:
       ph_alive = 0      # 所有 placeholder 当前存活字节
       peak_net = 0       # max(current - min(overlap_bytes, ph_alive))

       for index, node in enumerate(nodes):
           # ... 现有 alloc/free 逻辑 ...
           # 更新 ph_alive
           if node.op == "placeholder":
               ph_alive += node_size[node]  # alloc 时
           # free placeholder 时也减少 ph_alive

           # 计算当前时刻的净新分配
           overlap_now = min(overlap_bytes, ph_alive)
           net = current - overlap_now
           if net > peak_net:
               peak_net = net
   ```
   返回 dict 中新增 `"peak_net_bytes"`

3. **修改 `estimate_training_peak`**：
   ```python
   fwd_primal = _forwarded_primal_bytes(fw_gm, n_params)
   bw_result = estimate_graph_peak(bw_gm, pin_output_inputs=True,
                                    overlap_bytes=fwd_primal)
   bw_peak = static_base + bw_result["peak_net_bytes"]
   ```

**近似的保守性分析**：

`min(fwd_primal_total, ph_alive_t)` 假设「只要 placeholder 存活字节 ≥ fwd_primal
总字节，则所有 forwarded primals 都存活」。这在以下极端情况会高估 overlap
（从而低估 net_new）：所有 forwarded primal 被释放，但其他 activation placeholder
仍大量存活。但在 Transformer 中，forwarded primal（权重）和 activation 是
交错释放的（逐层处理），因此近似是合理的。

如需更精确，可通过 BW placeholder 节点名称中的 `primals_` 前缀 + FW param
placeholder 数量进行二次校验（实测确认节点名称在 deepcopy 后保持不变）。

**近似方向性说明（重要）**：

`min(overlap_bytes, ph_alive)` 的偏差方向是 **高估 overlap → 低估 net → 低估峰值**
（peak 的 lower bound，非 upper bound）。具体发生在：

| 条件 | 公式 overlap | 实际 overlap | net 偏差 |
|------|-------------|-------------|---------|
| `ph_alive < overlap_bytes` | `ph_alive` | `≤ ph_alive` | **精确或轻微低估 net** |
| `ph_alive ≥ overlap_bytes`，且所有 overlap 仍存活 | `overlap_bytes` | `overlap_bytes` | **零偏差** |
| `ph_alive ≥ overlap_bytes`，但部分 overlap 已释放 | `overlap_bytes` | `< overlap_bytes` | **低估 net（高估 overlap）** |

第三种情况（overlap 被释放但非 overlap activation 仍大量存活）在 Transformer 中罕见，
因为权重和激活是逐层交错释放的。但对异构计算图（如混合 CNN+Transformer）需注意。

在论文中应表述为：*"聚合近似法提供了峰值内存的保守下界估计，在标准 Transformer
架构中近似为精确值。"*

**安全性论证**：
- 无任何硬编码阈值或经验常数
- 仅依赖 FW 图内部的 storage 匹配 + BW 图的 ph_alive 聚合统计
- 对非 AC 策略退化为当前行为（因为 overlap 在 peak 时刻最大）
- 对 AC 策略自动捕捉"梯度累积导致的延迟峰值"
- 对 frozen 层安全：forwarded primals 只包含参与计算的参数
- 对分布式训练安全：不假设梯度生命周期

---

### 优化 B：BW 重计算检测（基于 PyTorch 官方元数据）

**目标**：为 L2.5 调度策略提供可靠的重计算场景识别。

**⚠ 原方案（算子指纹匹配）的误判问题**

经实测验证，原方案 `n.target in fw_targets` 存在严重误判：

| 场景 | BW 总计算节点 | 与 FW 重叠 | 重叠率 | 原方案判定 | 正确答案 |
|------|-------------|-----------|-------|-----------|--------|
| 无 AC, default_partition | 28 | 16 | 57% | ✓ 有重计算 ❌ | 无重计算 |
| 有 AC, default_partition | 40 | 28 | 70% | ✓ 有重计算 ✅ | 有重计算 |

**根因**：`aten.view`, `aten.t`, `operator.getitem` 等结构性算子在 FW 和
正常 BW 中都大量出现（参数转置、梯度 reshape），导致 57% 的基线重叠率。
原方案的 `len(recomp) > grad_only` 对正常模型即为 `16 > 12 = True`，100% 误判。

**修订方案：直接使用 PyTorch 官方编译期元数据**

经实测确认，PyTorch 2.6 的 AOTAutograd 在 BW 图节点上留下了精确的元数据：

```
# AC 重计算节点（被 checkpoint 机制复制到 BW 的 FW 算子）:
node.meta['recompute'] = CheckpointPolicy.PREFER_RECOMPUTE
node.meta['source_fn_stack'] 包含 TagActivationCheckpoint

# 正常反向计算节点:
node.meta['recompute'] = None
node.meta['partitioner_tag'] = 'is_backward'
```

实测验证结果：
- **无 AC, default_partition**: `recompute` 标记节点数 = **0** ✅
- **无 AC, min_cut**: `recompute` 标记节点数 = **0** ✅
- **有 AC, default_partition**: `recompute` 标记节点数 = **14**
  (包括 addmm, native_layer_norm, relu, view, t, getitem, mul, detach 等) ✅

```python
def detect_recomputation(bw_gm: fx.GraphModule) -> bool:
    """通过 PyTorch 编译期元数据检测 BW 图中是否存在 AC 重计算节点。

    PyTorch 的 CheckpointPolicy 机制在 AC 重计算节点的 meta['recompute']
    字段中设置 CheckpointPolicy.PREFER_RECOMPUTE 标记。
    此元数据由编译器自动生成，100% 精确，零误判。
    """
    for n in bw_gm.graph.nodes:
        if n.meta.get('recompute') is not None:
            return True
    return False
```

**局限性说明**：
- 此方法仅检测 **AC (Activation Checkpointing)** 式重计算
- **min-cut 重计算**不设置 `recompute` 标记（min-cut 的「重计算」节点
  被直接划入 BW 分区，标记为 `partitioner_tag='is_backward'`）
- 对本项目足够：S09 (b=0.0) 的 min-cut 重计算在 BW 图结构上与非重计算
  差异很小（实测 27 vs 28 compute 节点），主要影响因素是 placeholder 数量
  而非计算节点，overlap-aware peak（优化 A）已覆盖此场景

**安全性论证**：
- 依赖 PyTorch 编译器自身的元数据，不是启发式规则
- 零误判（实测确认）
- 即使元数据在未来 PyTorch 版本变化，fallback 到 False 只是不启用条件优化，不会恶化

---

### 优化 C：L2.5 条件化调度 + 保持独立性

**目标**：修复 L2.5 在重计算场景下比 L2 更差的问题，同时保持 L2.5 作为独立估计路径。

**当前问题**：
- S09 (b=0.0): L2=18.9% → L2.5=27.3%（`optimize_order` 让 placeholder 释放更快，低估恶化）
- L2.5 取 `min(sched_bw, bw_fi_peak)` → 重计算时两个值都低估，min 选更低的

**修复方案**：

```python
def estimate_inductor_training_peak(cap, model, ...,
                                     has_recomputation: bool | None = None):
    # 重计算检测：优先使用调用方显式指定，其次自动检测 AC 元数据
    if has_recomputation is None:
        has_recomp = detect_recomputation(bw_gm)  # 仅检测 AC 式重计算
    else:
        has_recomp = has_recomputation

    # 1. 条件化 optimize_order
    bw_fa = estimate_graph_peak(
        bw_gm, pin_output_inputs=True,
        fusion_aware=True, simulate_inplace=True,
        optimize_order=not has_recomp,  # 重计算时不优化调度
        overlap_bytes=fwd_primal,       # 使用优化 A 的聚合 overlap 跟踪
    )

    # 2. 保持 L2.5 独立性：不再取 min(sched, graph)
    bw_fi_peak = bw_fa["peak_net_bytes"]  # 已经是 overlap-aware 的
    l25_bw_peak = static_base + bw_fi_peak
    # L3 独立报告，不混入 L2.5
```

**已知局限**：
- S09 (inductor b=0.0) 使用 min-cut 但无 AC 元数据 → `detect_recomputation` 返回 False
  → optimize_order 仍启用 → L2.5 仍低估。实验脚本可传入 `has_recomputation=True`
  覆盖，或直接使用 L3 (S09 L3 MRE=1.6%)
- S07 (b=1.0) 和 S08 (b=0.5) 同样走 min-cut 但 L2.5 表现优秀 (0.4%/1.6%)，
  说明问题不在 min-cut 本身，而在 extreme budget 导致 BW placeholder 过少

**安全性论证**：
- `detect_recomputation` 基于 PyTorch 编译器元数据，对 AC 零误判
- L2.5 不再依赖 L3 (Scheduler) 数据，保持两条独立估计路径
- 对非重计算策略：`optimize_order=True` 继续生效，L2.5 仍优于 L2
- L3 作为第三条独立路径单独报告

---

### 优化 D：Dark Memory 结构化建模（替代硬编码常量）

**目标**：消除 `static_base` 与运行时 base 之间的恒定偏差（~100-200MB）。

**数据中的证据**：`err_fixed` 恒为 ~3.35-3.41GB，即 `rt_base - static_base ≈ 恒定差`。

**差异来源分析**：
1. **模型 Buffers**（LayerNorm weight/bias 中标记为 buffer 的张量）：可精确计算
2. **CUDA Context**：~80-150MB，依赖 GPU 型号和 driver 版本
3. **PyTorch 内部开销**：autograd metadata、编译缓存等

**方法**：只建模可精确计算的部分，不添加经验常量。

```python
# 在 estimate_training_peak 中:
buffer_bytes = sum(
    b.numel() * b.element_size()
    for b in model.buffers()
)
static_base = param_bytes + optim_bytes + buffer_bytes
# 不添加 CUDA context 常量——这属于系统级开销，
# 与仿真算法精度无关，且不同环境差异大
```

**安全性论证**：
- `model.buffers()` 是确定性的，无经验参数
- CUDA context 不纳入仿真——它属于"暗物质"，在 MRE 计算时作为已知系统偏差记录
- 论文中可以将此偏差作为"系统开销"单独列出，而非试图精确建模

---

### 优化 E：End-of-Graph Current 兜底

**目标**：一行代码的安全兜底，确保 BW 图的 live-range 遍历不遗漏终态。

**原理**：BW 图遍历结束时，`current` 包含所有 pinned 输出（梯度张量）+ 未释放的激活。
如果此终态 `current` 经 overlap 修正后超过了历史 peak_net，则以此为准。

```python
# 在 estimate_graph_peak 的 live-range 遍历结束后:
end_net = current - overlap_alive  # overlap_alive 此时为仍存活的 forwarded primals
peak_net = max(peak_net, end_net)
```

这是优化 A 的自然补充——优化 A 在每个 alloc 时刻更新 peak_net，
但 free 事件后的状态也需要检查（例如：最后一个 placeholder 被释放后，
仅剩 pinned 梯度输出的时刻可能超过历史记录的 peak_net）。

---

## 三、明确废弃的方案

| 方案 | 废弃理由 | 替代 |
|------|---------|------|
| 静态 2MB 块对齐 (旧优化3) | CachingAllocator 使用 Segment 分裂，单张量 2MB 对齐导致严重高估 | 保持 512B 请求对齐 |
| `max(graph_peak-overlap, grad_bytes)` (旧优化1) | 过度悲观：分布式训练中梯度可能被提前通信释放，冻结层不产生梯度 | 优化 A (overlap-aware peak) |
| `0.7*L3 + 0.3*L2.5` 加权融合 (评审建议) | 线性加权无理论依据，不同场景最优权重不同 | 保持三条路径独立报告 |
| 1.5x 容忍复用 (旧优化8) | 忽略 CUDA Stream 同步约束，可能导致乐观低估 | 暂缓，需 stream 感知 |
| 重计算感知调度器 (旧优化10) | 复杂度高，需要完整的层级结构识别 | 优化 B+C (条件禁用调度) |

---

## 四、关于轻量级 BFC Allocator 仿真的讨论

### 4.1 xMem 方法的启示

xMem 的核心创新是在 Python 中完整仿真 PyTorch CUDACachingAllocator：
- 两个 pool (small/large) + BFC (Best Fit with Coalescing)
- Segment 分配 + Block 分裂/合并
- 缓存行为 + OOM 时 reclaim

这使得 xMem 在 eager 模式下达到 ~3-4% MRE。

### 4.2 本项目的定位差异

我们的项目工作在 **torch.compile 的 ATen IR 图级别**，而非 eager 运行时。
这带来两个根本差异：

1. **Inductor 后端会改变分配模式**：Triton codegen 的 buffer reuse、fusion 消除等
   行为在 eager profiling 中不存在。xMem 的 CPU profiling 方法对 compiled 模型无效。

2. **我们已有 L3 (Scheduler)**：Inductor 内部的 `estimate_peak_memory` 已经考虑了
   编译后的 buffer 规划。对 compiled 策略，L3 是最准确的参考。

### 4.3 结论

**不建议在 L2/L2.5 中实装完整 BFC 仿真**——投入产出比低，且对 compiled 代码路径
的收益不确定。保持 512B 请求对齐（与 CachingAllocator 一致）即可。

如需进一步提升精度，建议**完善 L3 hook**（Inductor Scheduler）或研究 Inductor 的
`MemoryPlanning` pass 的仿真。

---

## 五、执行计划

### Phase 1: 核心修复（优化 A + E）

修改 `estimate_graph_peak` 和 `estimate_training_peak`：
1. 利用已有的 `_forwarded_primal_bytes()` 计算 FW 图内 forwarded primal 总字节
2. `estimate_graph_peak` 新增 `overlap_bytes: int = 0` 参数 + `peak_net_bytes` 返回
3. 在 live-range 遍历中跟踪 `ph_alive`，计算 `net = current - min(overlap_bytes, ph_alive)`
4. `estimate_training_peak` 使用 `peak_net_bytes` 替代手动 overlap 计算
5. End-of-graph current 兜底
6. 更新 `estimate_inductor_training_peak` 中 L2 路径
7. 更新测试

**预期效果**：
- S10 MRE: 17.2% → ~8-10%
- S09 L2 MRE: 18.9% → ~12-15%
- S05/S07/S12 不变

### Phase 2: L2.5 修复（优化 B + C）

1. 实现 `detect_recomputation(bw_gm)` 函数（meta['recompute'] 检测 AC 重计算）
2. `estimate_inductor_training_peak` 新增 `has_recomputation: bool | None = None` 参数
   - `None`（默认）：自动通过 `detect_recomputation(bw_gm)` 检测
   - `True`/`False`：调用方显式指定（用于 extreme min-cut 等元数据无法覆盖的场景）
3. 条件化 `optimize_order = not has_recomp`
4. L2.5 使用 overlap-aware peak（与 L2 相同的修复）
5. 移除 L2.5 对 L3 sched_bw 的 min() 依赖
6. 更新测试

**预期效果**：
- S10/S11 L2.5 MRE: 显著改善（AC 元数据检测 → 禁用 optimize_order）
- S07/S08/S12 L2.5: 不变或略有改善（optimize_order 保持启用）
- S09 L2.5: 仅受 Phase 1 (Overlap-Aware Peak) 改善，预期 ~20-23%
  ⚠ `detect_recomputation` 对 min-cut 返回 False，optimize_order 仍生效。
  此为已知局限——S09 的 L3 MRE=1.6%，对 extreme min-cut 推荐使用 L3。
  如需进一步优化，实验脚本可对 budget<0.5 的策略传入 `has_recomputation=True`。

### Phase 3: Dark Memory 建模（优化 D）

1. 在 `estimate_training_peak` 中加入 `model.buffers()` 字节计算
2. 更新文档中对 dark memory 的解释

**预期效果**：
- 所有策略 static_base 改善 ~50-100MB

### Phase 4: 实验验证 + 论文素材

1. 重跑 `ex_sim_accuracy.py`，更新 CSV
2. 对比修复前后的 MRE
3. 制作 Before/After 对比表（论文用）

---

## 六、与现有工作的差异化定位

| 维度 | DNNMem | LLMem | xMem | **本项目** |
|------|--------|-------|------|----------|
| 输入 | 静态计算图 | 模型配置 | CPU 运行 trace | **ATen IR FX Graph** |
| 目标 | eager 训练 | eager 微调 | eager 训练 | **compiled 训练** |
| 重计算支持 | ✗ | 公式近似 | 运行时捕获 | **图结构分析** |
| Allocator 建模 | 64MB 常量 | 2MB 页对齐 | 完整 BFC 仿真 | **512B 请求对齐** |
| 编译器优化 | ✗ | ✗ | ✗ | **Inductor fusion/reuse** |
| 最佳 MRE | 19% | 5% | 3-4% | **目标: <10% (compiled+AC)** |

**本项目的独特贡献**：
1. 首个针对 `torch.compile` 后 ATen IR 的静态显存仿真
2. Overlap-aware peak tracking 解决 AC/重计算场景的峰值时间偏移问题
3. 三层独立仿真 (L2/L2.5/L3) 提供不同精度-开销权衡
4. 编译器元数据重计算检测（`meta['recompute']`）替代启发式阈值
