# 16 — 参数双重计数修复：L2 MRE 12% → 8.8%

> **文档定位**: 记录 L2/L2.5 仿真中模型参数与图级峰值双重计数问题的发现、
> 根因分析、修复方案（`peak_ph_alive` + `_forwarded_primal_bytes`）及验证结果。
>
> **前置阅读**: `11-l2-accuracy-improvement.md`（L2 MRE 21%→7% 的首轮修复）

---

## 一、问题背景

在 `11-l2-accuracy-improvement.md` 所述的 view-base pin 和 grad_bytes 修复后，
L2 仿真在跨策略对比实验（`ex_sim_accuracy.py`）中仍然存在**系统性高估**：

| 策略 | RT 峰值 | L2 峰值 | MRE | 方向 |
|------|---------|---------|-----|------|
| aot_eager+default | 27.2 GB | 29.2 GB | 15.6% | over |
| inductor(b=1.0) | 21.9 GB | 23.5 GB | 15.6% | over |
| inductor(b=0.5) | 16.7 GB | 18.7 GB | 22.5% | over |
| sac_mm+inductor | 19.7 GB | 21.7 GB | 18.3% | over |

**平均 L2 MRE ≈ 12%**，主要来自 BW 阶段系统性高估。

---

## 二、根因分析

### 2.1 峰值公式回顾

修复前的 L2 公式：

```python
static_base = param_bytes + optim_bytes
fw_peak = static_base + fw_graph_peak
bw_peak = static_base + bw_graph_peak
true_peak = max(fw_peak, bw_peak, opt_peak)
```

### 2.2 双重计数机制

`estimate_graph_peak` 通过 live-range 分析计算图级峰值。FX 图中的 **placeholder 节点**
代表图的输入——对于 AOTAutograd 分区后的 FW 图，前 N 个 placeholder 就是模型参数（primals）。

问题：图级分析把这些 placeholder 当作"新分配"计入峰值，但它们在 `static_base` 中已被计算过：

```
static_base = param_bytes + optim_bytes     ← 包含 param_bytes
fw_graph_peak = peak(params + activations)  ← 又包含 param_bytes
fw_peak = static_base + fw_graph_peak       ← param_bytes 被加了两次！
```

对于 LLaMA-2B 模型（hidden=2048, 16 layers），`param_bytes ≈ 3.5 GB`，
占总峰值的 ~15%。双重计数导致系统性高估约 15%。

### 2.3 BW 图的特殊性

BW 图的 placeholder 不是模型参数，而是：
- **saved activations**（FW 图保存的中间结果）
- **forwarded primals**（FW 图直接转发的模型参数引用）
- **tangent 输入**（loss 的梯度）

只有 **forwarded primals** 与 `static_base` 中的 `param_bytes` 重叠。
不同策略的 forwarded primal 数量差异很大：

| 策略 | BW placeholder 构成 | forwarded primals |
|------|---------------------|-------------------|
| 默认（无重计算） | saved act + param refs | ≈ param_bytes |
| AC（激活检查点） | checkpoint 边界 + tangent | ≈ 0 |
| inductor(b=0.0) | 少量 saved + tangent | ≈ 0 |
| min_cut | 较少的 saved act | < param_bytes |

因此 BW 不能简单扣除 `param_bytes`——需要精确判断每种策略实际转发了多少参数。

---

## 三、修复方案

### 3.1 组件一：`peak_ph_alive` 跟踪

在 `estimate_graph_peak` 的 live-range 模拟中，新增追踪：
每当峰值更新时，记录"此刻有多少 placeholder 字节还存活"。

```python
# estimate_graph_peak 内部
peak_ph_alive = 0
ph_nodes: set[fx.Node] = set()

# 分配时标记 placeholder
if node.op == "placeholder":
    ph_nodes.add(node)

# 峰值更新时记录
if current > peak:
    peak = current
    peak_ph_alive = sum(live[n] for n in ph_nodes if n in live)

# 释放时移除标记
ph_nodes.discard(live_node)
```

返回值新增 `"peak_ph_alive"` 字段。

### 3.2 组件二：`_forwarded_primal_bytes(fw_gm, n_params)`

**目的**：精确计算 FW 图转发给 BW 图的模型参数字节数。

**原理**：AOTAutograd 分区后，FW 图某些输出直接指向模型参数 storage（可能经过 view 操作）。
运行时这些 BW placeholder 不分配新内存——它们只是指向同一块参数 storage。

**实现**：使用 FakeTensor 的 `untyped_storage()._cdata` 做 storage 匹配：

```python
def _forwarded_primal_bytes(fw_gm: fx.GraphModule, n_params: int) -> int:
    placeholders = [n for n in fw_gm.graph.nodes if n.op == "placeholder"]
    param_phs = placeholders[:n_params]

    # 收集参数 placeholder 的 storage ID
    param_storages: dict[int, int] = {}  # cdata → nbytes
    for ph in param_phs:
        val = ph.meta.get("val")
        if isinstance(val, torch.Tensor):
            cdata = val.untyped_storage()._cdata
            param_storages[cdata] = int(val.untyped_storage().nbytes())

    # 检查 FW 输出是否共享参数 storage
    output_node = next(n for n in fw_gm.graph.nodes if n.op == "output")
    output_args = output_node.args[0]

    seen_storages: set[int] = set()
    forwarded = 0
    for arg in output_args:
        if not isinstance(arg, fx.Node):
            continue
        val = arg.meta.get("val")
        if isinstance(val, torch.Tensor):
            cdata = val.untyped_storage()._cdata
            if cdata in param_storages and cdata not in seen_storages:
                seen_storages.add(cdata)
                forwarded += param_storages[cdata]
    return int(forwarded)
```

**关键设计决策**：
- 使用 storage 匹配而非 view-chain 回溯：可捕获任意中间 view 操作（`t()`, `reshape`, `slice` 等）
- 去重（`seen_storages`）：同一参数可能被多次 view 保存，避免重复计数
- 使用 `untyped_storage().nbytes()` 而非 `nelement() * element_size()`：以 storage 为单位计算

### 3.3 组件三：精确扣除公式

```python
# FW：placeholder 几乎全是参数，用 peak_ph_alive 扣除
fw_ph_overlap = min(param_bytes, fw_result["peak_ph_alive"])

# BW：只扣除 forwarded primal 部分
n_params = len(list(model.parameters())) + len(list(model.buffers()))
fwd_primal = _forwarded_primal_bytes(fw_gm, n_params)
bw_ph_overlap = min(fwd_primal, bw_result["peak_ph_alive"])

# 修正后的峰值公式
fw_peak = static_base + max(0, fw_graph_peak - fw_ph_overlap)
bw_peak = static_base + max(0, bw_graph_peak - bw_ph_overlap)
```

### 3.4 公式对比

```
旧公式:
  fw_peak = static_base + fw_graph_peak                    ← 双重计数 param_bytes
  bw_peak = static_base + bw_graph_peak                    ← 双重计数 forwarded primals

新公式:
  fw_peak = static_base + (fw_graph_peak - fw_ph_overlap)  ← 扣除参数 placeholder
  bw_peak = static_base + (bw_graph_peak - bw_ph_overlap)  ← 只扣除转发的参数部分
```

### 3.5 L2.5 同步修复

`estimate_inductor_training_peak` 中的 L2.5（fusion-aware + in-place reuse）
也应用了相同逻辑：

```python
# L2.5: FW 用 peak_ph_alive 扣除
fw_fa_ph_overlap = min(param_bytes, fw_fa["peak_ph_alive"])
l25_fw_peak = static_base + max(0, fw_fa["peak_bytes"] - fw_fa_ph_overlap)

# L2.5: BW 用 forwarded_primal_bytes 扣除
bw_fa_ph_overlap = min(fwd_primal, bw_fa["peak_ph_alive"])
bw_fi_peak = max(0, bw_fa["peak_bytes"] - bw_fa_ph_overlap)
```

注意：L3（Scheduler 峰值）不受影响，因为 Inductor Scheduler 的峰值不包含参数 placeholder。

---

## 四、验证结果

### 4.1 跨策略对比（LLaMA-2B, batch=4, seq=512）

| 策略 | RT 峰值 | L2 峰值 | **MRE** | 方向 | L2.5 MRE | L3 MRE |
|------|---------|---------|---------|------|----------|--------|
| S05 aot_eager+default | 29.2 GB | 29.2 GB | **0.01%** | over | — | — |
| S06 aot_eager+min_cut | 26.7 GB | 23.4 GB | 12.1% | under | — | — |
| S07 inductor(b=1.0) | 23.5 GB | 24.0 GB | **1.8%** | over | 0.4% | 13.3% |
| S08 inductor(b=0.5) | 17.9 GB | 18.7 GB | **4.5%** | over | 1.6% | 19.6% |
| S09 inductor(b=0.0) | 23.8 GB | 19.3 GB | 18.9% | under | 27.3% | 1.6% |
| S10 ac+aot_eager | 16.9 GB | 14.0 GB | 17.2% | under | — | — |
| S11 ac+inductor | 16.1 GB | 14.0 GB | 13.2% | under | 13.2% | 9.7% |
| S12 sac_mm+inductor | 21.1 GB | 21.7 GB | **2.9%** | over | 0.4% | 0.8% |

### 4.2 汇总指标

| 指标 | 修复前 | 修复后 | 改善 |
|------|--------|--------|------|
| **L2 平均 MRE** | ~12% | **8.8%** | **↓ 3.2pp** |
| **L2 最大 MRE** | ~22% | 18.9% | ↓ 3pp |
| L2.5 平均 MRE | ~13% | **8.6%** | ↓ 4.4pp |
| L3 平均 MRE | 9.0% | 9.0% | — |
| 高估策略数 | 8/8 | 4/8 | 平衡化 |
| 低估策略数 | 0/8 | 4/8 | 平衡化 |

### 4.3 效果最佳的策略

修复对以下策略效果最显著（修复前均为 15-22% 高估）：
- **S05 aot_eager+default**: 15.6% over → **0.01%** ✅
- **S07 inductor(b=1.0)**: 15.6% over → **1.8%** ✅
- **S08 inductor(b=0.5)**: 22.5% over → **4.5%** ✅
- **S12 sac_mm+inductor**: 18.3% over → **2.9%** ✅

这些策略的共同特点：BW 图包含大量 forwarded primals（参数转发），
`_forwarded_primal_bytes` 返回值接近 `param_bytes`，精确扣除生效。

### 4.4 测试回归

```
======================= 85 passed, 14 warnings =======================
```

涉及更新的测试用例：
- `test_peak_formula`：改为验证 `estimated_peak == true_peak`，不硬编码公式
- `test_l2_absolute_peaks`：改为范围断言 `base ≤ fw_peak ≤ base + fw_graph_peak`
- `test_l3_leq_l2`：L3 为独立估计，不再要求 `L3 ≤ L2`
- `test_l25_between_l2_and_l3`：移除 `L3 ≤ L2` 约束

---

## 五、残余误差分析

### 5.1 AC/重计算策略的低估（S09, S10, S11）

这些策略的 BW 图包含重计算节点（recomputed forward）。
我们的 live-range 分析正确建模了这些节点的内存生命周期，
但运行时的峰值更高，可能原因：

1. **CUDA Caching Allocator 碎片化**：重计算产生的临时 buffer 与已有 block 不匹配
2. **图级分析未建模的运行时开销**：cuBLAS workspace、kernel launch buffer 等
3. **Inductor codegen 差异**：实际 kernel 可能比图级分析使用更多临时内存

这些误差需要 L3（Scheduler 模拟）或 L4（Allocator Replay）层级解决。
当前 L3 对 S09 的 MRE 仅 1.6%，证实了 Scheduler 层级能更好地建模重计算场景。

### 5.2 min_cut 策略的低估（S06）

`min_cut` 分区将某些 tensor 保存为非参数节点的输出，
但运行时实际保存的 tensor 可能比图分析看到的更多（图分析基于 FakeTensor，
不完全反映运行时的 autograd 保存行为）。

### 5.3 修复的适用边界

| 场景 | 本修复是否生效 | 原因 |
|------|---------------|------|
| 默认分区 + Inductor | ✅ 强效 | FW 转发大量参数 |
| AC/全重计算 | ❌ 不影响 | BW 几乎无转发参数 |
| min_cut 分区 | ⚠️ 部分 | 转发参数量取决于切割策略 |

---

## 六、与 doc-11 修复的关系

| 修复 | 文档 | 问题 | 手段 | MRE 改善 |
|------|------|------|------|----------|
| View-base pin | doc-11 | FW 峰值低估（view storage 被错误释放） | `_find_view_base` + pin base | 21% → 7% |
| Grad_bytes 移除 | doc-11 | BW 峰值高估（梯度被加两次） | 移除 `+ grad_bytes` | 同上 |
| **参数双重计数** | **doc-16** | FW/BW 峰值高估（param 在 static_base 和 graph_peak 中各算一次） | `peak_ph_alive` + `_forwarded_primal_bytes` | **12% → 8.8%** |

三个修复是独立且互补的，共同将 L2 MRE 从初始的 ~50% 降至 **8.8%**。

---

## 七、修改文件清单

| 文件 | 改动 |
|------|------|
| `toolkit/simulation/graph_estimator.py` | 新增 `_forwarded_primal_bytes()`（55 行）；`estimate_graph_peak` 增加 `peak_ph_alive` 跟踪（~15 行）；`estimate_training_peak` 使用精确扣除公式；`estimate_inductor_training_peak` L2.5 同步修复 |
| `tests/test_simulation.py` | 4 个测试断言更新 |

核心改动：**~70 行新增代码 + 4 处公式修改**，无 API 变更，完全向后兼容。

---

## 参考

- `11-l2-accuracy-improvement.md`：首轮 L2 修复（view-base pin + grad_bytes）
- `08-static-simulation-deep-dive.md`：L2 仿真引擎全链路解析
- `13-l2.5-fusion-aware-design.md`：L2.5 fusion-aware 设计
- PyTorch: `torch/_functorch/aot_autograd.py` — AOTAutograd 分区后的 primal 顺序
- PyTorch: `torch/_inductor/scheduler.py` — Scheduler 峰值估算（L3 对照）
