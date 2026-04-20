# 09 — 运行时 Profiler 与验证机制源码深度解析

> **文档定位**: 对本项目运行时 profiler（Pillar 4）的全链路解析，
> 涵盖 CUDA Caching Allocator 内存模型、分阶段峰值测量原理、
> CUDA Event 计时、IQR mean 聚合策略、Memory Snapshot / Chrome Trace 导出、
> 以及 Validator 的静态-运行时对比校验机制。
>
> **前置阅读**: `08-static-simulation-deep-dive.md`（静态仿真引擎）

---

## 一、概述：运行时 Profiler 在项目中的角色

```
┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
│ Pillar 1   │    │ Pillar 2   │    │ Pillar 3   │    │ Pillar 4   │
│ 策略注入    │ →  │ 图捕获     │ →  │ 静态仿真    │ →  │ 运行时验证  │ ← 本文
└────────────┘    └────────────┘    └────────────┘    └────────────┘
```

Pillar 4 提供**运行时 ground truth**——真正在 GPU 上执行训练步，
通过 CUDA 提供的硬件级 API 测量内存和时间。其核心价值：

1. **校验仿真精度**：对比 L1/L2 估算值与真实值，计算 MRE
2. **提供绝对基准**：作为策略比较的客观标尺
3. **诊断误差来源**：定位仿真误差来自参数/激活/allocator 哪一层

---

## 二、PyTorch CUDA Caching Allocator 内存模型

理解 profiler 需要先理解 PyTorch 的两层内存管理：

### 2.1 两层架构

```
┌──────────────────────────────────────────┐
│              Python Tensors              │
│  x = torch.randn(1000, 1000, device='cuda')  │
└───────────────────┬──────────────────────┘
                    │ malloc/free
┌───────────────────▼──────────────────────┐
│         CUDA Caching Allocator           │  ← PyTorch 管理层
│  维护 block pool，复用已释放的 block       │
│  allocated = 当前活跃的 tensor 总字节数     │
│  reserved  = 向 CUDA 申请的总字节数        │
└───────────────────┬──────────────────────┘
                    │ cudaMalloc / cudaFree
┌───────────────────▼──────────────────────┐
│           CUDA Driver / GPU VRAM          │  ← 硬件层
└──────────────────────────────────────────┘
```

### 2.2 allocated vs reserved

| 指标 | 含义 | API |
|------|------|-----|
| **allocated** | 当前有 Python tensor 引用的内存 | `torch.cuda.memory_allocated()` |
| **reserved** | Caching Allocator 向 CUDA 申请的总内存（含空闲 block） | `torch.cuda.memory_reserved()` |

```
allocated ≤ reserved

reserved - allocated = 空闲 block pool（已释放但未归还给 CUDA 的内存）
```

**实际验证**（在项目 conda 环境中）：
```python
x = torch.randn(1000, 1000, device='cuda')   # 4 MB tensor
# allocated: 4.0 MB, reserved: 21.0 MB       # reserved 远大于 allocated

y = torch.randn(1000, 1000, device='cuda')
# allocated: 8.0 MB, reserved: 21.0 MB       # reserved 不变（从空闲 pool 分配）

del y
# allocated: 4.0 MB, reserved: 21.0 MB       # reserved 不减（block 回到 pool）

torch.cuda.empty_cache()
# allocated: 4.0 MB, reserved: 21.0 MB       # empty_cache 后 pool 归还
```

### 2.3 peak 追踪机制

PyTorch 内部维护一个 `peak` 计数器，在每次分配后更新 `peak = max(peak, current)`。

```
reset_peak_memory_stats()    →  peak := current
                                 （重新开始追踪）

... 执行一些操作 ...

max_memory_allocated()       →  返回自上次 reset 以来的 peak
```

**关键**: `reset_peak_memory_stats` 不改变 `current`，只把 `peak` 重置为当前值。
这允许我们**分阶段**测量 FW / BW / OPT 各自的峰值。

### 2.4 memory_stats() 的 122 个统计键

```python
stats = torch.cuda.memory_stats(device)
```

分为 10 个大类 × {all, large_pool, small_pool} × {current, peak, allocated, freed}：

| 类别 | 含义 |
|------|------|
| `allocated_bytes` | 当前被 tensor 占用的字节 |
| `reserved_bytes` | Caching Allocator 保留的字节 |
| `active_bytes` | 活跃 block 的字节（含内部碎片） |
| `requested_bytes` | 用户请求的字节（不含对齐填充） |
| `allocation` | 分配操作计数 |
| `segment` | CUDA segment 计数 |
| `inactive_split` / `inactive_split_bytes` | 被 split 后闲置的 block |
| `oversize_allocations/segments` | 超大分配统计 |
| `num_alloc_retries` | 分配失败重试次数 |
| `num_ooms` | OOM 次数 |

本项目主要使用：
- `allocated_bytes.all.peak` → `measure_step` 中的 `peak_alloc`
- `reserved_bytes.all.peak` → `peak_reserved`

---

## 三、`measure_step()` — 全步测量

**源文件**: `toolkit/profiler/step_profiler.py` 第 53-113 行

### 3.1 设计目的

单次测量整个训练步（FW+BW+OPT 不分开）的峰值内存和时间。

### 3.2 测量流程

```python
def measure_step(name, forward_fn, optimizer, *, repeats=6, warmup=2, device="cuda"):
    # ① Warmup：消除 JIT 编译、CUDA context 初始化的影响
    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        loss = forward_fn()
        loss.backward()
        optimizer.step()

    for _ in range(repeats):
        # ② 清理：释放 caching allocator 空闲块
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        # ③ 基座测量：zero_grad 后只剩 param + optim states
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated(device)

        # ④ CUDA Event 计时
        ev_s, ev_f, ev_b, ev_e = [torch.cuda.Event(enable_timing=True) for _ in range(4)]

        ev_s.record()
        loss = forward_fn()       # ← FW
        ev_f.record()

        loss.backward()           # ← BW
        ev_b.record()

        optimizer.step()          # ← OPT
        ev_e.record()
        torch.cuda.synchronize()  # ← 等待所有 GPU 操作完成

        # ⑤ 读取 peak（全步峰值，因为只 reset 了一次）
        stats = torch.cuda.memory_stats(device)
        peak_alloc = stats["allocated_bytes.all.peak"]
        peak_reserved = stats["reserved_bytes.all.peak"]

    # ⑥ IQR mean 聚合
    return StepResult(
        peak_allocated=int(iqr_mean(peaks_alloc)),
        peak_reserved=int(iqr_mean(peaks_reserved)),
        base_allocated=int(sum(bases) / len(bases)),
        ...
    )
```

### 3.3 `StepResult` 数据结构

```python
@dataclass
class StepResult:
    name: str
    peak_allocated: int       # IQR mean of allocated peak
    peak_reserved: int        # IQR mean of reserved peak
    base_allocated: int       # avg of base (param + optim states)
    activation_delta: int     # peak - base
    elapsed_ms: float         # 全步时间
    fw_ms: float              # FW 时间
    bw_ms: float              # BW 时间
    opt_ms: float             # OPT 时间
```

---

## 四、`measure_phased()` — 分阶段测量

**源文件**: `toolkit/profiler/step_profiler.py` 第 116-200 行

### 4.1 设计目的

分别测量 FW / BW / OPT 三个阶段的**独立峰值**，这是与静态仿真四峰值体系对齐的关键。

### 4.2 核心技巧：每阶段独立 reset peak

```python
def measure_phased(name, forward_fn, optimizer, *, repeats=5, warmup=3, device="cuda"):
    for _ in range(repeats):
        torch.cuda.empty_cache()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated(device)

        # ── FW 阶段 ──
        torch.cuda.reset_peak_memory_stats(device)    # peak := current
        ev_s.record()
        loss = forward_fn()
        torch.cuda.synchronize()
        fw_peak = torch.cuda.max_memory_allocated(device)    # FW 阶段独立峰值
        after_fw = torch.cuda.memory_allocated(device)       # FW 结束后的当前内存

        # ── BW 阶段 ──
        torch.cuda.reset_peak_memory_stats(device)    # peak := current（含 saved acts）
        ev_f.record()
        loss.backward()
        torch.cuda.synchronize()
        bw_peak = torch.cuda.max_memory_allocated(device)    # BW 阶段独立峰值
        after_bw = torch.cuda.memory_allocated(device)       # BW 结束后（grad 仍在）

        # ── OPT 阶段 ──
        torch.cuda.reset_peak_memory_stats(device)    # peak := current（含 grad）
        ev_b.record()
        optimizer.step()
        ev_e.record()
        torch.cuda.synchronize()
        opt_peak = torch.cuda.max_memory_allocated(device)   # OPT 阶段独立峰值
        after_opt = torch.cuda.memory_allocated(device)
```

**三次 `reset_peak_memory_stats`** 是整个分阶段测量的核心：

| reset 时机 | peak 从何值开始追踪 | 测量到的峰值含义 |
|------------|-------------------|---------------|
| FW 开始前 | base (param + optim) | FW 阶段绝对峰值 = base + fw_activation_peak |
| BW 开始前 | after_fw (base + saved acts) | BW 阶段绝对峰值 = base + grad + bw_act_peak |
| OPT 开始前 | after_bw (base + grad) | OPT 阶段绝对峰值 = base + grad + opt_temp |

### 4.3 时间线采样点

```python
after_fw  = torch.cuda.memory_allocated()   # FW 结束时的 current
                                             # ≈ base + saved_activations
after_bw  = torch.cuda.memory_allocated()   # BW 结束时的 current
                                             # ≈ base + grad_bytes（activations 已释放）
after_opt = torch.cuda.memory_allocated()   # OPT 结束时的 current
                                             # ≈ base + grad_bytes（opt_temp 已释放）
```

这些采样点用于绘制 phase timeline chart，与静态仿真的同名输出直接对比。

### 4.4 聚合与输出

```python
# IQR mean 聚合多次运行
agg_fw = int(iqr_mean([r.fw_peak for r in results]))
agg_bw = int(iqr_mean([r.bw_peak for r in results]))
agg_opt = int(iqr_mean([r.opt_peak for r in results]))
agg_grad = int(iqr_mean([r.after_bw - r.after_fw for r in results]))

fwbw = max(agg_fw, agg_bw)
overall = max(agg_fw, agg_bw, agg_opt)
peaks = {agg_fw: "FW", agg_bw: "BW", agg_opt: "OPT"}

return PhaseResult(
    fw_peak=agg_fw, bw_peak=agg_bw, opt_peak=agg_opt,
    overall_peak=overall,
    fwbw_peak=fwbw,
    peak_phase=peaks[overall],    # "FW" / "BW" / "OPT"
    grad_bytes=agg_grad,          # after_bw - after_fw 估算梯度大小
    ...
)
```

### 4.5 `PhaseResult` 数据结构

```python
@dataclass
class PhaseResult:
    name: str
    fw_peak: int              # FW 阶段绝对峰值
    bw_peak: int              # BW 阶段绝对峰值
    opt_peak: int             # OPT 阶段绝对峰值
    after_fw: int             # FW 结束后 current
    after_bw: int             # BW 结束后 current
    after_opt: int            # OPT 结束后 current
    base_allocated: int       # 基座（param + optim states）
    overall_peak: int         # max(fw, bw, opt)
    activation_delta: int     # overall_peak - base
    fwbw_peak: int            # max(fw, bw)
    peak_phase: str           # "FW" / "BW" / "OPT"
    grad_bytes: int           # 梯度估算 (after_bw - after_fw)
    fw_ms: float; bw_ms: float; opt_ms: float; step_ms: float
```

---

## 五、CUDA Event 计时原理

### 5.1 为什么不用 `time.time()`

GPU 操作是异步的——`loss = forward_fn()` 在 Python 侧立即返回，
但 GPU 内核还在执行。用 `time.time()` 测的是 CPU 发射时间，不是 GPU 执行时间。

### 5.2 CUDA Event 机制

```python
ev_start = torch.cuda.Event(enable_timing=True)
ev_end   = torch.cuda.Event(enable_timing=True)

ev_start.record()        # 在 CUDA stream 中插入时间戳标记
loss = forward_fn()      # GPU 异步执行
ev_end.record()          # 在 CUDA stream 中插入结束标记

torch.cuda.synchronize() # 等待所有 GPU 操作完成
elapsed = ev_start.elapsed_time(ev_end)  # 返回毫秒
```

**原理**：
1. `record()` 在当前 CUDA stream 中插入一个硬件时间戳
2. `synchronize()` 阻塞 CPU 直到所有 GPU 操作完成
3. `elapsed_time()` 读取两个硬件时间戳的差值（GPU 时钟精度，μs 级）

### 5.3 本项目的四事件布局

```
ev_s ─── FW ─── ev_f ─── BW ─── ev_b ─── OPT ─── ev_e
│                │                │                 │
├── fw_ms ───────┤                │                 │
│                ├── bw_ms ───────┤                 │
│                │                ├── opt_ms ───────┤
├──────────────── step_ms ────────────────────────┤
```

---

## 六、IQR Mean 聚合策略

**源文件**: `toolkit/profiler/step_profiler.py` 第 6-16 行

### 6.1 实现

```python
def iqr_mean(values):
    if len(values) <= 2:
        return sum(values) / len(values)         # 样本太少，直接均值
    ordered = sorted(values)
    n = len(ordered)
    q1 = n // 4
    q3 = n - n // 4
    middle = ordered[q1:q3]                      # 去掉 25% 最低 + 25% 最高
    return sum(middle) / len(middle)
```

### 6.2 为什么不用 median 或 mean

| 方法 | 问题 |
|------|------|
| **mean** | 对异常值敏感：一次 GC 触发或 CUDA context 初始化会让某次测量偏高 |
| **median** | 只用一个数据点，信息浪费 |
| **IQR mean** | 去掉上下 25% 异常值后取均值，稳健且利用多个数据点 |

实测结果：两次独立运行的 IQR mean diff = 0.00%（完全可复现）。

### 6.3 样本量选择

- `measure_step`: warmup=2, repeats=6 → IQR mean 使用中间 4 个值
- `measure_phased`: warmup=3, repeats=5 → IQR mean 使用中间 3-4 个值

warmup 轮数更多是因为 phased 测量中每轮需要三次 `synchronize()`，
首轮可能受 JIT 编译和 CUDA lazy init 的影响更大。

---

## 七、Memory Snapshot 导出

> **注意**: `snapshot.py` 已在 v6.1 代码清理中删除（未被任何实验脚本引用）。以下保留作为设计参考。

**原源文件**: `toolkit/profiler/snapshot.py`

### 7.1 实现

```python
def capture_snapshot(forward_fn, optimizer, path, device="cuda", max_entries=100000):
    try:
        torch.cuda.memory._record_memory_history(max_entries=max_entries)
        optimizer.zero_grad(set_to_none=True)
        loss = forward_fn()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        torch.cuda.memory._dump_snapshot(str(target))
        torch.cuda.memory._record_memory_history(enabled=None)  # 停止记录
        return True, target.stat().st_size
    except Exception as exc:
        torch.cuda.memory._record_memory_history(enabled=None)
        return False, str(exc)
```

### 7.2 原理

1. **`_record_memory_history(max_entries)`**：开启 Caching Allocator 的历史记录模式，
   记录每次 alloc/free 的：
   - 分配大小、地址
   - Python 调用栈 + CUDA 调用栈
   - 时间戳
   - block 元信息（pool, segment, stream）

2. **`_dump_snapshot(path)`**：将所有记录序列化为 pickle 文件

3. **用途**：
   - 使用 `pytorch.org/memory_viz` 可视化查看器打开
   - 查看每个 tensor 的生命周期、分配来源
   - 诊断内存泄漏、碎片化

### 7.3 与 profiler 的区别

| 维度 | `measure_phased` | `capture_snapshot` |
|------|------------------|--------------------|
| 输出 | 数值（peak, current） | 完整分配历史（pickle） |
| 精度 | 峰值级 | 每次 alloc/free 级 |
| 性能开销 | 极低（只读计数器） | 中等（记录调用栈） |
| 用途 | 自动化对比 | 人工诊断 |

---

## 八、Chrome Trace 时间线导出

> **注意**: `timeline.py` 已在 v6.1 代码清理中删除（未被任何实验脚本引用）。以下保留作为设计参考。

**原源文件**: `toolkit/profiler/timeline.py`

### 8.1 实现

```python
def capture_timeline(forward_fn, optimizer, path, device="cuda", num_steps=5):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(num_steps):
            optimizer.zero_grad(set_to_none=True)
            loss = forward_fn()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize(device)

    prof.export_chrome_trace(str(target))
```

### 8.2 原理

1. **`torch.profiler.profile`**：PyTorch 2.x 的高级 profiler，基于 Kineto
2. **activities**：同时记录 CPU 和 CUDA 活动
3. **record_shapes=True**：记录每个 op 的输入 tensor shape
4. **profile_memory=True**：记录内存分配/释放事件
5. **`export_chrome_trace`**：导出为 Chrome `chrome://tracing` 格式 JSON

### 8.3 用途

用 Chrome 浏览器或 Perfetto 打开，可以看到：
- CPU 端的 Python 调用（op dispatch）
- GPU 端的 kernel 执行（GEMM, softmax, ...）
- 内存分配时间线
- CPU-GPU 的异步关系

---

## 九、Validator — 静态-运行时对比校验

**源文件**: `toolkit/profiler/validator.py`

### 9.1 `validate()` — 主验证函数（第 27-68 行）

```python
def validate(static_result, runtime_result, run_mode="compiled"):
    static_peak  = static_result["true_peak"]           # 静态仿真估算
    runtime_peak = runtime_result.overall_peak           # 运行时实测

    mre_allocated = abs(static_peak - runtime_peak) / runtime_peak
    direction = "over" if static_peak > runtime_peak else "under"

    breakdown = {
        "static_param": static_result["param_bytes"],
        "static_grad":  static_result["grad_bytes"],
        "static_optim": static_result["optimizer_bytes"],
        "static_act":   static_result.get("act_peak", ...),
        "runtime_peak":     runtime_peak,
        "runtime_base":     runtime_result.base_allocated,
        "runtime_act_delta": runtime_result.activation_delta,
        "allocator_overhead": reserved - allocated,
    }

    # 分阶段 MRE
    for phase in ("fw_peak", "bw_peak", "opt_peak"):
        breakdown[f"mre_{phase}"] = abs(static[phase] - runtime[phase]) / runtime[phase]

    return ValidationResult(mre_allocated=mre, direction=direction, breakdown=breakdown)
```

### 9.2 MRE 计算

```
MRE = |static_peak - runtime_peak| / runtime_peak

示例（LLaMA, B10 修复后）:
  static  = 201.8 MB (L2)
  runtime = 188.7 MB
  MRE = |201.8 - 188.7| / 188.7 = 6.9%
  direction = "over"（静态高估）
```

### 9.3 `analyze_error_sources()` — 误差归因（第 71-96 行）

```python
def analyze_error_sources(static_result, runtime_result):
    total_error = static_peak - runtime_peak      # 可正可负

    return {
        "total_error": total_error,
        "total_error_pct": total_error / runtime_peak * 100,
        "sources": [
            ("fixed (param+grad+optim) vs base",
             static_fixed - runtime_base),          # dark memory 差异
            ("activation (static vs delta)",
             static_act - runtime_act),              # 激活估算差异
            ("allocator overhead (reserved-alloc)",
             allocator_overhead),                    # 碎片/预留差异
        ],
    }
```

三个误差来源：
1. **fixed vs base**：`static_fixed = param + grad + optim` vs `runtime_base`。
   差异 = dark memory（CUDA context + buffer + alignment padding），实测 16-19 MB。
2. **activation 差异**：L2 仿真的激活估算 vs 运行时的 `peak - base`。
3. **allocator overhead**：`reserved - allocated`，Caching Allocator 预留但未使用的内存。

### 9.4 `ValidationResult` 数据结构

```python
@dataclass
class ValidationResult:
    tag: str                  # "compiled" / "eager" / ...
    static_peak: int          # 静态仿真 true_peak
    runtime_peak: int         # 运行时 overall_peak
    runtime_reserved: int     # 运行时 reserved（含空闲 pool）
    mre_allocated: float      # MRE（对标 allocated）
    mre_reserved: float       # MRE（对标 reserved）
    run_mode: str
    direction: str            # "over" / "under"
    breakdown: dict           # 分项对比 + 分阶段 MRE
```

---

## 十、measure_step vs measure_phased 对比

| 维度 | `measure_step` | `measure_phased` |
|------|---------------|-----------------|
| peak reset | 整步 1 次 | 每阶段 1 次（共 3 次） |
| 输出 | 整步 peak + 整步时间 | 三阶段 peak + 三阶段时间 |
| 数据结构 | `StepResult` | `PhaseResult` |
| warmup | 2 轮 | 3 轮 |
| repeats | 6 轮 | 5 轮 |
| 用途 | 快速测量、策略筛选 | 精确校验、仿真对比 |
| synchronize | 仅结尾 1 次 | 每阶段 1 次（共 3 次） |
| 性能开销 | 低 | 中（synchronize 破坏 CPU-GPU overlap） |

**`measure_phased` 的 synchronize 开销**：每次 `synchronize` 迫使 CPU 等待 GPU 完成，
破坏了正常训练中的 CPU-GPU 流水线。因此 `step_ms` 会比实际训练慢。
但这不影响内存测量的准确性——测量本身需要在确定 GPU 状态后读取。

---

## 十一、完整调用链

```
# ====== 运行时测量 ======
model = LlamaForCausalLM(config).cuda().train()
compiled = torch.compile(model, backend="inductor")
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
forward_fn = lambda: compiled(input_ids=ids, labels=ids).loss

measure_phased("llama_inductor", forward_fn, optimizer)
    │
    ├── warmup × 3
    │     └── zero_grad → forward → backward → step
    │
    ├── repeats × 5
    │     ├── empty_cache()
    │     ├── zero_grad(set_to_none=True)
    │     ├── synchronize() → base = memory_allocated()
    │     │
    │     ├── reset_peak_memory_stats()                 # FW peak 从 base 开始
    │     ├── ev_s.record() → forward_fn() → synchronize()
    │     ├── fw_peak = max_memory_allocated()
    │     ├── after_fw = memory_allocated()
    │     │
    │     ├── reset_peak_memory_stats()                 # BW peak 从 after_fw 开始
    │     ├── ev_f.record() → loss.backward() → synchronize()
    │     ├── bw_peak = max_memory_allocated()
    │     ├── after_bw = memory_allocated()
    │     │
    │     ├── reset_peak_memory_stats()                 # OPT peak 从 after_bw 开始
    │     ├── ev_b.record() → optimizer.step()
    │     ├── ev_e.record() → synchronize()
    │     └── opt_peak = max_memory_allocated()
    │
    └── IQR mean 聚合
          fw_peak  = iqr_mean([r.fw_peak  for r in results])
          bw_peak  = iqr_mean([r.bw_peak  for r in results])
          opt_peak = iqr_mean([r.opt_peak for r in results])
          overall  = max(fw, bw, opt)
          peak_phase = {fw: "FW", bw: "BW", opt: "OPT"}[overall]

# ====== 仿真精度验证 ======
static_result = estimate_training_peak(fw_gm, bw_gm, model)
runtime_result = measure_phased(...)

validate(static_result, runtime_result)
    │
    ├── mre = |static_peak - runtime_peak| / runtime_peak
    ├── direction = "over" / "under"
    ├── 分阶段 MRE: mre_fw, mre_bw, mre_opt
    └── 误差归因: dark_base / activation / allocator_overhead

# ====== 诊断工具 ======
capture_snapshot(forward_fn, optimizer, "snapshot.pickle")
    └── _record_memory_history → 执行 → _dump_snapshot
        → 用 pytorch.org/memory_viz 可视化

capture_timeline(forward_fn, optimizer, "trace.json")
    └── torch.profiler.profile → export_chrome_trace
        → 用 Chrome / Perfetto 可视化
```

---

## 十二、源码文件索引

| 文件 | 内容 | 行数 |
|------|------|------|
| `toolkit/profiler/__init__.py` | 模块导出 | 5 行 |
| `toolkit/profiler/step_profiler.py` | `iqr_mean` + `StepResult` + `PhaseResult` + `measure_step` + `measure_phased` | 201 行 |
| ~~`toolkit/profiler/snapshot.py`~~ | `capture_snapshot` — 已删除 | - |
| ~~`toolkit/profiler/timeline.py`~~ | `capture_timeline` — 已删除 | - |
| `toolkit/profiler/validator.py` | `validate` + `analyze_error_sources` + `ValidationResult` | 97 行 |

---

## 参考

- `08-static-simulation-deep-dive.md`：静态仿真引擎（L1/L2），与本文对标
- `03-experiments.md`：MRE 6.9% 实验数据
- PyTorch docs: [CUDA Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)
- PyTorch docs: [Memory Snapshot](https://pytorch.org/docs/stable/torch_cuda_memory.html)
- PyTorch docs: [torch.profiler](https://pytorch.org/docs/stable/profiler.html)
