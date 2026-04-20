# 01 — 架构设计与技术决策

> **文档定位**: 项目架构全景——四支柱闭环、五种策略、核心 API、仿真细节、设计原则。
> 合并自: `.windsurf/plans/02-architecture-99269f.md` + `.windsurf/plans/11-consolidated-技术路线.md`

---

## 一、四支柱闭环架构

```
            ┌──────────────┐
            │  Pillar 1    │  strategy/: AC / SAC / Budget / partition_fn
            │  策略注入层   │
            └──────┬───────┘
                   │ 变换后的模型
                   ▼
            ┌──────────────┐
            │  Pillar 2    │  capture/: AOT compile → FW/BW GraphModule
            │  图捕获层     │  models/: 多模型注册 + 离线创建
            └──────┬───────┘
                   │ FX 图 + FakeTensor meta
                   ▼
            ┌──────────────┐
            │  Pillar 3    │  simulation/: L1 公式 → L2 事件驱动
            │  仿真引擎     │  (L3 BFC / L4 CPU Replay → 未来)
            └──────┬───────┘
                   │ 估算结果
                   ▼
            ┌──────────────┐
            │  Pillar 4    │  profiler/: GPU 运行时 ground truth
            │  验证层       │  output/: 表格 + 图表 + 导出
            └──────────────┘  web/: Gradio UI (可选)
```

核心价值：**不需要真正跑完训练，就能估算不同重计算策略下的峰值显存**。

| 维度 | 选择 | 依据 |
|------|------|------|
| 图捕获 | `torch.compile` + `aot_module_simplified` | PyTorch 2.x 官方方向，与 TorchTitan 一致 |
| 仿真精度 | L2 事件驱动（FX 图遍历 + 512B 对齐） | MRE **6.9%** (B10修后, compiled 口径) |
| 运行时基线 | **仅 compiled**（不用 eager） | 图仿真天然对标 AOT compiled 执行路径 |
| 首批模型 | GPT-2 + LLaMA + Mistral | 0 graph break，覆盖 learned pos / RoPE / GQA |
| 聚合方式 | IQR mean | 两次运行 diff=0.00%，消除异常值 |

---

## 二、五种重计算策略

| 策略 | 落点 | 实现技术 | 影响对象 | 实测结论 |
|------|------|---------|---------|---------|
| baseline | graph partition 层 | `default_partition` | 保存全部中间激活 | 作为无重计算基线 |
| Classic AC | module/block 层 | `checkpoint(use_reentrant=False)` | 整个 TransformerBlock 在 BW 重算 | FW peak -12~25% |
| SAC | op policy 层 | `create_selective_checkpoint_contexts` | 按 op 类型选择 save/recompute | eager 有 overhead，需 Dynamo 路径 |
| min_cut | graph partition 层 | `min_cut_rematerialization_partition` | 编译器自动减少 saved activations | saved_act -12~20%，L2 peak -6~9% |
| Memory Budget | compiler config 层 | `activation_memory_budget` | 以预算约束自动搜索重计算点 | PyTorch 2.6 可设置且可编译 |

- **Classic AC**：通过包裹 `block_class_name` 对应的模块，把整块前向变成 checkpoint region。优点是实现最稳、效果直观；缺点是粒度粗，额外算力开销也最大。
- **SAC**：不是整块重算，而是给算子打 policy。比如对 `mm/bmm/addmm` 这类 matmul 选择 `MUST_SAVE`，对其它轻量 op 偏向 `PREFER_RECOMPUTE`。
- **min_cut**：不是用户手写 checkpoint，而是 AOT Autograd 在 joint graph 上做 partition。它决定哪些 tensor 留在 FW output、哪些在 BW 重算，是 toolkit 最重要的"编译器级"重计算路径。
- **Memory Budget**：是 `min_cut` 之上的更高层接口，本质仍依赖 AOT graph + partitioner，只是把"保多少激活"变成一个可调预算。

---

## 三、图捕获完整流程

1. **准备模型与输入**：通过 `ModelRegistry` 离线创建 GPT-2 / LLaMA / Mistral，准备 `input_ids`。
2. **先注入策略，再做 capture**：AC/SAC/min_cut 会改变 joint graph 和 saved activations。
3. **定义 AOT backend**：`torch.compile(model, backend=_backend, dynamic=True)`；内层调 `aot_module_simplified`。
4. **包装 fw/bw compiler**：返回 `make_boxed_func(g.forward)`，同时深拷贝保存 `fw_gm`/`bw_gm`。
5. **触发一次真实执行**：`out = compiled(input_ids=..., **model_kwargs)`，外部计算 `loss = loss_fn(out)` + `backward()`。
6. **AOT 执行 joint → partition → 拆分**：根据 partition_fn 划分，提取 FW/BW 两个 GraphModule。
7. **得到仿真所需结构**：
   - FW `output` = 用户输出 + saved activations
   - BW `placeholder` = saved activations + tangent inputs
   - 这正是 `bw_peak` 天然包含 `saved_act` 的根本原因。

---

## 四、目录结构

```
ATenIR-Selective-Recomputation/
├── toolkit/                     # ★ 核心框架
│   ├── __init__.py              # __version__ = "0.1.0"
│   ├── models/                  # 多模型支持
│   │   ├── registry.py          # ModelSpec + SMALL_PRESETS + BLOCK_CLASS_MAP
│   │   └── adapters.py          # get_hidden / get_num_layers / get_num_heads / get_intermediate
│   ├── capture/                 # 图捕获
│   │   ├── aot_capture.py       # capture_graphs(model, ids, loss_fn, model_kwargs=...)
│   │   ├── analysis.py          # graph_stats / analyze_graph / count_fw_outputs
│   │   └── inductor_capture.py  # capture_inductor_graphs() — L3 Scheduler hook
│   ├── strategy/                # 重计算策略
│   │   ├── classic_ac.py        # wrap_with_checkpoint / unwrap_checkpoint
│   │   ├── sac.py               # SAC_POLICIES + wrap_with_sac
│   │   ├── partition.py         # get_partition_fn(default|min_cut)
│   │   └── memory_budget.py     # Budget API wrapper (try/except 降级)
│   ├── simulation/              # 静态仿真引擎
│   │   ├── config_estimator.py  # L1: 公式推导 (per-model, 含 fused_optimizer)
│   │   └── graph_estimator.py   # L2: 事件驱动 + storage aliasing + 512B align
│   ├── profiler/                # 运行时分析 + 验证
│   │   ├── step_profiler.py     # StepResult + PhaseResult + measure_step/measure_phased
│   │   ├── snapshot.py          # Memory Snapshot (.pickle)
│   │   ├── timeline.py          # Chrome Trace + Memory Timeline
│   │   └── validator.py         # MRE 计算 + 分阶段 MRE + 误差来源分析
│   ├── output/                  # 输出层
│   │   ├── console.py           # tabulate 表格 (key-based 字节格式化)
│   │   ├── charts.py            # matplotlib 图表 (bar/stacked/heatmap)
│   │   └── export.py            # JSON / CSV / HTML 导出
│   ├── web/                     # Gradio UI
│   │   ├── app.py               # Blocks 主入口 (gr.State 全局配置)
│   │   ├── helpers.py           # format_result_bytes / build_model / validate
│   │   └── pages/               # 5 Tab 功能页面
│   └── utils/                   # 共享工具
│       ├── view_ops.py          # is_view_node() — storage aliasing via _cdata
│       ├── tensor_utils.py      # val_bytes() — hint_int + tuple 递归
│       └── formatting.py        # fmt() / format_bytes() / normalize_row()
├── toolkit_examples/            # 端到端示例
│   ├── ex1_multi_model_capture.py
│   ├── ex2_strategy_comparison.py
│   └── ex3_simulation_accuracy.py
├── tests/                       # 106 tests (Phase 10.0 后全部通过)
└── docs/                        # 本文档集
```

---

## 五、核心 API 设计

### 5.1 `models/`

```python
@dataclass
class ModelSpec:
    config_cls: type
    model_cls: type
    block_class_name: str
    loss_fn: Callable  # (model_output) -> scalar loss
    small_preset: dict

class ModelRegistry:
    def create_model(self, name: str, **overrides) -> nn.Module
    def get_config(self, name: str, **overrides) -> PretrainedConfig
    def default_loss_fn(self, name: str) -> Callable
    def list_models(self) -> list[str]  # ["gpt2", "llama", "mistral"]
```

> **`loss_fn` 的核心地位**: `capture_graphs()`、`measure_step()`、`validate()` 全部依赖同一 `loss_fn`
> 来保证图结构与运行时行为一致。`loss_fn` 只负责"从 model output 提取 scalar loss"，
> 不负责再次调用 model。

### 5.2 `capture/aot_capture.py`

```python
def capture_graphs(
    model: nn.Module,
    sample_input_ids: torch.Tensor,
    loss_fn: Callable,
    *,
    model_kwargs: dict | None = None,  # 传递 labels 等额外参数
    partition_fn=default_partition,
    dynamic: bool = True,
) -> tuple[fx.GraphModule, fx.GraphModule]:
    """AOT compile -> (fw_gm, bw_gm).
    B10 修复: model_kwargs 允许传 labels，使图包含 CE loss 节点。
    B15 修复: try/finally 确保 torch._dynamo.reset() 清理。
    标准调用: capture_graphs(model, ids, lambda out: out.loss,
                             model_kwargs={"labels": ids})"""
```

### 5.3 `simulation/`

```python
# config_estimator.py — Level 1
def estimate_from_config(
    config, batch, seq, dtype=torch.float32,
    optimizer="adam", fused_optimizer=False,
) -> dict:  # → fw_peak, bw_peak, opt_peak, fwbw_peak, true_peak, peak_phase, ...

# graph_estimator.py — Level 2
def estimate_graph_peak(gm, pin_output_inputs=False, align=512) -> dict
def estimate_training_peak(fw_gm, bw_gm, model, optimizer_cls=Adam,
                           fused_optimizer=False) -> dict
```

### 5.4 `profiler/`

```python
@dataclass
class StepResult:
    name: str
    peak_allocated: int      # IQR mean
    peak_reserved: int
    base_allocated: int
    activation_delta: int
    elapsed_ms: float
    fw_ms: float; bw_ms: float; opt_ms: float

@dataclass
class PhaseResult:
    name: str
    fw_peak: int; bw_peak: int; opt_peak: int
    overall_peak: int        # max(fw, bw, opt)
    base_allocated: int
    fw_ms: float; bw_ms: float; opt_ms: float; step_ms: float

def measure_step(name, forward_fn, optimizer, *, repeats=6, warmup=2) -> StepResult
def measure_phased(name, forward_fn, optimizer, *, repeats=5, warmup=3) -> PhaseResult

@dataclass
class ValidationResult:
    tag: str
    static_peak: int; runtime_peak: int; runtime_reserved: int
    mre_allocated: float; mre_reserved: float
    run_mode: str; direction: str; breakdown: dict

def validate(static_result, runtime_result, run_mode="compiled") -> ValidationResult
```

### 5.5 `strategy/`

```python
def wrap_with_checkpoint(model, block_class_name, use_reentrant=False) -> nn.Module
def unwrap_checkpoint(model) -> nn.Module

SAC_POLICIES = {"save_matmuls": ..., "save_attention": ..., "recompute_all": ...}
def wrap_with_sac(model, block_class_name, policy_name="save_matmuls")

def get_partition_fn(name="default") -> Callable  # "default" | "min_cut"
def set_memory_budget(budget: float = 0.5) -> bool
def clear_memory_budget()
```

---

## 六、数据流

```
reg = ModelRegistry()
model = reg.create_model("gpt2", n_layer=4)
loss_fn = reg.default_loss_fn("gpt2")         ★ 统一 loss 来源
    │
    ├── strategy/ 注入 AC/SAC/partition_fn
    │
    ▼
capture_graphs(model, ids, loss_fn, model_kwargs={"labels": ids})
    → (fw_gm, bw_gm)
    │
    ├── graph_stats(fw_gm)
    │
    ▼
estimate_training_peak(fw_gm, bw_gm, model)  → {fw_peak, bw_peak, true_peak, ...}
estimate_from_config(config, batch, seq)      → {fw_peak, bw_peak, true_peak, ...}
    │
    ▼
optimizer = Adam(model.parameters())
forward_fn = lambda: loss_fn(model(ids, labels=ids))
measure_step/measure_phased("run", forward_fn, optimizer)  → StepResult/PhaseResult
    │
    ▼
validate(static_result, runtime_result, run_mode="compiled")  → ValidationResult
    │
    ▼
output/: console table + charts + JSON/CSV/HTML
web/: Gradio UI (optional)
```

> **loss_fn 贯穿路径**: ModelSpec → capture_graphs → measure_step → validate。
> 任何环节使用不同 loss 都会导致 MRE 失真（CE vs sum 差异 30.6%）。

---

## 七、静态仿真技术细节

### 7.1 L1 静态仿真：Config 公式法

L1 不依赖真实图，只依赖模型配置：
- 从 `config` 提取 `hidden_size / n_layer / n_head / intermediate_size / vocab_size`
- 按模型类型区分 GPT-2 的 learned position embedding、LLaMA/Mistral 的 RoPE、Mistral 的 GQA
- B12 修复后：LLaMA/Mistral MLP 使用 `3*H*I`（gate+up+down），GPT-2 使用 `2*H*I`
- 支持 `fused_optimizer` 参数：fused Adam 时 `opt_temp=0`，否则 `opt_temp=param_bytes`
- 输出四峰值 + peak_phase + timeline 采样点

L1 定位是**粗估**，不是最终 MRE 主结论来源。

### 7.2 L2 静态仿真：FX 图事件驱动

L2 的输入是 `fw_gm` / `bw_gm` 两张带 `meta['val']` 的 FX 图，算法流程：

1. **按执行顺序遍历节点**：以 `GraphModule.graph.nodes` 的顺序近似运行时执行顺序。
2. **计算节点字节数**：通过 `val_bytes(node.meta['val'])` 读取 FakeTensor 的 `numel × element_size`，对 tuple/list 递归求和，用 `hint_int` 处理 SymInt。
3. **识别零分配节点**：如果 `is_view_node(node)` 为真（`_cdata` storage aliasing 检测），则该节点只做 alias，不增加 `current_bytes`。
4. **对齐到 allocator 粒度**：对真实分配节点做 512B 对齐 `aligned = (nb + 511) & ~511`。
5. **分配与释放**：分配时把 `aligned` 加到 `current_bytes`；维护 use-count，在某个 tensor 的最后一个 consumer 之后释放。
6. **处理 saved activations**：对 FW 图启用 `pin_output_inputs=True`，不立即释放 output 所依赖的张量，模拟 AOT saved activations 穿越到 BW。
7. **记录时间线与峰值**：每个节点都更新 `timeline` 和 `peak_bytes`，最终得到 `fw_peak` / `bw_peak`。
8. **拼出训练步总峰值**（四峰值体系）：
   ```python
   fw_peak  = static_base + fw_graph_peak
   bw_peak  = static_base + grad_bytes + bw_graph_peak
   opt_peak = static_base + grad_bytes + opt_temp
   fwbw_peak = max(fw_peak, bw_peak)
   true_peak = max(fw_peak, bw_peak, opt_peak)
   ```

### 7.3 L2 为什么能比较准，但还不是 L3/L4

- **已覆盖**：op 级 activation 生命周期、view aliasing、SymInt、512B 对齐
- **尚未覆盖**：segment 粒度、碎片化、best-fit 复用、实际调度顺序、reserved memory 的缓存效应
- 当前结果应理解为**高精度 L2**，而不是 allocator 级 L3/L4

### 7.4 动态侧验证：compiled ground truth

动态侧不是另一套仿真，而是直接测 GPU 真实运行：

1. 复用静态侧同一配置：相同模型、输入、loss_fn、策略
2. 先 warmup 让 allocator/cache 稳定
3. 每次重复前 reset：`empty_cache()`、`reset_peak_memory_stats()`、`zero_grad(set_to_none=True)`
4. 记录 base 与分段时间：CUDA Event 包围 FW/BW/Opt
5. 读取真实峰值：`torch.cuda.memory_stats()` 的 `allocated_bytes.all.peak`
6. 鲁棒聚合：IQR mean
7. 定位峰值阶段：FW/BW/Opt 间分别 reset peak → PhaseResult
8. 静态-动态对齐：`validate()` 计算 MRE 并分析误差来源
9. 辅助观测：通过 `torch.cuda.memory_stats()` 观察 allocator 行为

### 7.5 仿真精度路线图

| Level | 方法 | 预期 MRE | 实测 MRE | 对标 |
|-------|------|---------|---------|------|
| **L1** | Config 公式推导 | ~20-30% | 44.7%→B12修后降低 | DNNMem |
| **L2** | FX 图事件驱动 + 512B | ~10-15% | **6.9%** (B10修后) | Inductor |
| L3 | + BFC Simulator | ~5-8% | 待实现 | LLMem |
| L4 | CPU 执行 + BFC 重放 | ~3-5% | 待实现 | xMem |

---

## 八、九条关键设计原则

### 1. View 检测: Storage Aliasing (`_cdata`)
- **替代**: 硬编码 `_VIEW_OP_NAMES`（遗漏 `operator.getitem`，过估 4.8%）
- **验证**: Phase 1 实测 39 个 getitem 全正确识别
- **代码**: `val.untyped_storage()._cdata == inp_val.untyped_storage()._cdata`

### 2. SymInt 处理: `hint_int(fallback=4096)`
- **替代**: `int(SymInt)` + `except: return 0`（静默吞异常）
- **验证**: Phase 1 实测 179 个 SymInt 0 失败

### 3. 图捕获: `torch.compile(backend=_backend, dynamic=True)`
- `dynamic=True` 通过 `torch.compile` 传递（`aot_module_simplified` 无此参数）
- `make_boxed_func(gm.forward)` 必须使用（否则 UserWarning）
- `loss` 在 `torch.compile` 外部计算

### 4. 峰值公式: 四峰值体系
- `bw_peak` 天然含 saved_act（BW placeholder），不需单独 `saved_act` 项
- `grad_bytes` 必须保留（BW 期间图外累积）
- 旧公式 `saved_act + bw_peak` 重复计算 72-102 MB
- v4.2 升级: `true_peak = max(fw_peak, bw_peak, opt_peak)`

### 5. 对标 compiled（不用 eager 当 baseline）
- compiled 比 eager 多用 ~20% 内存（GPT-2: 670 vs 576 MB）
- `estimate_graph_peak` 输出天然对应 compiled 执行路径

### 6. 统一 `loss_fn`
- CE loss vs sum loss: BW placeholder 差异 **30.6%**（57.5 vs 82.8 MB）
- `ModelSpec.loss_fn` 贯穿 capture → profiler → validator

### 7. IQR Mean 聚合
- 排序后取 [Q1, Q3] 区间均值，消除 GPU warmup/GC 异常值
- Phase 6 验证: 两次运行 peak 完全一致

### 8. 分阶段峰值测量 (PhaseResult)
- FW/BW/Opt 各阶段独立 `reset_peak_memory_stats`
- 大 batch 下 FW peak 主导，AC 可降 FW peak 12-25%
- 小 batch 下 Opt peak 主导（优化器 2x param）

### 9. 模型选型: GPT-2 + LLaMA + Mistral
- OPT: 3 graph break（动态控制流）
- Bloom: `OpaqueUnaryFn_log2` 错误（ALiBi）
- Falcon: 6 graph break
- GPT-NeoX: deepcopy 设备冲突

---

## 九、工业界定位

```
精度低                                              精度高
├── DeepSpeed 公式法 ──── DNNMem ──── 我们(ATen IR L2) ──── xMem ──── LLMem
│   (不算 activation)  (MRE ~19%)   (MRE 6.9%)        (CPU trace)  (需 GPU)
│
│   不需要 GPU ──────────────────────────────────────── 需要 GPU ───
```

- 与 PyTorch 2.x 官方方向一致（AOT + FX，对标 TorchTitan）
- 比 DeepSpeed 公式法精确得多，比 xMem 更轻量（不需实际运行）
- 详见 `02-research.md` §八

---

## 十、Web UI 设计

### v1 现状（Phase 9+10.0，已完成）

| Tab | 功能 | 输出 |
|-----|------|------|
| 模型选择 | 选择模型 + 参数配置（gr.State 全局共享） | 模型摘要 |
| 图捕获 | partition_fn → FW/BW 图 | IR (gr.Code) + graph_stats 表 |
| 策略对比 | 多选策略 → 对比 | 对比表 + 柱状图 (含 vs_baseline %) |
| 仿真分析 | L1/L2 + MRE | 统一对比表 + 分解堆叠图 |
| 运行时 | GPU profiling | 垂直 key-value 表 |

Phase 10.0 已修复核心问题：Tab 间配置共享（B1）、字节格式化（W3）、NaN 消除（W4）、MRE 百分比（W5）等。

### v2 方向（Phase 11，已暂停）

重点增加实验扫描 Tab 和 LaTeX/SVG 导出，面向论文答辩展示。

---

## 十一、Phase 1-6 验证数据汇总

| Phase | 模块 | 核心结论 | 详细报告 |
|-------|------|---------|---------|
| **1** | utils/ | `_cdata` view 4.8% 修正，SymInt 0 失败 | 见 04-dev-log |
| **2** | models/ | 3 模型离线创建，OPT→Mistral | 见 04-dev-log |
| **3** | capture/ | 3 模型 AOT capture，CE vs sum 30.6% | 见 04-dev-log |
| **4** | simulation/ | L2 MRE=2.2%(eager)→6.6%(compiled)→**6.9%**(B10修后) | 见 04-dev-log |
| **5** | strategy/ | AC -12~25%, min_cut -12~20%, MRE 7.5% | 见 04-dev-log |
| **6** | profiler/ | 5 组件全通过, L2 vs compiled MRE 6.6% | 见 04-dev-log |

### MRE 数据（B10 修复后，L2 vs compiled runtime）

| 模型 | L2 Peak | Runtime | MRE | 方向 |
|------|---------|---------|-----|------|
| GPT-2 | 208.5 MB | 224.0 MB | **6.9%** | under |
| LLaMA | 201.8 MB | 188.7 MB | **6.9%** | over |
| Mistral | 201.0 MB | 188.1 MB | **6.9%** | over |

> GPT-2 方向为 under（dark memory 16-19 MB）；LLaMA/Mistral 方向为 over。
> 对大模型 (7B+) dark memory <0.1%，可忽略。

---

## 十二、风险表

| 风险 | 级别 | 缓解 | 状态 |
|------|------|------|------|
| `loss_fn` 签名/labels 注入错位 | P0 | 统一为 `(model_output)->scalar`；CE loss 通过 model_kwargs 注入 labels | ✅ B10 已修 |
| capture 与 profiler 训练路径不一致 | P0 | 统一 model.train()、策略、compiled/dynamic、loss 语义 | ✅ |
| Web UI Tab2-5 忽略 Tab1 配置 | P0 | gr.State 全局配置 | ✅ B1 已修 |
| min_cut 对比基线错误 | P0 | Phase A 增加 compiled_default | ⬜ Phase A |
| L1 MLP 公式缺 gate_proj | P1 | 分模型类型 2/3 投影 | ✅ B12 已修 |
| `_cdata` 未来变化 | 低 | 封装在 `is_view_node()` | 未变 |
| SAC eager overhead | 中 | 需 Dynamo 路径 | P5 确认 |
| dark_base 16-19 MB gap | 低 | 论文中说明 L2 是 lower bound；大模型 <0.1% | 已量化 |
| Gradio 3.24 + pydantic v2 | 高 | app.py PredictBody 补丁 | ✅ 已修复 |
