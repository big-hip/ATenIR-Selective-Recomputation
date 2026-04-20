# 04 — 开发日志与 Bug 追踪

> **文档定位**: 完整开发记录——Phase 1-10 实施过程、Bug 清单(B1-B16/W1-W23)、验证数据。
> 合并自: `.windsurf/plans/` 系列计划文件 + 各 Phase 验证报告
>
> **论文对应**: 第 2 章附录（开发日志参考）
> **注意**: Phase 9 Web UI 已废弃，不纳入论文。Phase 11+ 仅作历史参考。

---

## 一、Phase 总览

| Phase | 内容 | 优先级 | GPU | 状态 |
|-------|------|--------|-----|------|
| **1** | `utils/` + 正确性测试 | P0 | ❌ | ✅ is_view_node 4.8% 修正, SymInt 0 失败 |
| **2** | `models/` — 3 模型注册 | P0 | ❌ | ✅ GPT-2+LLaMA+Mistral |
| **3** | `capture/` — 图捕获 | P0 | ✅ | ✅ SymInt 100%, loss_fn 差异 30.6% |
| **4** | `simulation/` — L1/L2 仿真 | P0 | ❌ | ✅ L2 MRE 2.2%(归档)→6.6%(compiled) |
| **5** | `strategy/` — 重计算策略 | P1 | ✅ | ✅ AC -24.3%, min_cut 需修基线 |
| **6** | `profiler/` — 运行时分析 | P1 | ✅ | ✅ 5 组件验证, L2 MRE 6.6% |
| **7** | `output/` — 输出层 | P1 | ❌ | ✅ console/charts/export |
| **8** | examples — 端到端示例 | P2 | ✅ | ✅ ex1/ex2/ex3 |
| **9** | `web/` — Gradio UI v1 | P2 | ✅ | ✅ 已废弃，不纳入论文 |
| **10.0** | Bug 修复 B1-B16 + 输出重构 | P0 | ✅ | ✅ 83 tests 全通过 |
| **10.1** | 策略对比修正 | P0 | ✅ | ✅ 已完成 |
| **11** | Web UI v2 重构 | P2 | ✅ | ⬜ 已取消 |
| **12** | 论文数据收集 | P1 | ✅ | ✅ 3 组实验 + F1-F7 图表 |

---

## 二、P0 Bug 清单（正确性）

### B1: View Op 检测遗漏 `getitem` → 显存过估 4.8% ✅

**根因**: `_is_view_op()` 使用硬编码 `_VIEW_OP_NAMES`，遗漏 `operator.getitem`。
getitem 节点从 `native_layer_norm`/`native_dropout`/`split` 等解包时共享存储。

**实验数据** (GPT-2, n_embd=256, n_layer=4, batch=4, seq=128):
```
旧方法 FW alloc: 230.3 MB
新方法 FW alloc: 219.8 MB → 修正 10.5 MB (4.8%)
分歧节点: 39 个 getitem，全部被正确识别为 view
```

**修复**: `is_view_node()` 使用 `untyped_storage()._cdata` storage aliasing 检测。
- `data_ptr()` 对 FakeTensor 返回 0（不可用）
- `_cdata` 正确区分 view vs alloc

### B2: 训练峰值公式错误 ✅

**旧公式**: `max(fw_peak, saved_act + bw_peak)` — `bw_peak` 已含 saved_act，重复计算 72-102 MB。

**正确公式**: `param + grad + optim + max(fw_peak, bw_peak)`
- `bw_peak` 天然含 saved activations（BW placeholder）
- `grad_bytes` 必须保留（BW 期间梯度在图外累积）
- v4.2 升级: `true_peak = max(fw_peak, bw_peak, opt_peak)`

**实测 MRE** (公式修正后, 3 种 GPT-2 规模):

| Config | Est | Compiled RT | MRE |
|--------|-----|-------------|-----|
| emb=128, L=2 | 286.7 MB | 305.6 MB | **-6.2%** |
| emb=256, L=4 | 663.3 MB | 671.2 MB | **-1.2%** |
| emb=512, L=6 | 1095.0 MB | 1116.8 MB | **-2.0%** |

### B3: `val_bytes` 静默吞异常 ✅

**旧代码**: `int(SymInt)` 失败时 `return 0` — 节点不参与内存计算。
**修复**: 使用 `hint_int(val.numel(), fallback=4096)`（与 PyTorch `_size_of` 一致）。
Phase 1 实测: 179 个 SymInt 全部成功，0 失败。

### B4: `base_allocated` 计算错误 ✅

`sum([base] * repeats) / repeats` 恒等于最后一次 `base`。修复: 循环内逐次收集再取均值。

### B5: `dynamic=True` + `loss_fn` 不一致 ✅

不同 `dynamic` 设置产生不同图结构。统一 `dynamic=True`。
CE loss vs sum loss: BW placeholder 差异 **30.6%**（57.5 vs 82.8 MB）→ 统一 `loss_fn`。

### B6: 度量聚合不一致 ✅

`verify_builtin_recompute.py` 用 IQR mean，`verify_static_vs_runtime.py` 用简单均值。统一为 IQR mean。

### B10: `capture_graphs` 不传 labels — L2 MRE 50%→6.9% ✅

**★ 项目最关键 bug。**

**根因**: `compiled(sample_input_ids)` 不传 labels，FW 图缺 CE loss 节点，外部用 `logits.sum()` 替代。

| 路径 | FW 节点 | BW 节点 | L2 peak | vs runtime MRE |
|------|---------|---------|---------|----------------|
| `logits.sum()` | 211 | 306 | 160 MB | **54.9%** |
| CE loss (labels) | 223 | 321 | 208 MB | **41.3%** |

**修复**: `capture_graphs` 增加 `model_kwargs` 参数，调用: `capture_graphs(model, ids, lambda out: out.loss, model_kwargs={"labels": ids})`

**B10 修复后隔离进程测量**:

| 模型 | L2 Peak | Runtime | MRE | 方向 |
|------|---------|---------|-----|------|
| GPT-2 | 208.5 MB | 224.0 MB | **6.9%** | under |
| LLaMA | 201.8 MB | 188.7 MB | **6.9%** | over |
| Mistral | 201.0 MB | 188.1 MB | **6.9%** | over |

---

## 三、P1 Bug 清单

### B11: `console._stringify` 误将非字节整数格式化 ✅

`_stringify(50257)` → `"49.1 KB"` 而非 `50257`。修复: key-based `BYTE_KEYS` 集合检测。

### B12: L1 公式 LLaMA/Mistral MLP 少算 1/3 参数 ✅

MLP 公式 `2*H*I` → LLaMA/Mistral 有 gate_proj+up_proj+down_proj = `3*H*I`。
修复后 L1 MRE 显著降低。

### B13: `summarize_model` kv_heads 显示错误 ✅

非 mistral 模型始终显示默认 heads 值。修复: `name in ("mistral", "llama")`。

### B14: `measure_phased` 使用 median 而非 iqr_mean ✅

与 `measure_step` 统计口径不一致。修复: 统一 iqr_mean 聚合。

### B15: `capture_graphs` 缺少 finally 中的 dynamo.reset() ✅

异常时全局状态不重置。修复: try/finally 包裹。

### B16: `_normalize_row` 重复定义 ✅

console.py 和 export.py 各有一份。修复: 提取到 `utils/formatting.py`。

---

## 四、Web UI Bug 清单 (B1/B2/B3-B6 + W1-W23)

### P0（已修复）

| ID | 问题 | 修复 |
|----|------|------|
| B1/W1/W2 | Tab2-5 完全忽略 Tab1 模型配置 | gr.State 全局共享 + shared_controls |
| B2/W13 | 策略柱状图 X 轴显示 "0","1","2" | charts.py 增加 "strategy" key |
| W3 | 所有 byte 值原始整数不可读 | format_result_bytes() 格式化 |
| W4 | Simulation DataFrame 61% NaN | 统一对比表（指标/L1/L2/Runtime） |

### P1（已修复）

| ID | 问题 | 修复 |
|----|------|------|
| B3/W7 | Slider 范围过小 | batch≤32, seq≤2048, hidden≤4096 |
| B4/W8 | 无输入验证 | hidden%heads 校验 |
| B5/W16 | matplotlib 内存泄漏 | plt.close(fig) |
| B6/W10 | LLaMA 不支持 kv_heads | mistral+llama 均设置 |
| W5 | MRE 显示为小数 | 转为百分比 "6.9%" |
| W6 | Runtime 单行横向表 | 垂直 key-value 格式 |
| W9 | Mode 名称不友好 | 中文标签 |
| W14 | 缺少 stacked bar 分解图 | param/grad/optim/act 堆叠图 |
| W15 | MRE 折线图只有 2 点 | 改为 bar chart |
| W18 | 错误信息以原始 JSON 展示 | 中文友好提示 |
| W22 | 策略缺节省百分比列 | vs_baseline 列 |

### P2（Phase 11 处理，已暂停）

| ID | 问题 |
|----|------|
| W11 | Capture 页面无 loss 类型选择 |
| W12 | 缺预设按钮 (GPT-2 124M, LLaMA-7B) |
| W17 | 无运行进度指示 |
| W19 | Artifact 无下载按钮 |
| W20 | 缺一键完整分析 |
| W21 | 无运行历史对比 |
| W23 | IR 纯文本无语法高亮 → gr.Code(language="python") |

---

## 五、Phase 1-6 验证数据

### Phase 1: utils/ ✅

- `is_view_node()` storage aliasing 修正 4.8% 过估（39 个 getitem）
- `val_bytes()` + `hint_int`: 179 个 SymInt 全部解析成功
- `make_boxed_func` 是 `aot_module_simplified` 推荐包装方式
- 图仿真对标 compiled（非 eager），eager 比 compiled 少用 ~20% 内存

### Phase 2: models/ ✅

- GPT-2 + LLaMA + **Mistral** 离线创建 + CPU forward + `.loss` 成功
- OPT: 3 graph break → 替换为 Mistral
- Bloom: `OpaqueUnaryFn_log2`; Falcon: 6 break; GPT-NeoX: deepcopy 冲突
- `named_parameters()` 自动去重 weight tying

### Phase 3: capture/ ✅

- 3 模型 AOT capture 成功，SymInt 100% resolvable
  - GPT-2: 295/0, LLaMA: 356/0, Mistral: 1011/0
- `dynamic=True` 必须通过 `torch.compile` 传递
- CE vs sum loss: BW placeholder 差异 **30.6%**
- CPU vs GPU 捕获差异 < 1%

### Phase 4: simulation/ ✅

- **L2 MRE = 2.2%**（归档口径, 3 模型 eager 基线）
  - GPT-2: 0.9%, LLaMA: 2.7%, Mistral: 2.9%
- **L1 MRE = 44.7%**（LLaMA/Mistral 参数公式不完整）
- Bug-2 回归: 旧公式过估 72-102 MB，新公式消除重复
- 3 模型静态图: `bw_peak >= fw_peak`（不等于运行时阶段判断）

### Phase 5: strategy/ ✅

验证配置: batch=8, seq=128, hidden=512, n_layer=6

- **Classic AC**: GPT-2 peak **-24.3%** (1558→1179 MB), 3 模型 FW peak 降低 12-25%
- **分阶段峰值**: 3 模型峰值均在 FW 阶段
- **L2 + recomputation MRE = 7.5%**
- **Memory Budget**: PyTorch 2.6 可设置且可编译
- SAC eager 有 overhead，需 Dynamo 路径

**min_cut 对比基线修正** (GPT-2, n_layer=6, hidden=512, batch=8, seq=128):

| 策略 | Peak (MB) | vs eager | vs compiled_default |
|------|-----------|----------|---------------------|
| eager_baseline | 1558 | — | — |
| compiled_default | 1745 | +12.0% | — |
| compiled_min_cut | 1652 | +6.0% | **-5.3%** |
| classic_ac | 1179 | **-24.3%** | — |

### Phase 6: profiler/ ✅

验证配置: batch=2, seq=64

- **StepResult**: IQR mean 稳定（两次运行 diff=0.00%）
- **PhaseResult**: 分阶段峰值验证通过，小 batch 下 Opt peak 主导
- **L2 vs compiled MRE**:

| 模型 | L2 Peak | Compiled RT | MRE | 方向 |
|------|---------|-------------|-----|------|
| GPT-2 | 370.9 MB | 432.8 MB | 14.3% | under |
| LLaMA | 386.7 MB | 397.6 MB | 2.7% | under |
| Mistral | 381.2 MB | 392.6 MB | 2.9% | under |
| **平均** | | | **6.6%** | |

> GPT-2 MRE 14.3% 偏高; B10 修复后降至 6.9%。

- **Snapshot**: `_record_memory_history` + `_dump_snapshot` 可用
- **Timeline**: Chrome Trace 可用，Memory Timeline 需 try/except

---

## 六、Phase 7-9 实施记录

### Phase 7: output/ ✅

- console (tabulate), charts (matplotlib, Agg 后端), export (JSON/CSV/HTML)
- 修复: FuncFormatter 替代 set_yticklabels

### Phase 8: examples/ ✅

- ex1_multi_model_capture, ex2_strategy_comparison, ex3_simulation_accuracy
- 待优化: ex2 缺 compiled_default 对照; ex3 配置范围过小

### Phase 9: web/ Gradio UI v1 ✅

- 5-tab 布局, pydantic v2 兼容补丁
- 23/23 tests + 6/6 HTTP API
- 环境修复: charset-normalizer + gradio_client

---

## 七、Phase 10.0 执行记录（2025-04-14）

### 执行顺序

```
核心公式/API 修复:
  B10 → B15 → B12 → B11 → B14 → B16
Web UI 输入修复:
  B1/W1/W2 → B3/W7 → B4/W8 → B6/W10 → B13 → W9 → W18
Web UI 输出重构:
  W3 → W4 → W5 → W6 → B2/W13 → W14 → W15 → B5/W16 → W22
  → W23 (gr.Code)
```

### 修改文件清单（14 文件）

- `toolkit/capture/aot_capture.py` — B10 model_kwargs + B15 try/finally
- `toolkit/simulation/config_estimator.py` — B12 MLP 3×H×I
- `toolkit/profiler/step_profiler.py` — B14 iqr_mean
- `toolkit/utils/formatting.py` — B16 normalize_row 统一
- `toolkit/utils/__init__.py` — 导出 normalize_row
- `toolkit/output/console.py` — B11 key-based BYTE_KEYS
- `toolkit/output/charts.py` — B2 "strategy" key + B5 plt.close
- `toolkit/output/export.py` — B16 导入 normalize_row
- ~~`toolkit/web/`~~ — B1/W 系列 Web UI 修复（v6.1 已删除整个 web 目录）
- ~~`tests/test_web.py`~~ — 33 tests（v6.1 随 web 目录一同删除）

### 验收结果

- **106 tests 全部通过**（0 regressions）
- B10 修复: L2 MRE 50%→**6.9%**
- B1 修复: Tab2-5 使用 Tab1 配置
- 所有 byte 值格式化显示
- Simulation 对比表 0% NaN

---

## 八、v3.5 深度审计关键发现

### Dark Memory 分析

| 模型 | param | formula_base | runtime_base | dark_base | dark/peak |
|------|-------|-------------|-------------|----------|----------|
| GPT-2 | 26.3 MB | 78.9 MB | 95.7 MB | **16.8 MB** | 7.5% |
| LLaMA | 33.3 MB | 99.8 MB | 118.8 MB | **19.0 MB** | 10.1% |
| Mistral | 33.1 MB | 99.4 MB | 118.4 MB | **19.0 MB** | 10.1% |

组成: CUDA 运行时上下文 ~16 MB + 模型 buffers 0.5-1.3 MB + allocator 对齐 ~1-2 MB。
大模型 (7B+) dark_base/peak < 0.1%，可忽略。论文中明确说明。

### 编译开销分析

| 模型 | eager_base | compiled_base | 编译开销 |
|------|-----------|--------------|---------|
| GPT-2 | 120.6 MB | 120.6 MB | **0 B** |
| LLaMA | 133.9 MB | 135.0 MB | **1.1 MB** |
| Mistral | 134.5 MB | 134.5 MB | **0 B** |

**结论**: `torch.compile` + AOTAutograd 编译开销几乎为零。

---

## 九、API 审计摘要（PyTorch 2.6）

| API | 状态 | 注意事项 |
|-----|------|---------|
| `checkpoint(use_reentrant=False)` | ✅ 稳定 | — |
| `create_selective_checkpoint_contexts` | ✅ eager | SAC+compile 在 2.6 崩溃 |
| `activation_memory_budget` | ⚠️ 实验性 | `torch._functorch.config`, Phase 5 可用 |
| `aot_module_simplified` | ✅ 稳定 | 核心入口 |
| `default_partition` / `min_cut_rematerialization_partition` | ✅ | — |
| `FakeTensorMode(allow_non_fake_inputs=True)` | ✅ | CPU 模型 |
| `_record_memory_history` / `_dump_snapshot` | ✅ internal | 封装+版本检测 |
| `export_memory_timeline` | ⚠️ | 可能 ValueError, 需 try/except |

---

## 十、风险与缓解（截至 Phase 10.0）

| 风险 | 级别 | 缓解 | 状态 |
|------|------|------|------|
| B10 capture 不传 labels | P0 | model_kwargs 参数 | ✅ 已修 |
| B1 Tab 间配置不共享 | P0 | gr.State | ✅ 已修 |
| min_cut 对比基线错误 | P0 | compiled_default | ⬜ Phase A |
| B12 L1 MLP 缺 gate_proj | P1 | 分模型 2/3 投影 | ✅ 已修 |
| `_cdata` 未来变化 | 低 | 封装在 is_view_node() | 未变 |
| SAC eager overhead | 中 | 需 Dynamo 路径 | P5 确认 |
| dark_base gap | 低 | 论文说明 | 已量化 |
| Gradio 3.24 + pydantic v2 | 高 | PredictBody 补丁 | ✅ 已修 |
