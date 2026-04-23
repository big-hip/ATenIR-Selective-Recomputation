# 17 — 当前仿真语义修复记录

> **日期**: 2026-04-23  
> **目的**: 记录本轮 P0/P1 修复，明确旧 CSV 失效原因，并给出重跑实验入口。

---

## 一、为什么旧数据不再作为结论

`toolkit_examples/outputs/` 中的旧 CSV 仍可作为历史参考，但不应再用于论文 MRE 结论。原因是旧版 L2.5 存在两类语义风险：

1. L2.5 fusion 使用过宽口径：旧逻辑接近"非 extern 即 fusable"，会把 unknown/materializing op 也纳入融合组，可能过度消除中间 tensor。
2. L2.5 safe reuse 近似过宽：placeholder、graph input、参数/buffer alias、pinned output base 和重计算 BW 图都需要更保守处理。

当前原则是先保证不低估危险场景，再重跑 Runtime/Profile 生成新的论文表格。

---

## 二、本轮 P0 修复

### 2.1 tuple/list view-base pin

`_find_view_base()` 已支持递归检查 input meta 中的 tuple/list tensor。这样 `native_layer_norm -> getitem -> output/consumer` 这类 tuple-producing view 链会继续回溯到真实 storage owner，避免 base tensor 在 output view 仍存活时提前 free。

新增回归测试：

- `tests/test_view_ops.py::test_getitem_output_pins_tuple_producing_base`

### 2.2 L2.5 fusion allowlist

`fusion_ops.py` 新增保守 `FUSABLE_OPS` allowlist。只有简单 unary/binary/activation/mask pointwise 等明确安全的 op 会进入融合组；unknown op 默认 barrier。

当前暂不消除：

- embedding / gather / scatter
- clone / random / layout-changing ops
- 复杂 reduction

新增回归测试：

- `tests/test_fusion_aware.py::test_unknown_materializing_op_not_fusable`

### 2.3 L2.5 safe reuse 收紧

`estimate_graph_peak(..., simulate_inplace=True)` 现在禁止复用：

- placeholder / graph input
- pinned output input 及其 view base
- 参数/buffer alias 节点
- 调用方通过 `no_reuse_nodes` 传入的保护节点

`estimate_inductor_training_peak()` 在 `has_recomputation=True` 时禁用 BW safe reuse，只保留 fusion elimination。`ex_sim_accuracy.py` 与 `ex_model_generalization.py` 对 AC/SAC/budget=0.0 等策略显式标记重计算。

新增/相关测试：

- `tests/test_simulation.py::test_inplace_reuse_does_not_recycle_placeholder`

### 2.4 L3 Scheduler 诊断

L3 现在明确与 L2.5 独立，不再使用 `min(sched_bw, fusion_bw)`。结果中新增：

- `l3_fw_graph_input_bytes`
- `l3_bw_graph_input_bytes`
- `l3_static_base_added`

这些字段用于 GPU 可见环境下校准 Scheduler peak 是否已包含 graph input buffer。如果发现包含，需要在后续版本扣除对应 overlap，避免 `static_base + sched_peak` 双计。

### 2.5 Validator 误差归因

`analyze_error_sources()` 的 fixed error 已纳入 `buffer_bytes`，activation error 改为对齐 absolute phase peak：

- `static_fw_peak - runtime_fw_peak`
- `static_bw_peak - runtime_bw_peak`
- `static_opt_peak - runtime_opt_peak`

Graph-only `act_peak` 保留为内部分析字段，不再作为最终误差来源表的主口径。

---

## 三、本轮 P1 横向对比

新增 `ShapeSum_graph` baseline：

- `shape_sum_fw_bytes`
- `shape_sum_bw_bytes`
- `shape_sum_true_peak`
- `shape_sum_mre`
- `shape_sum_direction`

口径：非 view compute tensor shape bytes 总和 + static base，不做 live-range。它只是 naive shape-inference baseline，不宣称复现 DNNMem/LLMem/xMem。

新增脚本：

- `toolkit_examples/ex_horizontal_comparison.py --medium`

方法链路：

- `L1_config_formula`
- `ShapeSum_graph`
- `L2_live_range`
- `L2.5_fusion_only`
- `L2.5_safe_reuse`
- `L3_scheduler`
- `Runtime`

新增图表：

- `F8_horizontal_methods`
- `F9_l25_ablation`

---

## 四、2026-04-23 重跑结果

本轮已在 GPU 可见环境下完成主要实验重跑并生成 F0-F9 PNG 图表。

| 实验 | 数据文件 | 当前状态 | 关键结论 |
|------|---------|---------|---------|
| 12 策略仿真精度 | `ex_sim_accuracy.csv` | 已重跑 | L2/L2.5/L3 平均 MRE = 8.0% / 4.9% / 9.1% |
| 多模型泛化 | `ex_model_generalization.csv` | 已重跑 | L2/L2.5/L3 overall 平均 MRE = 8.4% / 7.0% / 7.0% |
| 峰值阶段 | `ex_peak_phase.csv` | 已重跑 | eager 多为 FW；Inductor 与 AC/budget 策略多为 BW；Adam 小 batch 可出现 OPT |
| 横向方法 medium | `ex_horizontal_comparison.csv` | 已重跑 medium | L1/ShapeSum/L2/L2.5/L3 平均 MRE = 38.8% / 315.1% / 8.5% / 10.6% / 5.4% |

需要注意：横向方法当前是 GPT-2 medium 配置（8L/512H, B=8, S=256），足以说明 ShapeSum baseline 和方法链路；若论文希望把 F8/F9 作为全模型强定量结论，可额外运行 paper 配置。

测试状态：`pytest -m "not inductor"` 为 83 passed / 12 deselected（约 40 秒）；全量 `pytest tests/ -x -q` 为 95 passed（约 5 分 36 秒）。

---

## 五、推荐复现实验顺序

```bash
conda run -n torch2.6-gpu python -m pytest tests/ -x -q -m "not inductor"
conda run -n torch2.6-gpu python -m pytest tests/ -x -q
conda run -n torch2.6-gpu python toolkit_examples/test_sim_quick.py
conda run -n torch2.6-gpu python toolkit_examples/ex_horizontal_comparison.py --medium
conda run -n torch2.6-gpu python toolkit_examples/ex_sim_accuracy.py
conda run -n torch2.6-gpu python toolkit_examples/ex_model_generalization.py
conda run -n torch2.6-gpu python toolkit_examples/ex_peak_phase.py
conda run -n torch2.6-gpu python toolkit_examples/generate_paper_figures.py
```
