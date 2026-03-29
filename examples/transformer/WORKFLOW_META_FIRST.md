# Transformer Meta-First Workflow

本文档定义当前推荐主链：

1. `meta + FakeTensor` 作为策略迭代与快速仿真主通道。
2. `runtime` 仅用于周期性校准与最终结论确认。

## Why Meta-First

- 速度快：不依赖真实 CUDA kernel 编译与执行。
- 迭代快：可以高频试验策略参数（如 `{"6": 0}`、`{"7": 0.5}`）。
- 风险可控：在图捕获层面就能看到 `saved_values` 变化趋势。

## Known Difference vs Runtime

meta 图和 runtime 图并非完全一致，差异主要来自：

- 随机/种子相关路径（例如 `inductor_random`、`inductor_lookup_seed`）。
- 设备特化算子（例如 attention backward 的 CPU/CUDA 特化差异）。
- 部分 decompositions 在 fake 模式和真实设备模式下路径不同。

结论：

- 可以用 meta 图做策略筛选与快速仿真。
- 不能把 meta 结果直接当作最终运行时性能结论。

## Recommended Cycle

1. `capture static fast`：快速筛策略。
2. `capture static high`：提高图结构与真实训练态一致性。
3. `capture-compare`：抽样对照 runtime，观察图签名重叠率与差异算子。
4. `benchmark/train`：仅对候选策略做真实设备验证。

## Commands

在 `examples/transformer` 下：

```bash
# 1) 快速 meta 捕获
RECOMPUTE='{"6": 0}' python capture.py --mode static --static-profile fast

# 2) 高一致性 meta 捕获（失败时脚本内会自动回退）
RECOMPUTE='{"6": 0}' python capture.py --mode static --static-profile high

# 3) meta 与 runtime 对照
RECOMPUTE='{"6": 0}' python capture.py --mode static --static-profile fast --compare-runtime

# 4) 菜单入口
./run.sh capture
./run.sh capture-compare
```

## Scope Cleanup Rule

当前目录只保留主链脚本：

- `capture.py`
- `benchmark.py`
- `train.py`
- `main.py`

非主链诊断脚本统一放到 `legacy/`，按需再引入。
