# aten_recompute 关键设计要点（存档备忘）

> 该目录已被 `toolkit/` 完全取代并删除。以下记录其中值得参考的设计思路。

---

## 1. min-cut 最优重计算求解（core/min_cut.py）

将 AOT Autograd joint graph 的前向部分建模为流网络，用 NetworkX 最小割求解最优保存/重计算划分：

- **节点拆分模型**：每个 FX 节点 N → (N_in, N_out)，内部边容量 = `mem_bytes × 1.1^dist_from_bw`
- **距离加权**：离反向图越远的节点，保存代价越高（鼓励保留），越近的节点越容易被重算
- **source/sink 连接**：source → 禁止重计算节点（compute-intensive/random/primal），saved_values → sink
- **getitem siblings 一致性**：tuple-producing 父节点的 getitem 输出若出现 saved/recompute 混合状态，保守回退为全部保存，防止梯度正确性问题
- **memory_budget 过滤**：min-cut 结果按预算截断时，优先重算大张量以最大化节省

**参考文献**: Chen et al. 2016, Checkmate (MLSys 2020), PyTorch `min_cut_rematerialization_partition`

## 2. 自定义 partition_fn 注入选择性重计算（core/partition.py）

在 AOT Autograd 的 joint graph 切分阶段注入自定义逻辑：

- 7 种策略体系：no_recompute(0) → full(1) → keyword(2) → stride(3) → ratio(4) → op_type(5) → chain_depth(6) → min_cut(7)
- **`_find_required_primals()`**：移除 saved_value 后，BFS 追溯重计算链依赖的 primal placeholder，仅补充必要的，避免 saved_values 膨胀
- **`_cleanup_mark_layer()`**：清理自定义标记节点时检测悬空引用 + 去重

## 3. 自定义 ATen 算子实现层级标记（core/tag.py）

通过 `torch.library.custom_op` 注册 `my_compiler::mark_layer`，用 forward pre-hook 在不修改模型代码的前提下将 layer_rank 注入 FX 图：

- 正确注册 FakeTensor 推导规则（shape 不变，返回 empty_like）
- 正确注册 Autograd 规则（identity gradient，grad_output 直接透传）
- 使 partition_fn 能基于层级 rank 做策略决策

## 4. SOTA 方法对比数据（analysis/comparison.py）

结构化整理的重计算方法文献对比：

| 方法 | 粒度 | 最优性 | 决策时机 |
|------|------|--------|----------|
| PyTorch Checkpoint | 按层 | 手动 | 静态 |
| PyTorch SAC | 按算子 | 策略函数 | 编译时 |
| Checkmate | 按算子 | ILP 最优 | 离线 |
| Rotor | 按层 | DP 最优 | 离线 |
| DTR | 按张量 | 在线贪心 | 动态 |
| 本项目 Strategy 7 | 按算子 | min-cut 最优 | 编译时 |

正交方向：ActNN/GACT/COAT（压缩）、vDNN/POET/SSDTrain（卸载）、Capuchin/DELTA/Adacc（混合）、AdaPipe/Obscura（流水线感知）
