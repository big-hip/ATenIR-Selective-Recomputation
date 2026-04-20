# 03 — Min-Cut Rematerialization 效果分析报告

> **文档定位**: 项目核心实验报告，所有策略设计和后续实验的数据基础。
> 原始文件: `docs/research-mincut-analysis.md`
>
> 实验日期：2025-04-15
> 环境：PyTorch 2.6.0 + CUDA 12.4, conda env torch2.6-gpu
> 模型：LLaMA / GPT-2 (多种尺寸), batch=2

---

## 一、背景

`min_cut_rematerialization_partition` 是 PyTorch AOTAutograd 提供的分区策略，
使用 max-flow/min-cut 算法决定前向传播中哪些中间张量应保存给反向传播、
哪些应在反向时重新计算，目标是最小化前向-反向之间传递的张量总字节。

在本项目的策略比较页面中，min_cut 策略相比 eager baseline 并未显示出显著的内存节省。
本报告详细分析了原因并提出解决方案。

---

## 二、实验数据

### 2.1 compiled_default vs compiled_min_cut（公平比较，同为 make_boxed_func 后端）

| Model            | eager    | default(saved) | min_cut(saved) | mc vs default |
|------------------|----------|----------------|----------------|---------------|
| llama-2L-128H    | 53.7MB   | 72.0MB (80)    | 71.3MB (65)    | +1.0%         |
| llama-6L-512H    | 202.4MB  | 352.8MB (216)  | 334.2MB (173)  | +5.3%         |
| llama-12L-768H   | 701.6MB  | 1131.6MB (420) | 1054.0MB (335) | +6.9%         |
| llama-12L-1024H  | 874.1MB  | 1624.7MB (420) | 1511.9MB (335) | +6.9%         |
| gpt2-2L-128H     | 81.0MB   | 134.6MB (83)   | 133.5MB (77)   | +0.8%         |
| gpt2-6L-512H     | 385.5MB  | 607.2MB (219)  | 575.2MB (201)  | +5.3%         |
| gpt2-12L-768H    | 1095.4MB | 1793.8MB (423) | 1640.7MB (387) | +8.5%         |

**结论**: min_cut 相比 default partition 节省 1-8.5%，保存张量数减少 8-20%。
但所有 compiled 策略均大幅劣于 eager（34-86% 更多内存）。

### 2.2 activation_memory_budget 调参（llama-6L-512H）

| 配置                     | saved | act_delta | vs default partition |
|--------------------------|-------|-----------|----------------------|
| min_cut (budget=1.0)     | 173   | 332.8MB   | +5.3%                |
| min_cut (aggressive)     | 153   | 316.6MB   | +9.1%                |
| min_cut (budget=0.5)     | 153   | 315.0MB   | +9.6%                |
| min_cut (budget=0.3)     | 96    | 248.9MB   | +28.5%               |
| min_cut (budget=0.0)     | 84    | 239.1MB   | +31.4%               |
| min_cut (aggr+budget=0.3)| 89    | 244.2MB   | +29.9%               |

**结论**: budget=0 时相比 default partition 节省 31%，证明 min-cut 算法本身有效。

### 2.3 Inductor 后端 vs make_boxed_func 后端（llama-6L-512H）

| Strategy                          | act_delta | vs eager |
|-----------------------------------|-----------|----------|
| eager baseline                    | 203.2MB   | —        |
| compile(min_cut) make_boxed_func  | 332.4MB   | -63.6%   |
| compile(inductor)                 | 242.0MB   | -19.1%   |
| compile(inductor, budget=0)       | 239.9MB   | -18.1%   |
| sac_save_matmuls + compile(inductor) | 272.7MB | -34.2%  |
| ac + compile(inductor)            | 240.4MB   | -18.3%   |

**结论**: 即使使用 Inductor，在此模型规模下 compiled 仍比 eager 差约 19%。

---

## 三、根因分析

### 根因 1（核心）：make_boxed_func 不做算子融合

我们的 `compile_with_partition` 使用 `make_boxed_func(fw_gm.forward)` 作为编译器，
这是一个恒等变换——不进行任何算子融合。

min-cut 算法的设计前提是存在融合编译器（Inductor / NVFuser）：
- 被重算的 pointwise op 会被融合进 backward kernel → 重算零成本
- 不保存的张量不需要独立 CUDA 分配 → 内存节省

但 make_boxed_func 下：
- 每个中间张量独立分配 CUDA 内存
- AOTAutograd 将 forward 输出 ALL saved tensors 同时驻留在内存
- 无融合 → 重算仍需完整的内存读写

这就是为何 compiled_default (348MB) >> eager (203MB)：AOTAutograd 的 saved-tensor
机制比 eager autograd 的增量释放机制占用更多内存。

### 根因 2：默认 ban 列表过于保守

PyTorch 2.6 中 `min_cut_rematerialization_partition` 的默认配置：
- `ban_recompute_not_in_allowlist = True` → 仅允许 pointwise/view op 重算
- `ban_recompute_reductions = True` → 禁止重算 reduction
- `ban_recompute_materialized_backward = True` → 禁止重算反向中需物化的节点
- `activation_memory_budget = 1.0` → 优化运行时间而非内存

对 Transformer 而言，占内存最大的中间张量：
- mm/bmm/addmm 输出 → 被 ban（compute_intensive_ops 列表）
- softmax 输出 → 被 ban（不在 recomputable_ops 列表）
- _scaled_dot_product_flash_attention 输出 → 被 ban
- layer_norm 中间结果 → 被 ban（materialized_backward）

结果：min-cut 只能选择重算 view/reshape/clone/type_cast 等很小的张量。

### 根因 3：对小模型编译开销占比大

小模型（<100M参数）：
- activation memory 只占 peak 的小部分
- 编译开销（dynamo guard、tracing、extra tensors）占比大
- 节省的少量激活内存被编译开销抵消

---

## 四、关键代码路径

```
torch._functorch.partitioners.py:
  min_cut_rematerialization_partition()
    → choose_saved_values_set(joint_graph, node_info, memory_budget)
      → solve_min_cut(joint_graph, node_info, min_cut_options, dont_ban)
        → should_ban_recomputation(node)  # 核心：决定哪些 op 不允许重算
        → get_node_weight(node)           # 核心：节点内存权重
        → nx.minimum_cut()               # NetworkX max-flow/min-cut
```

关键配置项（torch._functorch.config）：
- `aggressive_recomputation`: 放松所有 ban → 允许更多重算
- `activation_memory_budget`: 0.0~1.0, 0=最大内存节省, 1=最大运行时优化
- `ban_recompute_not_in_allowlist`: 是否只允许白名单 op 重算
- `ban_recompute_reductions`: 是否禁止重算 reduction op
- `recompute_views`: 是否允许重算 view op
- `max_dist_from_bw`: 禁止重算离反向图太远的节点

---

## 五、Inductor 后端可行性验证（2025-04-15 补充）

### 5.1 关键澄清：Inductor 不会失去 partition_fn 控制

Inductor 内部直接调用 `min_cut_rematerialization_partition`
（见 `torch/_inductor/compile_fx.py:1780`），
而 `torch._functorch.config` 中的所有参数（activation_memory_budget,
aggressive_recomputation, ban_recompute_* 等）都是全局配置，
对 Inductor 同样生效。切换到 Inductor **不会失去任何参数控制能力**。

### 5.2 Inductor + budget + SAC 全组合实验

技术上所有组合均可正确运行，无报错。

#### 小模型 (llama-6L-512H, batch=2, seq=128)

| Strategy                    | act_delta | vs eager  |
|-----------------------------|-----------|-----------|
| eager_baseline              | 203.2MB   | —         |
| eager_classic_ac            | 204.8MB   | -0.8%     |
| eager_sac_save_matmuls      | 203.2MB   | +0.0%     |
| inductor_b1.0               | 241.2MB   | -18.7%    |
| inductor_b0.7               | 241.7MB   | -19.0%    |
| inductor_b0.0               | 241.5MB   | -18.8%    |
| sac_mm+inductor             | 273.4MB   | -34.6%    |
| ac+inductor                 | 240.4MB   | -18.3%    |
| sac_mm+inductor_b0          | 241.9MB   | -19.0%    |

#### 大模型 (llama-12L-1024H, batch=2, seq=256)

| Strategy                    | act_delta | vs eager  |
|-----------------------------|-----------|-----------|
| eager_baseline              | 868.3MB   | —         |
| **eager_classic_ac**        | **830.6MB** | **+4.3%** |
| eager_sac_save_matmuls      | 832.1MB   | +4.2%     |
| eager_sac_recompute_all     | 852.6MB   | +1.8%     |
| inductor_b1.0               | 920.8MB   | -6.0%     |
| inductor_b0.5               | 910.3MB   | -4.8%     |
| inductor_b0.0               | 926.0MB   | -6.6%     |
| ac+inductor                 | 933.3MB   | -7.5%     |
| sac_mm+inductor             | 988.9MB   | -13.9%    |

### 5.3 可行性结论

1. **技术可行性: ✅** — Inductor + budget + SAC + AC 任意组合均可运行
2. **budget 参数在 Inductor 下几乎无效** — Inductor + min-cut 默认已重算所有
   fusible ops，降低 budget 找不到更多可重算 op（剩下的都是 matmul/attention 等
   banned ops），各 budget 值内存差异 <1%
3. **SAC + Inductor 反而更差** — checkpoint 区域的重编译和边界管理开销 > 激活节省
4. **Inductor 固有开销 6-19%** — workspace buffers 和编译器内存管理模式导致
   所有 compiled 策略均比 eager 差
5. **对当前模型尺寸（≤100M参数），eager AC 是最有效策略** — 无编译开销，
   大模型上可节省 4.3%

**根本限制**: 当前可测试的模型尺寸下（受限于单卡显存），激活内存占 peak 比例
不够高，Inductor 的固有开销抵消了 min-cut/SAC 的收益。对于 7B+ 模型 + 大 batch
+ 长序列，预计 Inductor + min-cut 的收益会更显著。

---

## 六、峰值出现阶段分析（2025-04-15 补充）

### 6.1 分阶段峰值实验

测试模型: llama-12L-1024H, batch=2, seq=256
测量方法: 在 forward、backward、opt.step 三个阶段分别 reset_peak_memory_stats，
记录各阶段的独立峰值（相对 base）。

#### Adam 优化器

| Strategy              | fw_peak(MB) | bw_peak(MB) | opt_peak(MB) | TRUE_PEAK | 峰值阶段 |
|-----------------------|-------------|-------------|--------------|-----------|----------|
| eager_baseline        | 868.3       | 859.4       | 1666.4       | 1666.4    | OPT_STEP |
| eager_ac              | 240.9       | 841.1       | 1681.1       | 1681.1    | OPT_STEP |
| eager_sac_mm          | 601.6       | 831.1       | 1672.2       | 1672.2    | OPT_STEP |
| aot_eager (无融合)    | 748.8       | 917.1       | 1679.9       | 1679.9    | OPT_STEP |
| inductor (budget=1.0) | 611.4       | 919.2       | 1665.9       | 1665.9    | OPT_STEP |
| inductor (budget=0.0) | 80.7        | 918.1       | 1667.4       | 1667.4    | OPT_STEP |
| ac + inductor         | 107.5       | 932.8       | 1679.5       | 1679.5    | OPT_STEP |
| sac_mm + inductor     | 475.0       | 992.6       | 1680.0       | 1680.0    | OPT_STEP |

#### SGD 优化器（无优化器状态，消除 opt_peak 干扰）

| Strategy        | fw_peak(MB) | bw_peak(MB) | opt_peak(MB) | TRUE_PEAK | 峰值阶段 |
|-----------------|-------------|-------------|--------------|-----------|----------|
| eager_sgd       | 861.5       | 852.5       | 815.4        | 861.5     | FORWARD  |
| eager_ac_sgd    | 241.5       | 824.3       | 822.2        | 824.3     | BACKWARD |

### 6.2 关键结论

1. **Adam 下真正的峰值 100% 出现在 opt.step()，约 1666-1681MB，ALL 策略基本一致**
   - opt_peak ≈ grad_bytes (~830MB) + Adam step 临时张量 (~836MB ≈ param_bytes)
   - Adam 的 `denom = exp_avg_sq.sqrt() + eps` 为每个参数创建与参数同大小的临时张量
   - 所有激活策略（AC/SAC/min-cut）对 Adam 下的真实峰值 **无任何影响**

2. **SGD 下峰值在 FORWARD 或 BACKWARD，激活策略有效**
   - eager: 峰值在 FORWARD（861.5MB, 所有激活同时存活）
   - eager_ac: 峰值移至 BACKWARD（824.3MB, forward 仅 241.5MB），节省 4.3%

3. **max(fw_peak, bw_peak) 即"前反向峰值"才是衡量重计算策略效果的正确指标**

   重计算发生在 backward 阶段：AC/SAC 在 backward 中重算 forward 子图，
   min-cut 在 backward 中重算被标记的节点。因此重计算会**降低 fw_peak 但增加 bw_peak**。
   只看 fw_peak 会严重高估策略效果。

   | Strategy              | fw_peak | bw_peak | **fwbw_peak** | vs eager  |
   |-----------------------|---------|---------|---------------|-----------|
   | eager_baseline        | 868.3   | 859.4   | **868.3 (FW)** | —        |
   | eager_ac              | 240.9   | 841.1   | **841.1 (BW)** | -3.1%   |
   | eager_sac_mm          | 601.6   | 831.1   | **831.1 (BW)** | -4.3%   |
   | aot_eager (无融合)    | 748.8   | 917.1   | **917.1 (BW)** | +5.6%   |
   | inductor (budget=1.0) | 611.4   | 919.2   | **919.2 (BW)** | +5.9%   |
   | inductor (budget=0.0) | 80.7    | 918.1   | **918.1 (BW)** | +5.7%   |
   | ac + inductor         | 107.5   | 932.8   | **932.8 (BW)** | +7.4%   |
   | sac_mm + inductor     | 475.0   | 992.6   | **992.6 (BW)** | +14.3%  |

   关键观察：
   - **无 AC 的 eager 是唯一峰值在 FORWARD 的策略**（所有激活同时存活）
   - **所有使用重计算的策略，峰值都在 BACKWARD**（重算激活 + 梯度同时存在）
   - eager_ac: fw_peak 降 72%，但 bw_peak 仍有 841MB（梯度 ~839MB + 重算临时），
     前反向峰值仅降 **3.1%**
   - 所有 compiled 策略的 fwbw_peak **均劣于 eager**（5-14%）

4. **backward 峰值的组成分析**
   - bw_peak ≈ 梯度（~830MB, 与参数同大小）+ 重算时的临时激活 + backward 计算临时张量
   - 梯度是刚性开销，不受激活策略影响
   - 因此 bw_peak 的下限 ≈ grad_bytes（~830MB），激活策略只能优化其中的"重算临时"部分
   - 这就是为什么 AC 对 fwbw_peak 的实际收益很小（-3.1%）——梯度主导了 bw_peak

5. **之前实验（第二章）的测量方式是否正确？**
   - 第二章使用 `reset_peak → forward → backward → read_peak`，
     不包含 opt.step()，测的正是 max(fw_peak, bw_peak) = fwbw_peak ✓
   - 所以第二章的数据是正确的前反向峰值

### 6.3 重计算效果与 batch size 的关系

测试模型: llama-12L-1024H, seq=256, SGD 优化器（消除 opt_peak 干扰）

| batch | eager fwbw_peak | AC fwbw_peak | AC 节省 | eager 峰值位置 | AC 峰值位置 |
|-------|-----------------|--------------|---------|----------------|-------------|
| 1     | 819.0 (BW)      | 830.6 (BW)   | -1.4%   | BW (梯度主导)  | BW          |
| 2     | 859.9 (FW)      | 821.1 (BW)   | +4.5%   | FW (临界点)    | BW          |
| 4     | 1725.6 (FW)     | 828.5 (BW)   | +52.0%  | FW (激活主导)  | BW          |
| 8     | 3380.3 (FW)     | 963.9 (FW)   | +71.5%  | FW (激活主导)  | FW          |

关键规律：
- **fw_peak 与 batch 线性增长**: 399→860→1726→3380
- **grad_bytes 恒定**: ~810-834MB（= param_bytes，与 batch 无关）
- **AC 的 bw_peak 几乎恒定**: ~821-913MB（= grad_bytes + 单个 block 重算临时）
- **转折点**: 当 fw_peak ≈ grad_bytes 时（本模型 ≈ batch=2），AC 开始有效

结论：**重计算效果取决于 `fw_peak / grad_bytes` 比值**
- 比值 < 1（小 batch）: bw_peak 由梯度主导，重计算无效甚至有害
- 比值 ≈ 1（临界点）: 效果微弱
- 比值 >> 1（大 batch）: fw_peak 主导，重计算非常有效（52-71%）
- 实际训练场景（batch=16, 32, 64+）中此比值远 >> 1

### 6.4 指标选择建议

fwbw_peak = max(fw_peak, bw_peak) 是正确的总指标，但在小 batch 下会"隐藏"
重计算对激活的优化效果（因为 bw_peak 被梯度刚性成本主导）。

**论文中建议同时报告以下指标：**
- **fw_peak**: 前向峰值（重计算策略直接优化的目标）
- **bw_peak**: 反向峰值，标注其中 grad_bytes 的占比
- **fwbw_peak = max(fw_peak, bw_peak)**: 前反向峰值（总体评估）
- **true_peak = max(fw, bw, opt)**: 训练总峰值（标注峰值出现在哪个阶段）
- **activation_ratio = fw_peak / grad_bytes**: 解释重计算策略为什么有效/无效

**之前实验（batch=2）效果不显著的原因：**
不是策略本身无效，而是 batch=2 恰好在临界点上（fw_peak ≈ grad_bytes），
梯度的刚性成本掩盖了激活优化的收益。增大 batch 后效果显著。

---

## 七、修正后的策略分组设计

### 分组原则
- 每组有自己的 baseline，组内比较
- 跨组仅作"编译开销"等定性对比，不直接比绝对值

### Group 1: Eager（无编译）
- **Baseline**: eager（无 AC）
- classic_ac
- sac_save_matmuls
- sac_recompute_all

### Group 2: Compiled（图捕获 + 编译）
- **Baseline A**: aot_eager（图捕获 + default_partition，无融合）→ 展示"仅图捕获"的开销
- **Baseline B**: inductor (budget=1.0)（图捕获 + min-cut + Inductor 融合）→ 展示"编译优化后"的基准
- inductor (budget=0.5)
- inductor (budget=0.0)

两个 baseline 的对比意义：
- aot_eager vs inductor → 量化 Inductor 融合带来的收益
- aot_eager vs eager → 量化"图捕获+重放"本身的开销

注：aot_eager 仅作参照点。因其无融合，不在其上叠加 budget/SAC 变体。

### Group 3: SAC/AC + Inductor
- **Baseline**: inductor (budget=1.0)（同 Group 2 Baseline B）
- ac + inductor
- sac_mm + inductor
- sac_all + inductor
- sac_mm + inductor (budget=0.0)

### 测量指标
- **fw_peak**: 前向峰值（重计算直接优化的目标）
- **bw_peak**: 反向峰值（含梯度+重算开销），标注 grad_bytes 占比
- **opt_peak**: 优化器步骤峰值
- **fwbw_peak = max(fw, bw)**: 前反向峰值
- **true_peak = max(fw, bw, opt)**: 训练总峰值
- **峰值阶段**: 标注 true_peak 出现在哪个阶段
- **step_ms**: 单步训练耗时
- **compile_s**: 编译耗时

---

## 八、外部审查 4 点验证（2025-04-15）

以下 4 点由外部审查提出，逐一实验验证。
测试模型: llama-12L-1024H, batch=4, seq=256

### 8.1 P1: 显存碎片化 — ❌ 未确认

**假说**: Inductor/AOTAutograd 的执行序列导致显存碎片，`reserved` 远大于 `allocated`，
虚高了 peak 测量值。

**验证方法**: 对比 `max_memory_allocated`（实际张量占用）与 `max_memory_reserved`（缓存分配器水位）。

**结果**: warmup 后分配器已预留充足内存，measurement 阶段 `reserved` 增量 ≤ `allocated` 增量。
CUDA 缓存分配器高效复用预留块，**碎片化不是导致 compiled 策略峰值偏高的原因**。
`max_memory_allocated` 是准确的策略比较指标。

### 8.2 P2: FlashAttention 黑盒效应 — ❌ 未确认

**假说**: FlashAttention 内部已做重算优化，AC/Min-Cut 是在已优化结构上做"二次重算"，
所以收益小。禁用 FlashAttention 后 Min-Cut 优势应显著提升。

**验证**:

1. 首先确认模型实际使用的注意力实现：
   ```
   Attention class: LlamaAttention (手动 torch.matmul, 非 SDPA)
   Uses SDPA: False
   Uses manual matmul: True
   ```
   **HuggingFace transformers 4.35.2 的 LLaMA 默认使用手动注意力（非 FlashAttention）。**
   `torch.nn.attention.sdpa_kernel` 上下文管理器无效。

2. 实验对比（SGD, batch=4）：

   | SDPA 后端 | eager fw_peak | AC fwbw_peak | AC 节省 |
   |-----------|---------------|--------------|---------|
   | auto      | 1709.8MB      | 830.6MB      | 51.4%   |
   | math-only | 1713.0MB      | 826.4MB      | 51.8%   |

   差异仅 0.2%，因为模型根本没用 SDPA。

**结论**: FlashAttention 假说不成立。模型使用手动注意力，全量注意力权重矩阵
已作为普通激活保存。AC/Min-Cut 可以且已经在优化这些激活。
**注**: 若使用 `LlamaSdpaAttention` 或更新版 transformers（支持 SDPA），
此结论需重新验证。

### 8.3 P3: 编译耗时与运行时开销 — ✅ 确认

**假说**: Min-Cut 的 NetworkX max-flow 算法和 Inductor 编译带来显著时间开销，
重算也增加步骤时间。

**验证数据**（SGD, batch=4）：

| Strategy              | step_ms | compile_s | vs eager step |
|-----------------------|---------|-----------|---------------|
| eager                 | 82.7    | 0.4       | —             |
| eager_ac              | 102.5   | 0.2       | +24.0%        |
| inductor (budget=1.0) | 70.5    | **19.4**  | -14.8%        |
| inductor (budget=0.0) | 95.2    | **22.4**  | +15.1%        |
| ac + inductor         | 87.3    | **20.7**  | +5.6%         |

**结论**:
- **Inductor 编译耗时 19-22 秒**，是 eager 的 50-100 倍。对短训练任务不划算。
- **Inductor 融合使 steady-state step 快 15%**（70.5 vs 82.7ms）
- **budget=0.0 重算使 step 慢 35%**（95.2 vs 70.5ms）——重算有明确的时间成本
- **AC 增加 24% step 时间**（重算代价）
- 实际 ROI 取决于训练步数：compile_overhead / step_saving ≈ 19s / 12ms ≈ 1583 步回本

### 8.4 P4: Fused Optimizer — ✅✅ 关键确认

**假说**: `Adam(fused=True)` 减少 opt.step() 临时 buffer，使 opt_peak 降低，
此时激活策略节省的空间才真正有意义。

**验证数据**（batch=4）：

| Strategy                  | fwbw_peak | opt_peak  | true_peak | AC 节省 (true) |
|---------------------------|-----------|-----------|-----------|----------------|
| eager + Adam              | 1722.4    | **1647.5** | 1722.4   | —              |
| eager_ac + Adam           | 845.3     | **1673.7** | **1673.7 (OPT!)** | 2.8%  |
| eager + Adam(fused=True)  | 1722.4    | **822.2**  | 1722.4   | —              |
| eager_ac + Adam(fused=True)| 827.4    | **823.2**  | **827.4 (BW)** | **52.0%** |
| eager + SGD               | 1710.9    | 820.1     | 1710.9   | —              |
| eager_ac + SGD            | 827.4     | 823.2     | 827.4    | **51.6%**      |

**关键发现**:

1. **`Adam(fused=True)` 使 opt_peak 从 1647→822MB（降 50%）**
   - 常规 Adam 的 `denom = exp_avg_sq.sqrt() + eps` 为每个参数创建临时张量
   - Fused Adam 在单个 CUDA kernel 内完成，不创建中间张量

2. **这彻底改变了 AC 的实际效果**：
   - 常规 Adam: AC true_peak 被 opt_peak(1674) 封顶 → 节省仅 **2.8%**
   - Fused Adam: opt_peak(823) < fwbw_peak(827) → AC true_peak = fwbw_peak → 节省 **52.0%**
   - 与 SGD 效果一致（51.6%）——fused Adam 消除了优化器对峰值的掩盖

3. **实际训练推荐**: 使用 `Adam(fused=True)` 是标准实践，
   此时激活策略的收益可完整体现在 true_peak 上。

### 8.5 四点总结

| 审查点 | 假说 | 结论 | 对报告影响 |
|--------|------|------|-----------|
| P1: 碎片化 | 碎片虚高了编译策略峰值 | ❌ 未确认 | 无需修改 |
| P2: FlashAttention | 黑盒掩盖了 AC 效果 | ❌ 未确认（模型用手动 attn） | 补充说明 |
| P3: 编译耗时 | 应量化时间成本 | ✅ 确认 | **需加入指标** |
| P4: Fused Optimizer | fused Adam 解锁 AC 效果 | ✅✅ 关键确认 | **需修改核心结论** |

**P4 修正了第六章的核心结论：**
之前说"Adam 下所有策略的 true_peak 基本一致，AC 无效"——
这只在 `fused=False` 时成立。使用 `fused=True`（生产标配）后，
opt_peak 大幅降低，AC 的 52% 节省可完整体现在 true_peak 上。

---

## 九、参考资料

1. Chillee (Horace He), "Min-cut optimal(*) recomputation with AOTAutograd",
   PyTorch Dev Discuss, 2022.
   https://dev-discuss.pytorch.org/t/min-cut-optimal-recomputation-i-e-activation-checkpointing-with-aotautograd/467

2. PyTorch Blog, "Current and New Activation Checkpointing Techniques in PyTorch", 2024.
   https://pytorch.org/blog/activation-checkpointing-techniques/

3. PyTorch Source: torch/_functorch/partitioners.py (v2.6.0)
   - min_cut_rematerialization_partition (L1703-1912)
   - choose_saved_values_set (L1466-1705)
   - solve_min_cut (L815-1053)
   - get_default_op_list (L1220-1382)

4. PyTorch Source: torch/_functorch/config.py
   - activation_memory_budget, aggressive_recomputation, ban_recompute_* 等配置项
