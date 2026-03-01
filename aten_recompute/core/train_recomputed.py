"""
train_recomputed.py
将重计算优化后的 FW/BW GraphModule 包装为可训练的 autograd.Function 并执行训练。

核心难点：AOT Autograd 在 dynamic=True 时会从模型输入张量的动态维度中提取 SymInt
作为独立的整数 primal，且排在参数之前，因此需要在运行时自动推断调用约定。

公开接口
--------
make_recomputed_fn(fw_gm, bw_gm, params_flat)
    将 FW/BW 图包装为可调用对象，返回 call(*model_tensor_inputs) 函数。

run_training(fw_gm, bw_gm, model, step_fn, ...)
    通用训练循环，接受一个无参的 step_fn() 可调用对象执行每步训练。
    也提供带领域参数的 Transformer 便捷包装（向后兼容）。
"""

import torch
from typing import Callable, Optional


# ──────────────────────────────────────────────────────────────────────────────
# 工具：SymInt 相等性判断
# ──────────────────────────────────────────────────────────────────────────────

def _match_symint(sym_val, dim_val) -> bool:
    """
    判断 FW 图中某个 SymInt placeholder 的 meta 值是否与张量某维度的符号值相同。

    优先顺序：
      1. 对象同一性 (is)          ← 最可靠，同一 tracing 上下文对象相同
      2. SymInt.node 同一性       ← 同一符号变量的不同包装对象
      3. str() 字符串匹配         ← 兜底（对 s0、s1 等简单变量仍可靠）
    """
    if dim_val is sym_val:
        return True
    sym_node = getattr(sym_val, 'node', None)
    dim_node = getattr(dim_val, 'node', None)
    if sym_node is not None and sym_node is dim_node:
        return True
    try:
        return str(dim_val) == str(sym_val)
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────────────
# 核心：从 FW/BW GraphModule 推断完整的调用约定
# ──────────────────────────────────────────────────────────────────────────────

def _detect_fw_calling_convention(fw_gm, bw_gm, n_params_flat: int) -> dict:
    """
    检查 FW/BW GraphModule，返回运行时组装参数所需的全部信息：

    返回字典键说明：
      n_front              前置非 param primals 数量 (SymInt + 模型张量输入)
      front_is_symint      长度 n_front 的 bool 列表，True=SymInt，False=张量
      symint_recipes       长度 n_front 的列表，SymInt 位置为 (tensor_idx, dim)，
                           张量位置为 None
      n_model_tensor_inputs 模型张量输入的数量（不含 SymInt）
      model_out_fw_indices  FW 返回元组中模型输出所在的位置（不被 BW 消费的项）
      bw_saved_fw_indices   BW 各 saved 输入在 FW 返回元组中的对应位置
      bw_saved_is_tensor    与 bw_saved_fw_indices 等长，True=张量，False=标量
    """
    fw_ph_nodes = [n for n in fw_gm.graph.nodes if n.op == 'placeholder']
    n_primals   = len(fw_ph_nodes)
    n_front     = n_primals - n_params_flat

    # ── 1. 判断前置 placeholder 类型 ──────────────────────────────────────────
    front_is_symint = [
        not isinstance(ph.meta.get('val'), torch.Tensor)
        for ph in fw_ph_nodes[:n_front]
    ]
    n_model_tensor_inputs = sum(1 for x in front_is_symint if not x)

    # ── 2. 为每个 SymInt 找到对应的输入张量维度 ────────────────────────────────
    tensor_meta_list = [
        fw_ph_nodes[i].meta.get('val')
        for i, is_s in enumerate(front_is_symint)
        if not is_s
    ]
    symint_recipes = []
    for i, is_s in enumerate(front_is_symint):
        if not is_s:
            symint_recipes.append(None)
            continue
        sym_val = fw_ph_nodes[i].meta.get('val')
        recipe  = None
        if sym_val is not None:
            for t_idx, t_meta in enumerate(tensor_meta_list):
                if t_meta is None:
                    continue
                for dim, d in enumerate(t_meta.shape):
                    if _match_symint(sym_val, d):
                        recipe = (t_idx, dim)
                        break
                if recipe:
                    break
        symint_recipes.append(recipe)

    # ── 3. 构建 FW 输出 → BW 输入的索引映射 ───────────────────────────────────
    fw_output_node = next(n for n in fw_gm.graph.nodes if n.op == 'output')
    fw_out_nodes   = fw_output_node.args[0]
    fw_name_to_idx = {n.name: i for i, n in enumerate(fw_out_nodes)}

    bw_ph_nodes  = [n for n in bw_gm.graph.nodes if n.op == 'placeholder']
    saved_bw_phs = [ph for ph in bw_ph_nodes if not ph.name.startswith('tangents_')]

    bw_saved_fw_indices = [fw_name_to_idx[ph.name] for ph in saved_bw_phs]
    bw_saved_set        = set(bw_saved_fw_indices)

    # 模型输出 = FW 返回中唯一不被 BW 消费的那项
    model_out_fw_indices = [
        i for i in range(len(fw_out_nodes)) if i not in bw_saved_set
    ]

    # ── 4. BW saved 各项是张量还是标量 ────────────────────────────────────────
    bw_saved_is_tensor = [
        isinstance(fw_out_nodes[fw_idx].meta.get('val'), torch.Tensor)
        for fw_idx in bw_saved_fw_indices
    ]

    return dict(
        n_front=n_front,
        front_is_symint=front_is_symint,
        symint_recipes=symint_recipes,
        n_model_tensor_inputs=n_model_tensor_inputs,
        model_out_fw_indices=model_out_fw_indices,
        bw_saved_fw_indices=bw_saved_fw_indices,
        bw_saved_is_tensor=bw_saved_is_tensor,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 核心：将 FW/BW 图包装为 autograd.Function
# ──────────────────────────────────────────────────────────────────────────────

def make_recomputed_fn(fw_gm, bw_gm, params_flat):
    """
    将优化后的 FW/BW GraphModule 包装为可训练的 autograd.Function，
    返回 call(*model_tensor_inputs) 函数。

    调用约定（由图检查自动推断，无需手工指定）：
      FW 输入:  (SymInt?, tensor, SymInt?, tensor, ..., *params_flat)
      FW 输出:  (..., model_output 位于 model_out_fw_indices[0], ...)
      BW 输入:  (*saved_按_BW_图顺序, *tangents)   tangent 在最后
      BW 输出:  (*grad_all_primals)                顺序与 FW primals 相同
    """
    n_params_flat = len(params_flat)
    conv = _detect_fw_calling_convention(fw_gm, bw_gm, n_params_flat)

    front_is_symint       = conv['front_is_symint']
    symint_recipes        = conv['symint_recipes']
    n_model_tensor_inputs = conv['n_model_tensor_inputs']
    model_out_fw_indices  = conv['model_out_fw_indices']
    bw_saved_fw_indices   = conv['bw_saved_fw_indices']
    bw_saved_is_tensor    = conv['bw_saved_is_tensor']
    n_front               = conv['n_front']
    n_primals             = n_front + n_params_flat

    class _RecomputedStep(torch.autograd.Function):

        @staticmethod
        def forward(ctx, *tensor_args):
            # tensor_args = (*model_tensor_inputs, *params_flat)  仅含张量
            model_inputs = tensor_args[:n_model_tensor_inputs]
            params       = tensor_args[n_model_tensor_inputs:]

            # 从输入张量的 shape 中提取 SymInt 的具体整数值
            symint_values = [
                int(model_inputs[t_idx].shape[dim])
                for is_s, recipe in zip(front_is_symint, symint_recipes)
                if is_s and recipe is not None
                for t_idx, dim in [recipe]
            ]
            # 按 FW 图顺序交叉排列 SymInt 与张量，最后追加 params
            symint_iter = iter(symint_values)
            tensor_iter = iter(model_inputs)
            primals = [
                next(symint_iter) if is_s else next(tensor_iter)
                for is_s in front_is_symint
            ]
            primals.extend(params)

            fw_raw = fw_gm(*primals)

            # 提取模型输出：
            #   单输出 → 直接返回张量（向后兼容）
            #   多输出 → 返回元组，调用方按需解包
            if len(model_out_fw_indices) == 1:
                model_out = fw_raw[model_out_fw_indices[0]]
            else:
                model_out = tuple(fw_raw[i] for i in model_out_fw_indices)

            # 张量型 saved 存入 ctx.save_for_backward；标量型（SymInt 等）单独记录
            saved_tensors = [
                fw_raw[fw_idx]
                for fw_idx, is_t in zip(bw_saved_fw_indices, bw_saved_is_tensor)
                if is_t
            ]
            saved_scalars = [
                (bw_pos, fw_raw[fw_idx])
                for bw_pos, (fw_idx, is_t)
                in enumerate(zip(bw_saved_fw_indices, bw_saved_is_tensor))
                if not is_t
            ]
            ctx.save_for_backward(*saved_tensors)
            ctx.saved_scalars    = saved_scalars
            ctx.bw_n_saved       = len(bw_saved_fw_indices)
            ctx.bw_saved_is_tensor = bw_saved_is_tensor
            return model_out

        @staticmethod
        def backward(ctx, *tangents):
            # 按 BW 图期望的顺序重建 saved 参数列表
            bw_args = [None] * ctx.bw_n_saved
            t_iter  = iter(ctx.saved_tensors)
            for bw_pos, is_t in enumerate(ctx.bw_saved_is_tensor):
                if is_t:
                    bw_args[bw_pos] = next(t_iter)
            for bw_pos, val in ctx.saved_scalars:
                bw_args[bw_pos] = val

            # BW 图约定：(*saved, *tangents)，tangent 在最后
            bw_raw = bw_gm(*bw_args, *tangents)

            # BW 输出为所有 FW primals 的梯度（顺序与 FW primals 相同）：
            #   位置 0..n_front-1  : SymInt 及模型张量输入的梯度（通常为 None）
            #   位置 n_front..n_primals-1: param 梯度
            # 返回顺序需与 apply() 参数顺序一致：(*model_tensor_inputs, *params_flat)
            model_input_grads = tuple(
                bw_raw[i] for i, is_s in enumerate(front_is_symint) if not is_s
            )
            param_grads = tuple(bw_raw[n_front:n_primals])
            return model_input_grads + param_grads

    def call(*model_tensor_inputs):
        return _RecomputedStep.apply(*model_tensor_inputs, *params_flat)

    return call


# ──────────────────────────────────────────────────────────────────────────────
# 工具：按 FW 图 primal 顺序收集模型参数与 buffer
# ──────────────────────────────────────────────────────────────────────────────

def _collect_params_in_fw_order(model):
    """
    按 AOT Autograd 捕获 FW 图时的 primal 顺序收集模型参数与 buffer。

    原理：Dynamo 将模型的 get_attr 节点按前向遍历顺序提升为图输入 placeholder。
    因此 FW 图中参数/buffer primal 的顺序 = 深度优先遍历 named_modules() 时，
    对每个模块先收集其直属 _parameters，再收集其直属 _buffers 的顺序。
    同时通过 data_ptr() 去重以处理权重绑定（如 fc.weight = decoder_embedding.weight）。
    """
    result = []
    seen = set()
    for _, module in model.named_modules():
        for p in module._parameters.values():
            if p is not None:
                ptr = p.data_ptr()
                if ptr not in seen:
                    seen.add(ptr)
                    result.append(p)
        for b in module._buffers.values():
            if b is not None:
                ptr = b.data_ptr()
                if ptr not in seen:
                    seen.add(ptr)
                    result.append(b)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 公开接口：训练入口
# ──────────────────────────────────────────────────────────────────────────────

def run_training(
    fw_gm,
    bw_gm,
    model,
    step_fn: Optional[Callable[[], None]] = None,
    # ── 以下为向后兼容的 Transformer 特定参数（step_fn 优先）──────────────
    src_data=None,
    tgt_data=None,
    tgt_vocab_size: int = None,
    criterion=None,
    graph_capture=None,
    # ──────────────────────────────────────────────────────────────────────
    lr: float = 0.0001,
    n_steps: int = 10,
):
    """
    使用重计算优化后的 FW/BW 图进行训练验证。

    推荐用法（通用，解耦领域逻辑）::

        recomputed_forward = make_recomputed_fn(fw_gm, bw_gm, params_flat)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        def my_step():
            optimizer.zero_grad()
            loss = my_loss_fn(recomputed_forward(*my_inputs))
            loss.backward()
            optimizer.step()

        run_training(fw_gm, bw_gm, model, step_fn=my_step, n_steps=10)

    向后兼容用法（Transformer 特定）::

        run_training(fw_gm, bw_gm, model,
                     src_data=src_data, tgt_data=tgt_data,
                     tgt_vocab_size=tgt_vocab_size, criterion=criterion,
                     graph_capture=graph_capture, n_steps=10)

    Parameters
    ----------
    fw_gm / bw_gm : 重计算优化后的 GraphModule
    model         : 原始 nn.Module（用于获取 params 和构造优化器）
    step_fn       : 若提供，每步直接调用 step_fn()（无参数），忽略下方所有领域参数。
    src_data / tgt_data / tgt_vocab_size / criterion / graph_capture :
                  Transformer 特定的向后兼容参数，step_fn=None 时使用。
    lr            : Adam 学习率（仅在向后兼容路径下使用）
    n_steps       : 训练步数
    """
    print(f"\n[训练开始] 使用重计算优化图进行训练（{n_steps} 步）...")

    if step_fn is not None:
        # ── 通用路径：由调用方提供完整的训练步骤 ─────────────────────────────
        for step in range(n_steps):
            step_fn()
            if (step + 1) % max(1, n_steps // 5) == 0 or step == 0:
                print(f"  Step {step + 1:2d}/{n_steps}")
    else:
        # ── 向后兼容路径：Transformer seq2seq 特定逻辑 ──────────────────────
        assert src_data is not None and tgt_data is not None, (
            "step_fn=None 时必须提供 src_data / tgt_data。"
        )
        n_primals   = sum(1 for n in fw_gm.graph.nodes if n.op == 'placeholder')
        fw_ph_names = [n.name for n in fw_gm.graph.nodes if n.op == 'placeholder']
        print(f"  FW 图 placeholder 数量: {n_primals}，名字: {fw_ph_names}")

        if graph_capture is not None and getattr(graph_capture, 'fw_params_flat', None):
            params_flat = graph_capture.fw_params_flat
            print(f"  使用 sample_inputs data_ptr 匹配重建 params_flat（{len(params_flat)} 个）。")
        else:
            params_flat = _collect_params_in_fw_order(model)
            print(f"  警告：退回到模块树遍历顺序（{len(params_flat)} 个）。")

        recomputed_forward = make_recomputed_fn(fw_gm, bw_gm, params_flat)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

        for step in range(n_steps):
            optimizer.zero_grad()
            output = recomputed_forward(src_data, tgt_data[:, :-1])
            loss = criterion(
                output.contiguous().view(-1, tgt_vocab_size),
                tgt_data[:, 1:].contiguous().view(-1),
            )
            loss.backward()
            optimizer.step()
            print(f"  Step {step + 1:2d} | Loss: {loss.item():.4f}")

    print("训练完成。")
