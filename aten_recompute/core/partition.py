"""
partition.py

自定义 partition_fn，在 AOT Autograd 的 joint graph 切分阶段嵌入
选择性激活重计算逻辑。通过控制 saved_values 列表决定哪些中间激活
保存到 FW 输出、哪些在 BW 中自动重计算。

用法::

    from aten_recompute.core.partition import make_selective_partition_fn

    partition_fn = make_selective_partition_fn(strategy_config={"6": 0})
    aot_module_simplified(gm, inputs,
        fw_compiler=..., bw_compiler=...,
        partition_fn=partition_fn)
"""
import functools
import json
import operator
import os
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

import torch.fx as fx
from torch.fx.experimental.proxy_tensor import is_sym_node

from torch._functorch.partitioners import (
    _extract_fwd_bwd_modules,
    _extract_fwd_bwd_outputs,
    _extract_graph_with_inputs_outputs,
    _is_primal,
    _is_tangent,
    _is_fwd_seed_offset,
    _is_bwd_seed_offset,
    _is_backward_state,
    reordering_to_mimic_autograd_engine,
)
from torch._inductor.compile_fx import _recursive_joint_graph_passes

from ..utils.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  策略函数
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_strategy(strategy_config: Optional[dict] = None) -> Tuple[str, Any]:
    """
    解析策略配置。优先使用 strategy_config 参数，否则从 RECOMPUTE 环境变量读取。

    Returns
    -------
    (option, param) : 策略编号（str）和策略参数
    """
    if strategy_config is not None:
        cfg = strategy_config
    else:
        env_str = os.getenv("RECOMPUTE", "{}")
        try:
            cfg = json.loads(env_str)
            if not isinstance(cfg, dict):
                raise ValueError("RECOMPUTE 值必须是 JSON object")
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("[partition] RECOMPUTE 环境变量无效，回退到不重计算。原因: %s", e)
            return "0", None

    if not cfg:
        return "0", None
    option, param = next(iter(cfg.items()))
    return str(option), param


def _chain_depth_joint(
    node: fx.Node,
    saved_value_names: Set[str],
    primal_names: Set[str],
    forward_node_names: Set[str],
) -> int:
    """
    在 joint graph 上估算重计算链深度。

    从 node 向上 BFS，遇到其他 saved_value 或 primal placeholder 时停止，
    统计途经的额外中间节点数（不含 node 本身）。

    depth=0 表示 node 的所有输入均来自 saved_values 或 primals，
    是单算子重计算，代价最低。
    """
    depth = 0
    queue: deque = deque([node])
    visited: Set[str] = {node.name}
    stop_names = (saved_value_names - {node.name}) | primal_names
    while queue:
        n = queue.popleft()
        for inp in n.all_input_nodes:
            if inp.name in visited:
                continue
            visited.add(inp.name)
            if inp.name in stop_names or inp.name not in forward_node_names:
                continue
            depth += 1
            queue.append(inp)
    return depth


def _find_required_primals(
    removed_nodes: List[fx.Node],
    remaining_saved_names: Set[str],
    primal_names: Set[str],
    forward_node_names: Set[str],
) -> Set[str]:
    """
    从被移除的 saved_value 节点出发 BFS 向上追溯，
    找出重计算链**实际依赖**的 primal 集合。

    只有这些 primal 需要被补充到 saved_values 中，而非所有缺失 primal。

    停止条件：
    - 遇到 primal → 记录并停止（找到了重计算链的数据源）
    - 遇到剩余 saved_values 中的节点 → 停止（该分支已有保存值）
    - 遇到非前向节点 → 跳过
    """
    required: Set[str] = set()
    visited: Set[str] = set()
    queue: deque = deque()

    for node in removed_nodes:
        if node.name not in visited:
            visited.add(node.name)
            queue.append(node)

    while queue:
        n = queue.popleft()
        for inp in n.all_input_nodes:
            if inp.name in visited:
                continue
            visited.add(inp.name)

            if inp.name in primal_names:
                required.add(inp.name)
                continue

            if inp.name in remaining_saved_names:
                continue

            if inp.name not in forward_node_names:
                continue

            queue.append(inp)

    return required


def _apply_strategy(
    option: str,
    param: Any,
    saved_values: List[fx.Node],
    sorted_ranks: List[int],
    node_name_to_rank: Dict[str, int],
    forward_node_names: Set[str],
    primal_names: Set[str],
    joint_module: Optional[fx.GraphModule] = None,
) -> List[fx.Node]:
    """
    根据策略编号返回应被重计算的节点列表（从 saved_values 中移除的部分）。
    """
    if option == "0":
        return []

    if option == "1":
        # 策略 1：重计算全部非 placeholder saved_values
        result = [sv for sv in saved_values if sv.op != "placeholder"]
        logger.info("[partition] 策略 1：重计算全部 %d 个激活。", len(result))
        return result

    if option == "2":
        # 策略 2：按名称关键字
        keywords = param if isinstance(param, list) else [param]
        result = [sv for sv in saved_values if any(k in sv.name for k in keywords)]
        logger.info("[partition] 策略 2：按关键字 %s 匹配到 %d 个节点。", keywords, len(result))
        return result

    if option == "3":
        # 策略 3：按步幅选层
        if not param or len(param) != 2:
            logger.warning("[partition] 策略 3 配置格式错误，期望 [start, stride]，收到: %s", param)
            return []
        start, stride = int(param[0]), int(param[1])
        target_ranks = {
            rank for idx, rank in enumerate(sorted_ranks)
            if idx >= start and (idx - start) % stride == 0
        }
        result = [
            sv for sv in saved_values
            if node_name_to_rank.get(sv.name, -1) in target_ranks and sv.op != "placeholder"
        ]
        logger.info("[partition] 策略 3：start=%d, stride=%d，匹配 %d 个节点。", start, stride, len(result))
        return result

    if option == "4":
        # 策略 4：按比例选前 N% 层
        ratio = float(param) if param is not None else 0.5
        num_layers = int(len(sorted_ranks) * ratio)
        target_ranks = set(sorted_ranks[:num_layers])
        result = [
            sv for sv in saved_values
            if node_name_to_rank.get(sv.name, -1) in target_ranks and sv.op != "placeholder"
        ]
        logger.info("[partition] 策略 4：重计算前 %d/%d 层，匹配 %d 个节点。",
                    num_layers, len(sorted_ranks), len(result))
        return result

    if option == "5":
        # 策略 5：按算子类型
        op_types = param if isinstance(param, list) else [param]
        result = [
            sv for sv in saved_values
            if sv.op == "call_function"
            and hasattr(sv.target, "__name__")
            and sv.target.__name__ in op_types
        ]
        logger.info("[partition] 策略 5：按算子类型 %s 匹配 %d 个节点。", op_types, len(result))
        return result

    if option == "6":
        # 策略 6：自动廉价重计算
        max_depth = 0 if param is None else int(param)
        saved_value_names = {sv.name for sv in saved_values}
        result = [
            sv for sv in saved_values
            if sv.op != "placeholder"
            and _chain_depth_joint(sv, saved_value_names, primal_names, forward_node_names) <= max_depth
        ]
        logger.info("[partition] 策略 6：链深度 ≤ %d，匹配 %d 个节点。", max_depth, len(result))
        return result

    if option == "7":
        # 策略 7：min-cut 最优重计算
        if joint_module is None:
            logger.error("[partition] 策略 7 需要 joint_module 参数。")
            return []
        from .min_cut import solve_min_cut_recompute
        budget = 1.0 if param is None else float(param)
        result = solve_min_cut_recompute(
            joint_module, forward_node_names, saved_values,
            primal_names, node_name_to_rank, memory_budget=budget,
        )
        logger.info("[partition] 策略 7：min-cut 优化，memory_budget=%.2f，选择重计算 %d 个节点。",
                    budget, len(result))
        return result

    logger.warning("[partition] 未知策略 '%s'，不执行重计算。", option)
    return []


# ═══════════════════════════════════════════════════════════════════════════════
#  mark_layer 清理
# ═══════════════════════════════════════════════════════════════════════════════

def _cleanup_mark_layer(
    joint_module: fx.GraphModule,
    saved_values: List[fx.Node],
    saved_sym_nodes: List[fx.Node],
) -> None:
    """
    从 joint graph 中移除所有 mark_layer 节点和 layer_Rank kwargs。
    同时同步更新 saved_values / saved_sym_nodes 中的引用。

    必须在 _extract_fwd_bwd_modules 之前调用。
    """
    graph = joint_module.graph

    for node in list(graph.nodes):
        if node.op == "call_function" and "mark_layer" in str(node.target):
            replacement = node.args[0]
            # 更新 saved_values 中的引用
            for i, sv in enumerate(saved_values):
                if sv is node:
                    saved_values[i] = replacement
            for i, sn in enumerate(saved_sym_nodes):
                if sn is node:
                    saved_sym_nodes[i] = replacement
            node.replace_all_uses_with(replacement)
            graph.erase_node(node)

    for node in graph.nodes:
        if "layer_Rank" in node.kwargs:
            node.kwargs = {k: v for k, v in node.kwargs.items() if k != "layer_Rank"}

    graph.lint()
    joint_module.recompile()


# ═══════════════════════════════════════════════════════════════════════════════
#  层级传播
# ═══════════════════════════════════════════════════════════════════════════════

def _propagate_layer_ranks(
    joint_module: fx.GraphModule,
    forward_node_names: Set[str],
) -> Tuple[List[int], Dict[str, int]]:
    """
    在 joint graph 的前向部分传播 mark_layer 层级信息。

    Returns
    -------
    sorted_ranks : 出现在 saved_values 上的 layer_rank 列表（去重排序）
    node_name_to_rank : 节点名 → rank 映射
    """
    node_to_rank: Dict[fx.Node, int] = {}

    for node in joint_module.graph.nodes:
        if node.name not in forward_node_names:
            continue

        # mark_layer 节点：rank 来自 args[1]
        if (node.op == "call_function"
                and "mark_layer" in str(node.target)):
            rank = node.args[1]
            node_to_rank[node] = rank
            continue

        # 普通节点：继承所有父节点中最大的 rank
        parent_ranks = [
            node_to_rank[inp]
            for inp in node.all_input_nodes
            if inp in node_to_rank
        ]
        if parent_ranks:
            node_to_rank[node] = max(parent_ranks)

    # 收集所有不同 rank（后面策略要用）
    all_ranks = set(node_to_rank.values())
    sorted_ranks = sorted(all_ranks)
    node_name_to_rank = {n.name: r for n, r in node_to_rank.items()}

    logger.info("[partition] 层级传播完成，发现 ranks: %s", sorted_ranks)
    return sorted_ranks, node_name_to_rank


# ═══════════════════════════════════════════════════════════════════════════════
#  核心 partition_fn
# ═══════════════════════════════════════════════════════════════════════════════

def selective_recompute_partition(
    joint_module: fx.GraphModule,
    _joint_inputs,
    *,
    num_fwd_outputs: int,
    strategy_config: Optional[dict] = None,
) -> Tuple[fx.GraphModule, fx.GraphModule]:
    """
    自定义 partition_fn：在 joint graph 切分阶段嵌入选择性激活重计算。

    工作流：
    0. Inductor joint graph passes（replace_random 等预处理）
    A. 节点分类 → 收集 baseline saved_values
    B. 层级传播 → 推导 mark_layer rank 信息
    C. 策略筛选 → 决定哪些 saved_values 改为重计算
    D. 清理 mark_layer
    E. 调用 _extract_fwd_bwd_modules 切分图
    F. 返回 (fw_module, bw_module)
    """
    # ── 阶段 0：Inductor 预处理（replace_random、constant folding 等）──────────
    _recursive_joint_graph_passes(joint_module)

    joint_graph = joint_module.graph

    # ── 阶段 A：节点分类（复用 default_partition 逻辑）────────────────────────
    primal_inputs = list(filter(_is_primal, joint_graph.find_nodes(op="placeholder")))
    tangent_inputs = list(filter(_is_tangent, joint_graph.find_nodes(op="placeholder")))
    fwd_seed_offset_inputs = list(filter(_is_fwd_seed_offset, joint_graph.find_nodes(op="placeholder")))

    fwd_outputs, bwd_outputs = _extract_fwd_bwd_outputs(
        joint_module, num_fwd_outputs=num_fwd_outputs
    )

    # 构建 forward-only 子图以确定前向节点集合
    fwd_inputs = primal_inputs + fwd_seed_offset_inputs
    forward_only_graph = _extract_graph_with_inputs_outputs(
        joint_graph, fwd_inputs, fwd_outputs, "forward"
    )
    forward_node_names = {
        node.name for node in forward_only_graph.nodes if node.op != "output"
    }

    # 收集 baseline saved_values 和 saved_sym_nodes（保存所有 BW 需要的前向值）
    saved_values: List[fx.Node] = []
    saved_sym_nodes: List[fx.Node] = []

    for node in joint_graph.nodes:
        if node.name not in forward_node_names:
            continue
        if is_sym_node(node):
            saved_sym_nodes.append(node)
        elif "tensor_meta" not in node.meta and node.op == "call_function":
            # 不能保存 tuple-of-tensor，需要展开为 getitem
            users = node.users
            if all(user.target == operator.getitem for user in users):
                saved_values.extend(users)
        else:
            backward_usages = [
                n for n in node.users if n.name not in forward_node_names
            ]
            if not backward_usages:
                continue
            if "tensor_meta" in node.meta and all(
                is_sym_node(n) for n in backward_usages
            ):
                saved_sym_nodes.extend(backward_usages)
            else:
                saved_values.append(node)

    saved_values = list(dict.fromkeys(saved_values))
    saved_sym_nodes = list(dict.fromkeys(saved_sym_nodes))

    baseline_count = len(saved_values)
    logger.info("[partition] baseline saved_values: %d 个节点", baseline_count)

    # ── 阶段 B：层级传播 ──────────────────────────────────────────────────────
    sorted_ranks, node_name_to_rank = _propagate_layer_ranks(
        joint_module, forward_node_names
    )

    # ── 阶段 C：策略筛选 ──────────────────────────────────────────────────────
    option, param = _parse_strategy(strategy_config)
    logger.info("[partition] 使用策略 %s，参数: %s", option, param)

    primal_names = {n.name for n in primal_inputs}

    nodes_to_recompute = _apply_strategy(
        option, param,
        saved_values, sorted_ranks, node_name_to_rank,
        forward_node_names, primal_names,
        joint_module=joint_module,
    )

    if nodes_to_recompute:
        # 移除目标节点
        recompute_set = {id(n) for n in nodes_to_recompute}
        saved_values = [sv for sv in saved_values if id(sv) not in recompute_set]

        # 精确补充：只追加被移除节点的重计算链实际依赖的 primal，
        # 而非所有缺失 primal。过度补充会导致 saved_values 膨胀，
        # 反而增加 FW→BW 传递的节点数和峰值显存。
        remaining_saved_names = {sv.name for sv in saved_values}
        required_primal_names = _find_required_primals(
            nodes_to_recompute,
            remaining_saved_names,
            primal_names,
            forward_node_names,
        )
        existing_names = {sv.name for sv in saved_values}
        extra_primals = [
            p for p in primal_inputs
            if p.name in required_primal_names
            and p.name not in existing_names
            and not is_sym_node(p)
        ]
        if extra_primals:
            saved_values.extend(extra_primals)
            logger.info(
                "[partition] 精确补充 %d 个 primal（重计算链依赖 %d 个，已有 %d 个）",
                len(extra_primals),
                len(required_primal_names),
                len(required_primal_names) - len(extra_primals),
            )

        logger.info(
            "[partition] 移除 %d 个 saved_values（改为重计算），最终剩余 %d 个",
            len(nodes_to_recompute), len(saved_values),
        )

    # ── 阶段 D：清理 mark_layer ──────────────────────────────────────────────
    _cleanup_mark_layer(joint_module, saved_values, saved_sym_nodes)

    # ── 阶段 E：图切分 ───────────────────────────────────────────────────────
    fw_module, bw_module = _extract_fwd_bwd_modules(
        joint_module,
        saved_values,
        saved_sym_nodes=saved_sym_nodes,
        num_fwd_outputs=num_fwd_outputs,
    )

    # ── 阶段 F：BW 重排序（将重计算节点推迟到真正需要时执行）──────────────────
    bw_module = reordering_to_mimic_autograd_engine(bw_module)

    logger.info(
        "[partition] 切分完成。FW 节点数: %d, BW 节点数: %d",
        len(list(fw_module.graph.nodes)),
        len(list(bw_module.graph.nodes)),
    )

    return fw_module, bw_module


# ═══════════════════════════════════════════════════════════════════════════════
#  工厂函数
# ═══════════════════════════════════════════════════════════════════════════════

def make_selective_partition_fn(
    strategy_config: Optional[dict] = None,
) -> functools.partial:
    """
    返回一个符合 partition_fn(joint_module, _joint_inputs, *, num_fwd_outputs) 签名
    的可调用对象，绑定了策略配置。

    Usage::

        partition_fn = make_selective_partition_fn({"6": 0})
        aot_module_simplified(gm, inputs, ..., partition_fn=partition_fn)
    """
    return functools.partial(
        selective_recompute_partition,
        strategy_config=strategy_config,
    )
