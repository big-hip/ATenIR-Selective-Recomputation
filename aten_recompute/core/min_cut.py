"""
min_cut.py

基于最小割（min-cut）算法的最优重计算策略。

将 joint graph 的前向部分建模为流网络：
- 每个节点 N 拆分为 N_in / N_out，内部边容量 = 保存该节点的代价
- source 连接"禁止重计算"的节点（compute-intensive / random / primal）
- sink 连接"反向需要"的节点（baseline saved_values）
- 最小割将节点分为"保存"和"重计算"两组，最优地平衡内存与计算

参考：
- PyTorch min_cut_rematerialization_partition (torch._functorch.partitioners)
- "Training Deep Nets with Sublinear Memory Cost" (Chen et al. 2016)
- "Checkmate: Breaking the Memory Wall" (2020)

用法::

    from aten_recompute.core.min_cut import solve_min_cut_recompute

    nodes_to_recompute = solve_min_cut_recompute(
        joint_module, forward_node_names, saved_values,
        primal_names, node_name_to_rank,
    )
"""
import math
import operator
from collections import deque
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import torch
import torch.fx as fx

from ..utils.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Op 分类
# ═══════════════════════════════════════════════════════════════════════════════

_aten = torch.ops.aten

# 计算密集型 op —— 重计算代价太高，必须保存
COMPUTE_INTENSIVE_OPS: FrozenSet = frozenset([
    _aten.mm.default,
    _aten.bmm.default,
    _aten.addmm.default,
    _aten.convolution.default,
    _aten.conv2d.default,
    _aten.linear.default,
    _aten._scaled_dot_product_flash_attention.default,
    _aten._scaled_dot_product_efficient_attention.default,
    _aten._flash_attention_forward.default,
    _aten._efficient_attention_forward.default,
    _aten._scaled_mm.default,
])

# 随机 op —— 结果不可复现，必须保存
# 注意：_recursive_joint_graph_passes 通常已将这些替换为 inductor_seeds/inductor_random，
# 此处作为安全网保留
RANDOM_OPS: FrozenSet = frozenset([
    _aten.native_dropout.default,
    _aten.rand_like.default,
    _aten.randn_like.default,
    _aten.bernoulli_.float,
    _aten.bernoulli.p,
])


def _get_aten_target(node: fx.Node):
    """获取节点的 aten op target（overloadpacket 级别），用于集合匹配。"""
    if node.op != "call_function":
        return None
    target = node.target
    if hasattr(target, "overloadpacket"):
        return target  # 已经是具体 overload
    return target


# ═══════════════════════════════════════════════════════════════════════════════
#  代价估算
# ═══════════════════════════════════════════════════════════════════════════════

def _node_memory_bytes(node: fx.Node) -> int:
    """
    估算节点输出张量的内存占用（字节）。

    从 node.meta["val"]（FakeTensor）提取 numel * element_size。
    """
    val = node.meta.get("val", None)
    if val is None:
        # 尝试从 tensor_meta 获取
        tm = node.meta.get("tensor_meta", None)
        if tm is not None:
            try:
                numel = 1
                for s in tm.shape:
                    numel *= int(s)
                return numel * tm.dtype.itemsize
            except (TypeError, AttributeError):
                return 0
        return 0

    return _val_memory_bytes(val)


def _val_memory_bytes(val: Any) -> int:
    """递归计算 meta['val'] 的内存占用。"""
    if isinstance(val, torch.Tensor):
        try:
            return int(val.numel()) * val.element_size()
        except Exception:
            return 0
    if isinstance(val, (torch.SymInt, torch.SymBool)):
        return 1
    if isinstance(val, torch.SymFloat):
        return 4
    if isinstance(val, (list, tuple)):
        return sum(_val_memory_bytes(v) for v in val)
    return 0


# ═══════════════════════════════════════════════════════════════════════════════
#  距离计算
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_dist_from_bw(
    joint_module: fx.GraphModule,
    forward_node_names: Set[str],
) -> Dict[str, int]:
    """
    计算每个前向节点到反向图的"距离"。

    反向节点 dist = 0，前向节点 dist = min(所有 user 的 dist) + 1。
    距离越大 → 离反向越远 → 越适合被重计算（保存代价乘数越高）。
    """
    INF_DIST = 10000
    dist: Dict[str, int] = {}

    # 初始化：反向节点 dist=0，前向节点 dist=INF
    for node in joint_module.graph.nodes:
        if node.op == "output":
            dist[node.name] = INF_DIST
        elif node.name not in forward_node_names:
            dist[node.name] = 0  # 反向节点
        else:
            dist[node.name] = INF_DIST  # 前向节点，待计算

    # 反向传播距离：从后往前遍历
    for node in reversed(list(joint_module.graph.nodes)):
        if node.name not in forward_node_names:
            continue
        # 该节点的距离 = min(所有 user 的距离) + 1
        user_dists = [dist.get(u.name, INF_DIST) for u in node.users]
        if user_dists:
            dist[node.name] = min(user_dists) + 1

    return dist


# ═══════════════════════════════════════════════════════════════════════════════
#  重计算禁令判断
# ═══════════════════════════════════════════════════════════════════════════════

def _should_ban_recomputation(
    node: fx.Node,
    forward_node_names: Set[str],
    custom_ban_ops: Optional[FrozenSet] = None,
) -> bool:
    """
    判断节点是否应禁止被重计算（必须保存）。

    禁止条件：
    1. 计算密集型 op（matmul, conv, attention 等）
    2. 随机 op（dropout, rand 等）
    3. reduction 类（输出远小于输入，保存更划算）
    4. 用户自定义禁止列表
    """
    if node.op != "call_function":
        return False

    target = node.target

    # operator.getitem 是透明的
    if target is operator.getitem:
        return False

    # 计算密集型
    if target in COMPUTE_INTENSIVE_OPS:
        return True

    # 随机 op
    if target in RANDOM_OPS:
        return True

    # 用户自定义
    if custom_ban_ops and target in custom_ban_ops:
        return True

    # inductor seeds/random（_recursive_joint_graph_passes 替换后的形式）
    target_name = getattr(target, "__name__", str(target))
    if "inductor_seeds" in target_name or "inductor_lookup_seed" in target_name:
        return True

    # Reduction 启发式：输出远小于输入 → 保存更划算
    out_bytes = _node_memory_bytes(node)
    if out_bytes > 0 and node.all_input_nodes:
        in_bytes = sum(_node_memory_bytes(inp) for inp in node.all_input_nodes
                       if inp.name in forward_node_names)
        if in_bytes > 0 and out_bytes * 4 < in_bytes:
            return True

    return False


# ═══════════════════════════════════════════════════════════════════════════════
#  流网络构建
# ═══════════════════════════════════════════════════════════════════════════════

def _build_flow_graph(
    joint_module: fx.GraphModule,
    forward_node_names: Set[str],
    saved_value_names: Set[str],
    primal_names: Set[str],
    dist_from_bw: Dict[str, int],
    ban_fn,
):
    """
    构建 NetworkX 有向流网络。

    节点拆分模型：每个 FX 节点 N → (N_in, N_out)
    - N_in → N_out 边：容量 = mem_bytes * (1.1 ^ dist_from_bw)
    - N_out → U_in 边：容量 = ∞（数据依赖）
    - source → N_in 边：容量 = ∞（禁止重计算的节点）
    - N_in → sink 边：容量 = ∞（反向需要的节点）
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError(
            "策略 7 (min-cut) 需要 networkx。请安装: pip install networkx"
        )

    graph = nx.DiGraph()
    INF = float("inf")

    for node in joint_module.graph.nodes:
        if node.name not in forward_node_names:
            continue
        if node.op == "output":
            continue

        n_in = node.name + "_in"
        n_out = node.name + "_out"

        # ── 节点内部边（N_in → N_out）──────────────────────────
        # 所有节点均使用有限容量（包括 placeholder），
        # 避免与 source/sink 无穷边形成全无穷路径
        mem_bytes = _node_memory_bytes(node)
        d = min(max(dist_from_bw.get(node.name, 1), 1), 100)
        weight = max(mem_bytes * (1.1 ** d), 1.0)
        graph.add_edge(n_in, n_out, capacity=weight)

        # ── 数据依赖边（N_out → U_in）────────────────────────
        for user in node.users:
            if user.name in forward_node_names and user.op != "output":
                graph.add_edge(n_out, user.name + "_in", capacity=INF)

        # ── source 连接（禁止重计算的节点）──────────────────
        is_banned = (
            node.op == "placeholder"
            or node.name in primal_names
            or ban_fn(node)
        )
        if is_banned:
            graph.add_edge("source", n_in, capacity=INF)

        # ── sink 连接（反向需要、且可被重计算的节点）────────
        # 关键：连接 N_out → sink（不是 N_in → sink！）
        # 这样 flow 必须经过 N_in→N_out 内部边才能到 sink，
        # min-cut 才能通过切割内部边来决定"保存此节点"
        if node.name in saved_value_names and not is_banned:
            graph.add_edge(n_out, "sink", capacity=INF)

    # 确保 source 和 sink 存在
    if "source" not in graph:
        graph.add_node("source")
    if "sink" not in graph:
        graph.add_node("sink")

    return graph


# ═══════════════════════════════════════════════════════════════════════════════
#  公开 API
# ═══════════════════════════════════════════════════════════════════════════════

def solve_min_cut_recompute(
    joint_module: fx.GraphModule,
    forward_node_names: Set[str],
    saved_values: List[fx.Node],
    primal_names: Set[str],
    node_name_to_rank: Dict[str, int],
    memory_budget: float = 1.0,
    custom_ban_ops: Optional[FrozenSet] = None,
) -> List[fx.Node]:
    """
    使用 min-cut 算法求解最优重计算集合。

    Parameters
    ----------
    joint_module : 联合前向-反向 FX 图
    forward_node_names : 前向节点名称集合
    saved_values : baseline 中需要保存的节点列表
    primal_names : primal placeholder 名称集合
    node_name_to_rank : 节点 → 层级 rank 映射
    memory_budget : 内存预算比例。1.0 = 标准 min-cut；
                    < 1.0 = 只重计算到节省 (1 - budget) 比例为止
    custom_ban_ops : 额外禁止重计算的 op 集合

    Returns
    -------
    nodes_to_recompute : 应从 saved_values 中移除（改为重计算）的节点列表
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError(
            "策略 7 (min-cut) 需要 networkx 库。请安装: pip install networkx"
        )

    # 候选：只有非 placeholder 的 saved_values 才可能被重计算
    candidates = [sv for sv in saved_values if sv.op != "placeholder"]
    if not candidates:
        logger.info("[min_cut] 没有可重计算的候选节点。")
        return []

    saved_value_names = {sv.name for sv in saved_values}

    # ── 1. 计算距离 ──────────────────────────────────────────────────
    dist_from_bw = _compute_dist_from_bw(joint_module, forward_node_names)

    # ── 2. 构建禁止判断函数 ──────────────────────────────────────────
    def ban_fn(node):
        return _should_ban_recomputation(node, forward_node_names, custom_ban_ops)

    # ── 3. 构建流网络 ────────────────────────────────────────────────
    nx_graph = _build_flow_graph(
        joint_module, forward_node_names, saved_value_names,
        primal_names, dist_from_bw, ban_fn,
    )

    # 检查图的有效性
    if "source" not in nx_graph or "sink" not in nx_graph:
        logger.warning("[min_cut] 流网络缺少 source 或 sink，跳过 min-cut。")
        return []

    if nx_graph.number_of_edges() == 0:
        logger.warning("[min_cut] 流网络没有边，跳过 min-cut。")
        return []

    # ── 4. 求解最小割 ────────────────────────────────────────────────
    try:
        cut_value, (reachable, non_reachable) = nx.minimum_cut(
            nx_graph, "source", "sink"
        )
    except (nx.NetworkXError, nx.NetworkXUnbounded) as e:
        logger.warning("[min_cut] min-cut 求解失败: %s。回退到不重计算。", e)
        return []

    logger.info("[min_cut] min-cut 值: %.2f, source 侧: %d 节点, sink 侧: %d 节点",
                cut_value, len(reachable), len(non_reachable))

    # ── 5. 提取被切割的节点（这些节点应该被保存）─────────────────────
    # cutset = 从 reachable 到 non_reachable 的边
    cut_node_names: Set[str] = set()
    for u in reachable:
        if u not in nx_graph:
            continue
        for v in nx_graph[u]:
            if v in non_reachable:
                # 如果是 N_in → N_out 边被切割，说明该节点应保存
                if u.endswith("_in") and v.endswith("_out"):
                    base_u = u[:-3]   # 去掉 "_in"
                    base_v = v[:-4]   # 去掉 "_out"
                    if base_u == base_v:
                        cut_node_names.add(base_u)

    # ── 6. 确定重计算节点 ────────────────────────────────────────────
    # 不在 cut_node_names 中的候选节点 = 不需要保存 = 可以重计算
    nodes_to_recompute = [sv for sv in candidates if sv.name not in cut_node_names]

    logger.info(
        "[min_cut] min-cut 决定：保存 %d 个节点，重计算 %d 个节点（共 %d 个候选）",
        len(cut_node_names), len(nodes_to_recompute), len(candidates),
    )

    # ── 7. 内存预算过滤 ──────────────────────────────────────────────
    if memory_budget < 1.0 and nodes_to_recompute:
        total_saved_bytes = sum(_node_memory_bytes(sv) for sv in candidates)
        target_bytes = total_saved_bytes * (1.0 - memory_budget)

        # 按内存从大到小排序：优先重计算大张量以最大化节省
        sorted_nodes = sorted(nodes_to_recompute, key=_node_memory_bytes, reverse=True)
        cumulative = 0
        filtered = []
        for n in sorted_nodes:
            if cumulative >= target_bytes:
                break
            filtered.append(n)
            cumulative += _node_memory_bytes(n)

        logger.info(
            "[min_cut] 内存预算 %.1f%%：从 %d 个候选中选择 %d 个重计算"
            "（目标节省 %.2f MB，实际 %.2f MB）",
            memory_budget * 100, len(nodes_to_recompute), len(filtered),
            target_bytes / 1e6, cumulative / 1e6,
        )
        nodes_to_recompute = filtered

    # 按内存统计
    total_recomp_bytes = sum(_node_memory_bytes(n) for n in nodes_to_recompute)
    total_saved_bytes = sum(_node_memory_bytes(sv) for sv in candidates)
    logger.info(
        "[min_cut] 预计节省 %.2f MB / %.2f MB 激活内存 (%.1f%%)",
        total_recomp_bytes / 1e6, total_saved_bytes / 1e6,
        (total_recomp_bytes / max(total_saved_bytes, 1)) * 100,
    )

    return nodes_to_recompute
