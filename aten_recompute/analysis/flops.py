"""
flops.py

基于 FX 图的静态 FLOPs 与执行时间估算器。

通过遍历 AOT Autograd 编译后的 FW/BW GraphModule，从 meta['val'] 提取
张量形状，映射到 ATen op 的 FLOPs 公式，无需 GPU 运行即可估算：
  1. 各策略的前向/反向 FLOPs
  2. 重计算引入的额外 FLOPs 开销
  3. 基于 Roofline 模型的理论执行时间

用法::

    from aten_recompute.analysis import FLOPsEstimator

    estimator = FLOPsEstimator()
    estimator.compare_strategies(
        model=model,
        sample_inputs=(src, tgt),
        strategies={
            "no_recompute":  {"0": None},
            "ATenIR_strat6": {"6": 0},
        },
        module_lists=lambda m: [m.encoder_layers, m.decoder_layers],
        loss_fn=lambda out: criterion(out.view(-1, V), target.view(-1)),
    )
    estimator.report()
"""

import copy
import json
import operator
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.fx as fx
import torch.nn as nn

from ..utils.logger import get_logger
from ._activations import _fmt_bytes

logger = get_logger(__name__)

# ── matplotlib（可选）──────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def _find_cjk_font():
        from matplotlib import font_manager
        for name in (
            'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Noto Sans SC',
            'SimHei', 'Microsoft YaHei', 'AR PL UMing CN',
        ):
            try:
                path = font_manager.findfont(name, fallback_to_default=False)
                if path and any(
                    part.lower() in path.lower() for part in name.lower().split()
                ):
                    return name
            except Exception:
                pass
        return None

    _CJK_FONT = _find_cjk_font()
    if _CJK_FONT:
        plt.rcParams['font.sans-serif'] = [_CJK_FONT, 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

    _MPL_AVAILABLE = True
except ImportError:
    _CJK_FONT = None
    _MPL_AVAILABLE = False

__all__ = ["FLOPsEstimator"]

_aten = torch.ops.aten

# ═══════════════════════════════════════════════════════════════════════════════
#  FLOPs 计算核心
# ═══════════════════════════════════════════════════════════════════════════════

def _get_val_shape(val: Any) -> Optional[Tuple[int, ...]]:
    """从 meta['val'] 提取形状。"""
    if isinstance(val, torch.Tensor):
        try:
            return tuple(int(s) for s in val.shape)
        except (TypeError, RuntimeError):
            return None
    return None


def _numel_from_shape(shape: Tuple[int, ...]) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


def _node_input_shapes(node: fx.Node) -> List[Optional[Tuple[int, ...]]]:
    """提取节点所有输入的形状列表。"""
    shapes = []
    for inp in node.args:
        if isinstance(inp, fx.Node):
            val = inp.meta.get("val", None)
            shapes.append(_get_val_shape(val))
        else:
            shapes.append(None)
    return shapes


def _node_output_shape(node: fx.Node) -> Optional[Tuple[int, ...]]:
    """提取节点输出形状。"""
    val = node.meta.get("val", None)
    return _get_val_shape(val)


def _estimate_node_bytes_accessed(node: fx.Node) -> int:
    """估算节点访问的总字节数（输入 + 输出），用于 Roofline 模型。"""
    total = 0
    # 输入
    for inp in node.args:
        if isinstance(inp, fx.Node):
            val = inp.meta.get("val", None)
            if isinstance(val, torch.Tensor):
                try:
                    total += int(val.numel()) * val.element_size()
                except (TypeError, RuntimeError):
                    pass
    # 输出
    val = node.meta.get("val", None)
    if isinstance(val, torch.Tensor):
        try:
            total += int(val.numel()) * val.element_size()
        except (TypeError, RuntimeError):
            pass
    elif isinstance(val, (list, tuple)):
        for v in val:
            if isinstance(v, torch.Tensor):
                try:
                    total += int(v.numel()) * v.element_size()
                except (TypeError, RuntimeError):
                    pass
    return total


# ── 借用 PyTorch 内置 flop_registry ──────────────────────────────────────────

try:
    from torch.utils.flop_counter import flop_registry as _pytorch_flop_registry
    _HAS_PYTORCH_FLOP_REGISTRY = True
except ImportError:
    _pytorch_flop_registry = {}
    _HAS_PYTORCH_FLOP_REGISTRY = False


def _pytorch_registry_flops(node: fx.Node) -> Optional[int]:
    """尝试用 PyTorch 内置 flop_registry 计算节点 FLOPs。"""
    if not _HAS_PYTORCH_FLOP_REGISTRY:
        return None
    if node.op != "call_function":
        return None

    target = node.target

    # flop_registry 的 key 是 OpOverloadPacket，不是 OpOverload
    packet = getattr(target, "overloadpacket", None)
    if packet is None:
        return None

    handler = _pytorch_flop_registry.get(packet, None)
    if handler is None:
        return None

    # handler 签名: fn(*input_shapes, out_shape=out_shape)
    input_shapes = _node_input_shapes(node)
    # 过滤 None（非张量参数）
    tensor_shapes = [s for s in input_shapes if s is not None]
    if not tensor_shapes:
        return None

    out_val = node.meta.get("val", None)
    out_shape = _get_val_shape(out_val)

    try:
        return int(handler(*tensor_shapes, out_shape=out_shape))
    except Exception:
        return None


# ── 补充 FLOPs 映射（PyTorch registry 未覆盖的常见 op）────────────────────────

_ELEMENTWISE_OPS = frozenset([
    _aten.add.Tensor, _aten.add.Scalar,
    _aten.sub.Tensor, _aten.sub.Scalar,
    _aten.mul.Tensor, _aten.mul.Scalar,
    _aten.div.Tensor, _aten.div.Scalar,
    _aten.relu.default, _aten.relu_.default,
    _aten.gelu.default,
    _aten.silu.default, _aten.silu_.default,
    _aten.sigmoid.default, _aten.sigmoid_.default,
    _aten.tanh.default, _aten.tanh_.default,
    _aten.neg.default,
    _aten.abs.default,
    _aten.exp.default,
    _aten.log.default,
    _aten.sqrt.default,
    _aten.rsqrt.default,
    _aten.pow.Tensor_Scalar,
    _aten.clamp.default, _aten.clamp_min.default,
    _aten.where.self,
    _aten.gt.Scalar, _aten.lt.Scalar, _aten.ge.Scalar, _aten.le.Scalar,
    _aten.eq.Scalar, _aten.ne.Scalar,
    _aten.gt.Tensor, _aten.lt.Tensor,
])

_REDUCTION_OPS = frozenset([
    _aten.sum.default, _aten.sum.dim_IntList,
    _aten.mean.default, _aten.mean.dim,
    _aten.max.default, _aten.min.default,
    _aten.amax.default, _aten.amin.default,
])

# softmax / layer_norm 等复合 op 的 FLOPs 倍数（相对于 numel）
_COMPOSITE_FLOPS_MULTIPLIER = {
    _aten._softmax.default: 5,          # exp + sum + div + max + sub
    _aten._softmax_backward_data.default: 4,
    _aten.native_layer_norm.default: 5,  # mean + var + normalize
    _aten.native_layer_norm_backward.default: 8,
    _aten.native_batch_norm.default: 5,
    _aten._native_batch_norm_legit_functional.default: 5,
}

# 零 FLOPs op
_ZERO_FLOPS_OPS = frozenset([
    _aten.view.default, _aten.reshape.default,
    _aten.permute.default, _aten.transpose.int, _aten.t.default,
    _aten.expand.default, _aten.unsqueeze.default,
    _aten.squeeze.default, _aten.squeeze.dim,
    _aten.slice.Tensor, _aten.select.int,
    _aten.as_strided.default, _aten.detach.default, _aten.alias.default,
    _aten._unsafe_view.default,
    _aten.split.Tensor, _aten.split_with_sizes.default,
    _aten.cat.default,  # 内存复制，非计算
    _aten.embedding.default,  # 查表
    _aten.clone.default,  # 内存复制
    _aten.copy_.default,
    _aten.zeros_like.default, _aten.ones_like.default, _aten.full_like.default,
    _aten.empty_like.default,
])


def _fallback_node_flops(node: fx.Node) -> int:
    """对 PyTorch registry 未覆盖的 op 手动估算 FLOPs。"""
    if node.op != "call_function":
        return 0

    target = node.target

    # getitem 不计算
    if target is operator.getitem:
        return 0

    # 零 FLOPs
    if target in _ZERO_FLOPS_OPS:
        return 0

    # 逐元素 op: FLOPs = numel
    if target in _ELEMENTWISE_OPS:
        out_shape = _node_output_shape(node)
        if out_shape:
            return _numel_from_shape(out_shape)
        return 0

    # 归约 op: FLOPs = input numel
    if target in _REDUCTION_OPS:
        shapes = _node_input_shapes(node)
        if shapes and shapes[0]:
            return _numel_from_shape(shapes[0])
        return 0

    # 复合 op（softmax, layer_norm 等）
    if target in _COMPOSITE_FLOPS_MULTIPLIER:
        mult = _COMPOSITE_FLOPS_MULTIPLIER[target]
        shapes = _node_input_shapes(node)
        if shapes and shapes[0]:
            return mult * _numel_from_shape(shapes[0])
        return 0

    # 未知 op: 按逐元素估算（保守下界）
    out_shape = _node_output_shape(node)
    if out_shape:
        return _numel_from_shape(out_shape)
    return 0


def estimate_graph_flops(gm: fx.GraphModule) -> Dict:
    """
    遍历 FX 图节点，估算总 FLOPs。

    Returns
    -------
    {
        'total_flops':     int,   总 FLOPs
        'total_bytes':     int,   总访问字节数（Roofline 用）
        'num_nodes':       int,   图中总节点数
        'num_compute_nodes': int, 有计算的节点数
        'per_op_flops':    dict,  {op_name: total_flops} 按算子聚合
        'top_nodes':       list,  FLOPs 最高的 10 个节点
    }
    """
    total_flops = 0
    total_bytes = 0
    num_compute = 0
    per_op_flops: Dict[str, int] = {}
    node_flops_list: List[Tuple[str, str, int]] = []

    for node in gm.graph.nodes:
        if node.op not in ("call_function",):
            continue

        # 优先用 PyTorch 内置 registry
        flops = _pytorch_registry_flops(node)
        if flops is None:
            flops = _fallback_node_flops(node)

        bytes_accessed = _estimate_node_bytes_accessed(node)

        if flops > 0:
            num_compute += 1
            total_flops += flops
            total_bytes += bytes_accessed

            op_name = _op_name(node)
            per_op_flops[op_name] = per_op_flops.get(op_name, 0) + flops
            node_flops_list.append((node.name, op_name, flops))

    # Top-10 节点
    node_flops_list.sort(key=lambda x: -x[2])
    top_nodes = [
        {"name": name, "op": op, "flops": f}
        for name, op, f in node_flops_list[:10]
    ]

    return {
        "total_flops": total_flops,
        "total_bytes": total_bytes,
        "num_nodes": len(list(gm.graph.nodes)),
        "num_compute_nodes": num_compute,
        "per_op_flops": dict(sorted(per_op_flops.items(), key=lambda x: -x[1])),
        "top_nodes": top_nodes,
    }


def _op_name(node: fx.Node) -> str:
    """返回节点的可读算子名。"""
    if node.op == "call_function":
        target = node.target
        if target is operator.getitem:
            return "getitem"
        if hasattr(target, "_opname"):
            return f"aten.{target._opname}"
        return str(target).split(".")[-1]
    return node.op


# ═══════════════════════════════════════════════════════════════════════════════
#  Roofline 时间估算
# ═══════════════════════════════════════════════════════════════════════════════

# 已知 GPU 规格（peak TFLOPS FP32, 显存带宽 GB/s）
_GPU_SPECS = {
    "A100": (19.5, 2039),      # A100 SXM 80GB
    "A100-SXM": (19.5, 2039),
    "A100-PCIE": (19.5, 1555),
    "A6000": (38.7, 768),
    "V100": (15.7, 900),
    "H100": (51.2, 3350),      # H100 SXM
    "RTX 3090": (35.6, 936),
    "RTX 4090": (82.6, 1008),
    "L40": (90.5, 864),
}


def _get_gpu_specs(device: str = "cuda") -> Tuple[float, float]:
    """
    获取 GPU 规格：(peak_tflops_fp32, bandwidth_gb_s)。

    优先匹配已知规格表，回退到保守估算。
    """
    if not torch.cuda.is_available():
        return (19.5, 2039)  # 默认 A100 规格

    props = torch.cuda.get_device_properties(device)
    name = props.name

    for known_name, (tflops, bw) in _GPU_SPECS.items():
        if known_name.lower() in name.lower():
            return (tflops, bw)

    # 回退：保守估算
    logger.warning(
        "[FLOPsEstimator] 未知 GPU '%s'，使用保守估算 (20 TFLOPS, 900 GB/s)", name
    )
    return (20.0, 900.0)


def estimate_roofline_time(
    total_flops: int,
    total_bytes: int,
    peak_tflops: float,
    bandwidth_gb: float,
) -> Dict:
    """
    Roofline 模型估算执行时间。

    Parameters
    ----------
    total_flops : 总 FLOPs
    total_bytes : 总访问字节数
    peak_tflops : GPU 峰值算力 (TFLOPS FP32)
    bandwidth_gb : GPU 显存带宽 (GB/s)

    Returns
    -------
    {
        'compute_time_ms':  计算受限时间
        'memory_time_ms':   带宽受限时间
        'estimated_time_ms': max(compute, memory)
        'arithmetic_intensity': FLOPs / bytes
        'bottleneck':       'compute' | 'memory'
    }
    """
    compute_time_ms = (total_flops / (peak_tflops * 1e12)) * 1000 if peak_tflops > 0 else 0
    memory_time_ms = (total_bytes / (bandwidth_gb * 1e9)) * 1000 if bandwidth_gb > 0 else 0
    estimated_time_ms = max(compute_time_ms, memory_time_ms)
    ai = total_flops / total_bytes if total_bytes > 0 else float('inf')
    bottleneck = "compute" if compute_time_ms >= memory_time_ms else "memory"

    return {
        "compute_time_ms": compute_time_ms,
        "memory_time_ms": memory_time_ms,
        "estimated_time_ms": estimated_time_ms,
        "arithmetic_intensity": ai,
        "bottleneck": bottleneck,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  FLOPsEstimator
# ═══════════════════════════════════════════════════════════════════════════════

_LINE = "─" * 80
_BOLD = "═" * 80


def _fmt_flops(flops: int) -> str:
    """格式化 FLOPs 为可读字符串。"""
    if flops >= 1e15:
        return f"{flops / 1e15:.2f} PFLOPs"
    if flops >= 1e12:
        return f"{flops / 1e12:.2f} TFLOPs"
    if flops >= 1e9:
        return f"{flops / 1e9:.2f} GFLOPs"
    if flops >= 1e6:
        return f"{flops / 1e6:.2f} MFLOPs"
    if flops >= 1e3:
        return f"{flops / 1e3:.2f} KFLOPs"
    return f"{flops} FLOPs"


class FLOPsEstimator:
    """
    基于 FX 图的静态 FLOPs 与执行时间估算器。

    通过遍历 AOT Autograd 编译后的 FW/BW GraphModule，从 meta['val']
    提取张量形状，映射到 ATen op 的 FLOPs 公式，无需 GPU 运行。
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._results: Dict[str, Dict] = {}
        self._gpu_specs: Optional[Tuple[float, float]] = None

    def _get_specs(self) -> Tuple[float, float]:
        if self._gpu_specs is None:
            self._gpu_specs = _get_gpu_specs(self.device)
        return self._gpu_specs

    # ── 单配置估算 ────────────────────────────────────────────────────────

    def estimate_from_graphs(
        self,
        fw_gm: fx.GraphModule,
        bw_gm: fx.GraphModule,
        tag: str = "default",
    ) -> Dict:
        """
        从已编译的 FW/BW 图估算 FLOPs 与理论执行时间。

        Parameters
        ----------
        fw_gm : 前向 GraphModule（带 meta['val']）
        bw_gm : 反向 GraphModule
        tag : 配置标签
        """
        fw_result = estimate_graph_flops(fw_gm)
        bw_result = estimate_graph_flops(bw_gm)

        total_flops = fw_result["total_flops"] + bw_result["total_flops"]
        total_bytes = fw_result["total_bytes"] + bw_result["total_bytes"]

        peak_tflops, bandwidth_gb = self._get_specs()

        fw_time = estimate_roofline_time(
            fw_result["total_flops"], fw_result["total_bytes"],
            peak_tflops, bandwidth_gb,
        )
        bw_time = estimate_roofline_time(
            bw_result["total_flops"], bw_result["total_bytes"],
            peak_tflops, bandwidth_gb,
        )
        total_time = estimate_roofline_time(
            total_flops, total_bytes, peak_tflops, bandwidth_gb,
        )

        result = {
            "tag": tag,
            "fw_flops": fw_result["total_flops"],
            "fw_bytes": fw_result["total_bytes"],
            "fw_num_compute_nodes": fw_result["num_compute_nodes"],
            "fw_per_op_flops": fw_result["per_op_flops"],
            "fw_top_nodes": fw_result["top_nodes"],
            "bw_flops": bw_result["total_flops"],
            "bw_bytes": bw_result["total_bytes"],
            "bw_num_compute_nodes": bw_result["num_compute_nodes"],
            "bw_per_op_flops": bw_result["per_op_flops"],
            "bw_top_nodes": bw_result["top_nodes"],
            "total_flops": total_flops,
            "total_bytes": total_bytes,
            "fw_roofline": fw_time,
            "bw_roofline": bw_time,
            "total_roofline": total_time,
            "gpu_peak_tflops": peak_tflops,
            "gpu_bandwidth_gb": bandwidth_gb,
        }

        self._results[tag] = result
        return result

    # ── 多配置对比 ────────────────────────────────────────────────────────

    def compare_strategies(
        self,
        model: nn.Module,
        sample_inputs: tuple,
        strategies: Dict[str, Any],
        module_lists: Optional[Callable] = None,
        loss_fn: Optional[Callable] = None,
    ) -> Dict[str, Dict]:
        """
        编译多种配置并对比 FLOPs。

        Parameters
        ----------
        model : 原始模型（会被 deepcopy）
        sample_inputs : 模型输入元组
        strategies : {标签: 策略配置} 映射（仅 dict 类型，不支持 "checkpoint"）
        module_lists : 函数 (model) -> [ModuleList, ...]
        loss_fn : 函数 (output) -> loss
        """
        import torch._dynamo

        from ..core.tag import inject_layer_tags
        from .static import _StaticAnalysisBackend

        for tag, config in strategies.items():
            if config == "checkpoint":
                logger.info("[FLOPsEstimator] 跳过 '%s'（checkpoint 无独立 FX 图）", tag)
                continue

            torch._dynamo.reset()
            t0 = time.time()

            model_copy = copy.deepcopy(model)
            model_copy.train()

            if module_lists is not None:
                all_layers = []
                rank = 0
                for ml in module_lists(model_copy):
                    for layer in ml:
                        all_layers.append((layer, rank))
                        rank += 1
                inject_layer_tags(all_layers)

            backend = _StaticAnalysisBackend(strategy_config=config, use_decomp=True, use_meta=True)
            compiled = torch.compile(model_copy, backend=backend, dynamic=True)

            try:
                output = compiled(*sample_inputs)
                if loss_fn is not None:
                    loss = loss_fn(output)
                else:
                    loss = output.sum()
                loss.backward()
            except Exception:
                logger.warning(
                    "[FLOPsEstimator] '%s' 带分解表失败，降级重试。", tag
                )
                torch._dynamo.reset()
                model_copy = copy.deepcopy(model)
                model_copy.train()
                if module_lists is not None:
                    all_layers = []
                    rank = 0
                    for ml in module_lists(model_copy):
                        for layer in ml:
                            all_layers.append((layer, rank))
                            rank += 1
                    inject_layer_tags(all_layers)

                backend = _StaticAnalysisBackend(strategy_config=config, use_decomp=False, use_meta=True)
                compiled = torch.compile(model_copy, backend=backend, dynamic=True)
                output = compiled(*sample_inputs)
                if loss_fn is not None:
                    loss = loss_fn(output)
                else:
                    loss = output.sum()
                loss.backward()

            elapsed = time.time() - t0

            if backend.fw_gm is None or backend.bw_gm is None:
                logger.warning("[FLOPsEstimator] '%s' 未捕获 FW/BW 图，跳过。", tag)
                del model_copy, compiled, backend
                continue

            print(f"  [{tag}] FLOPs 分析完成 ({elapsed:.1f}s)")
            self.estimate_from_graphs(backend.fw_gm, backend.bw_gm, tag=tag)

            del model_copy, compiled, backend

        torch._dynamo.reset()
        return self._results

    # ── 报告 ──────────────────────────────────────────────────────────────

    def report(self) -> None:
        """打印 FLOPs 与时间估算对比表。"""
        if not self._results:
            return

        print(f"\n{_BOLD}")
        print("  FLOPs & 执行时间静态估算（基于 FX 图 meta['val']，无需 GPU 运行）")
        print(_BOLD)

        tags = list(self._results.keys())
        if tags:
            r0 = self._results[tags[0]]
            print(f"  GPU: {r0['gpu_peak_tflops']:.1f} TFLOPS FP32, "
                  f"{r0['gpu_bandwidth_gb']:.0f} GB/s")
            print(_LINE)

        # ── FLOPs 汇总表 ──
        print(f"\n  {'配置':<22} {'FW FLOPs':>14} {'BW FLOPs':>14} "
              f"{'总 FLOPs':>14} {'BW/FW':>8}")
        print(f"  {'─'*22} {'─'*14} {'─'*14} {'─'*14} {'─'*8}")

        for tag, r in self._results.items():
            ratio = r["bw_flops"] / r["fw_flops"] if r["fw_flops"] > 0 else 0
            print(f"  {tag:<22} "
                  f"{_fmt_flops(r['fw_flops']):>14} "
                  f"{_fmt_flops(r['bw_flops']):>14} "
                  f"{_fmt_flops(r['total_flops']):>14} "
                  f"{ratio:>7.2f}x")

        # ── Roofline 时间估算 ──
        print(f"\n  {'配置':<22} {'FW 理论':>10} {'BW 理论':>10} "
              f"{'总理论':>10} {'瓶颈':>10}")
        print(f"  {'─'*22} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

        for tag, r in self._results.items():
            print(f"  {tag:<22} "
                  f"{r['fw_roofline']['estimated_time_ms']:>8.2f}ms "
                  f"{r['bw_roofline']['estimated_time_ms']:>8.2f}ms "
                  f"{r['total_roofline']['estimated_time_ms']:>8.2f}ms "
                  f"{r['total_roofline']['bottleneck']:>10}")

        # ── 策略间对比 ──
        if len(tags) >= 2:
            print(f"  {'─'*78}")
            baseline = self._results[tags[0]]
            for cmp_tag in tags[1:]:
                cmp = self._results[cmp_tag]

                extra_bw = cmp["bw_flops"] - baseline["bw_flops"]
                saved_fw = baseline["fw_flops"] - cmp["fw_flops"]
                total_delta = cmp["total_flops"] - baseline["total_flops"]
                pct = (total_delta / baseline["total_flops"] * 100
                       if baseline["total_flops"] > 0 else 0)

                print(f"  [{tags[0]} → {cmp_tag}]")
                print(f"    BW 额外 FLOPs（重计算开销）: {_fmt_flops(extra_bw)} "
                      f"({extra_bw / baseline['bw_flops'] * 100:+.1f}% BW)")
                print(f"    总 FLOPs 变化: {_fmt_flops(abs(total_delta))} ({pct:+.1f}%)")

                # 时间对比
                time_delta = (cmp["total_roofline"]["estimated_time_ms"] -
                              baseline["total_roofline"]["estimated_time_ms"])
                print(f"    理论时间变化: {time_delta:+.2f} ms")

        # ── Top 算子 FLOPs 分布（仅第一个配置）──
        if tags:
            r0 = self._results[tags[0]]
            print(f"\n  FW 算子 FLOPs 分布 [{tags[0]}]:")
            for op, flops in list(r0["fw_per_op_flops"].items())[:8]:
                pct = flops / r0["fw_flops"] * 100 if r0["fw_flops"] > 0 else 0
                bar = "█" * int(pct / 2)
                print(f"    {op:<40} {_fmt_flops(flops):>14} ({pct:5.1f}%) {bar}")

        print(_BOLD)

    # ── 可视化 ────────────────────────────────────────────────────────────

    def plot_flops_comparison(
        self,
        save_dir: Optional[str] = None,
        show: bool = False,
    ) -> Optional[str]:
        """生成 FLOPs 对比图（双子图：总量 + 算子分布）。"""
        if not _MPL_AVAILABLE or not self._results:
            return None

        tags = list(self._results.keys())
        n = len(tags)
        use_cjk = _CJK_FONT is not None

        fig, axes = plt.subplots(2, 1, figsize=(max(8, n * 3), 10))

        gflops = 1e9
        colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974']

        # ── 上图：FW/BW FLOPs 分组柱状图 ──
        ax1 = axes[0]
        import numpy as np
        x = np.arange(n)
        width = 0.3

        fw_vals = [self._results[t]["fw_flops"] / gflops for t in tags]
        bw_vals = [self._results[t]["bw_flops"] / gflops for t in tags]

        bars_fw = ax1.bar(x - width/2, fw_vals, width, label='Forward',
                          color='steelblue', alpha=0.85)
        bars_bw = ax1.bar(x + width/2, bw_vals, width, label='Backward',
                          color='coral', alpha=0.85)

        for bars in (bars_fw, bars_bw):
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2, h + max(fw_vals + bw_vals) * 0.01,
                             f'{h:.1f}', ha='center', va='bottom', fontsize=8)

        # 标注额外 FLOPs
        if n >= 2:
            base_bw = self._results[tags[0]]["bw_flops"]
            for i in range(1, n):
                extra = self._results[tags[i]]["bw_flops"] - base_bw
                if extra > 0:
                    pct = extra / base_bw * 100
                    ax1.annotate(
                        f'+{pct:.1f}%',
                        xy=(i + width/2, bw_vals[i]),
                        xytext=(i + width/2 + 0.15, bw_vals[i] * 0.95),
                        fontsize=9, color='red', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='red', lw=1),
                    )

        ax1.set_xticks(x)
        ax1.set_xticklabels(tags, fontsize=10)
        ax1.set_ylabel('GFLOPs', fontsize=10)
        title1 = 'FLOPs 对比（前向 / 反向）' if use_cjk else 'FLOPs Comparison (Forward / Backward)'
        ax1.set_title(title1, fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(axis='y', linestyle=':', alpha=0.4)

        # ── 下图：Roofline 时间估算 + 计算密度 ──
        ax2 = axes[1]

        fw_times = [self._results[t]["fw_roofline"]["estimated_time_ms"] for t in tags]
        bw_times = [self._results[t]["bw_roofline"]["estimated_time_ms"] for t in tags]

        bars_fw2 = ax2.bar(x - width/2, fw_times, width, label='FW (Roofline)',
                           color='steelblue', alpha=0.85)
        bars_bw2 = ax2.bar(x + width/2, bw_times, width, label='BW (Roofline)',
                           color='coral', alpha=0.85)

        for bars in (bars_fw2, bars_bw2):
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2, h + max(fw_times + bw_times) * 0.01,
                             f'{h:.2f}', ha='center', va='bottom', fontsize=8)

        ax2.set_xticks(x)
        ax2.set_xticklabels(tags, fontsize=10)
        ax2.set_ylabel('Time (ms)', fontsize=10)
        title2 = 'Roofline 理论执行时间' if use_cjk else 'Roofline Estimated Time'
        ax2.set_title(title2, fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(axis='y', linestyle=':', alpha=0.4)

        plt.tight_layout()

        path = None
        if save_dir:
            path = os.path.join(save_dir, 'flops_estimation.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            logger.info("[FLOPsEstimator] 图表已保存至: %s", path)
        if show:
            plt.show()
        plt.close(fig)
        return path

    # ── 持久化 ────────────────────────────────────────────────────────────

    def save_report(
        self,
        model_name: Optional[str] = None,
        subfolder: str = "memory",
        save_plots: bool = True,
    ) -> str:
        """将 FLOPs 估算结果保存为 JSON，并可选生成对比图。"""
        from ..utils.save_ir import _default_ir_dir

        out_dir = _default_ir_dir(
            model_name or os.getenv("MODEL_NAME", "default_model"),
            subfolder=subfolder,
        )
        path = os.path.join(out_dir, "flops_estimation.json")

        payload: Dict = {}
        payload["configurations"] = {}

        for tag, r in self._results.items():
            entry = {
                "tag": tag,
                "fw_flops": r["fw_flops"],
                "bw_flops": r["bw_flops"],
                "total_flops": r["total_flops"],
                "fw_bytes": r["fw_bytes"],
                "bw_bytes": r["bw_bytes"],
                "total_bytes": r["total_bytes"],
                "fw_num_compute_nodes": r["fw_num_compute_nodes"],
                "bw_num_compute_nodes": r["bw_num_compute_nodes"],
                "fw_per_op_flops": r["fw_per_op_flops"],
                "bw_per_op_flops": r["bw_per_op_flops"],
                "fw_roofline": r["fw_roofline"],
                "bw_roofline": r["bw_roofline"],
                "total_roofline": r["total_roofline"],
                "gpu_peak_tflops": r["gpu_peak_tflops"],
                "gpu_bandwidth_gb": r["gpu_bandwidth_gb"],
            }
            payload["configurations"][tag] = entry

        # 策略间对比
        tags = list(self._results.keys())
        if len(tags) >= 2:
            baseline = self._results[tags[0]]
            comparisons = []
            for cmp_tag in tags[1:]:
                cmp = self._results[cmp_tag]
                extra_bw = cmp["bw_flops"] - baseline["bw_flops"]
                total_delta = cmp["total_flops"] - baseline["total_flops"]
                comparisons.append({
                    "baseline": tags[0],
                    "compared": cmp_tag,
                    "extra_bw_flops": extra_bw,
                    "extra_bw_pct": round(
                        extra_bw / baseline["bw_flops"] * 100, 2
                    ) if baseline["bw_flops"] > 0 else 0,
                    "total_flops_delta": total_delta,
                    "total_flops_delta_pct": round(
                        total_delta / baseline["total_flops"] * 100, 2
                    ) if baseline["total_flops"] > 0 else 0,
                    "time_delta_ms": round(
                        cmp["total_roofline"]["estimated_time_ms"] -
                        baseline["total_roofline"]["estimated_time_ms"], 4
                    ),
                })
            payload["comparisons"] = comparisons

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        logger.info("[FLOPsEstimator] 报告已保存至: %s", path)

        if save_plots:
            self.plot_flops_comparison(save_dir=out_dir)

        return path
