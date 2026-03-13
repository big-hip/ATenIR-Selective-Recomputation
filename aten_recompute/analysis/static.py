"""
static.py

基于 FX 图的静态峰值显存估算器。

通过 Storage 级生命周期仿真估算训练峰值显存，无需实际运行模型。
借鉴 PyTorch Inductor estimate_peak_memory 的事件驱动方法：
  1. 遍历 FX 图节点，按执行顺序记录 alloc/free 事件
  2. View op（share storage）不计新分配
  3. 前缀和求峰值：peak = max(cumsum(alloc - free))

用法::

    from aten_recompute.analysis import StaticEstimator

    estimator = StaticEstimator()
    estimator.compare_strategies(
        model=model,
        sample_inputs=(src, tgt),
        strategies={
            "no_recompute":   {"0": None},
            "ATenIR_strat6":  {"6": 0},
            "pytorch_ckpt":   "checkpoint",
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
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.fx as fx
import torch.nn as nn

from ..utils.graph_utils import get_output_node
from ..utils.logger import get_logger
from ._activations import _fmt_bytes, _saved_activation_bytes

logger = get_logger(__name__)


def _safe_decomp_table():
    """
    过滤 Inductor 分解表，移除 inductor 命名空间的算子。

    标准 ATen 分解（如 native_dropout → mul/gt/…）可安全在 eager 模式下执行；
    仅 inductor 私有算子（如 inductor._alloc_from_pool）需排除。
    """
    try:
        from torch._inductor.decomposition import select_decomp_table
        table = select_decomp_table()
        return {k: v for k, v in table.items()
                if not str(k).startswith("inductor.")}
    except ImportError:
        logger.warning("[StaticAnalysis] 无法导入 Inductor 分解表，使用默认分解。")
        return {}

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

__all__ = ["StaticEstimator"]


# ═══════════════════════════════════════════════════════════════════════════════
#  View Op 集合
# ═══════════════════════════════════════════════════════════════════════════════

_aten = torch.ops.aten

VIEW_OPS: frozenset = frozenset([
    _aten.view.default,
    _aten.reshape.default,
    _aten.slice.Tensor,
    _aten.permute.default,
    _aten.expand.default,
    _aten.unsqueeze.default,
    _aten.squeeze.default,
    _aten.squeeze.dim,
    _aten.transpose.int,
    _aten.t.default,
    _aten._unsafe_view.default,
    _aten.select.int,
    _aten.as_strided.default,
    _aten.detach.default,
    _aten.alias.default,
    _aten.split.Tensor,
    _aten.split_with_sizes.default,
    _aten.unfold.default,
    _aten.narrow.default,
])


# ═══════════════════════════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def _is_view_op(node: fx.Node) -> bool:
    """判断节点是否为 view op（共享输入 storage，不分配新内存）。"""
    if node.op != "call_function":
        return False
    if node.target is operator.getitem:
        return True
    return node.target in VIEW_OPS


def _val_bytes(val: Any) -> int:
    """递归计算 meta['val'] 的字节数（处理 Tensor/tuple/list/SymInt）。"""
    if isinstance(val, torch.Tensor):
        try:
            return int(val.numel()) * val.element_size()
        except (TypeError, RuntimeError):
            return 0
    if isinstance(val, (torch.SymInt, torch.SymBool)):
        return 0
    if isinstance(val, torch.SymFloat):
        return 0
    if isinstance(val, (list, tuple)):
        return sum(_val_bytes(v) for v in val)
    return 0


def _node_storage_bytes(node: fx.Node) -> int:
    """
    估算节点输出分配的新 storage 字节数。

    - placeholder / output 节点 → 0（非图内分配事件）
    - view op → 0（共享输入 storage）
    - 其他 → 从 meta['val'] 估算输出大小
    """
    if node.op in ("placeholder", "output", "get_attr"):
        return 0

    if _is_view_op(node):
        return 0

    val = node.meta.get("val", None)
    if val is not None:
        return _val_bytes(val)

    # fallback: tensor_meta
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


# ═══════════════════════════════════════════════════════════════════════════════
#  核心仿真：Storage 级生命周期分析
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_graph_peak_memory(
    gm: fx.GraphModule,
    pin_output_inputs: bool = False,
) -> Dict:
    """
    在 FX 图上仿真 storage 分配/释放，用事件驱动法求峰值。

    Parameters
    ----------
    gm : FX GraphModule（带 meta['val']）
    pin_output_inputs : 若为 True，output 节点的输入不在本图内释放
                        （用于 FW 图，因为 saved activations 生命周期延伸到 BW）

    Returns
    -------
    {
        'peak_bytes':       int,   峰值活跃内存
        'peak_step':        int,   峰值发生的执行步号
        'total_allocated':  int,   所有分配的总和（不计释放）
        'num_nodes':        int,   图中总节点数
        'num_alloc_nodes':  int,   产生新分配的节点数
        'node_sizes':       dict,  {node_name: bytes} 非零分配节点
        'timeline':         list,  [(step, cumulative_bytes), ...]
    }
    """
    nodes = list(gm.graph.nodes)
    num_steps = len(nodes)

    # Step 1: 为每个节点分配执行步号
    node_to_step: Dict[fx.Node, int] = {}
    for step, node in enumerate(nodes):
        node_to_step[node] = step

    # Step 2: 计算每个节点的最后使用步
    last_use: Dict[str, int] = {}
    for node in nodes:
        for inp in node.all_input_nodes:
            cur = last_use.get(inp.name, node_to_step.get(inp, 0))
            last_use[inp.name] = max(cur, node_to_step[node])

    # 无 user 的节点：last_use = 自身步号
    for node in nodes:
        if node.name not in last_use:
            last_use[node.name] = node_to_step[node]

    # Step 3: FW 图的 output 输入不释放（saved activations 延伸到 BW）
    if pin_output_inputs:
        output_node = get_output_node(gm)
        if output_node is not None:
            for inp in output_node.all_input_nodes:
                last_use[inp.name] = num_steps  # 永不在 FW 内释放

    # Step 4: 构建 alloc/free 事件数组
    alloc_at = [0] * (num_steps + 2)
    free_at = [0] * (num_steps + 2)
    node_sizes: Dict[str, int] = {}

    for node in nodes:
        nbytes = _node_storage_bytes(node)
        if nbytes <= 0:
            continue

        step = node_to_step[node]
        last = last_use.get(node.name, step)

        alloc_at[step] += nbytes
        if last + 1 < num_steps + 2:
            free_at[last + 1] += nbytes
        node_sizes[node.name] = nbytes

    # Step 5: 前缀和求峰值
    live = 0
    peak_bytes = 0
    peak_step = 0
    timeline: List[tuple] = []

    for step in range(num_steps):
        live += alloc_at[step] - free_at[step]
        timeline.append((step, live))
        if live > peak_bytes:
            peak_bytes = live
            peak_step = step

    total_allocated = sum(alloc_at)

    return {
        "peak_bytes": peak_bytes,
        "peak_step": peak_step,
        "total_allocated": total_allocated,
        "num_nodes": num_steps,
        "num_alloc_nodes": len(node_sizes),
        "node_sizes": node_sizes,
        "timeline": timeline,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  轻量后端：跳过 Inductor 编译，仅捕获 FW/BW 图
# ═══════════════════════════════════════════════════════════════════════════════

class _StaticAnalysisBackend:
    """
    跳过 Inductor 编译的轻量 torch.compile 后端。

    与 CompilerBackend 共享 partition_fn 逻辑，但 fw_compiler / bw_compiler
    直接返回 gm.forward（解释执行），省去 Triton kernel 生成耗时。
    仅用于 StaticEstimator 的图捕获场景。
    """

    def __init__(self, strategy_config: Optional[dict] = None, use_decomp: bool = True):
        self.strategy_config = strategy_config
        self.use_decomp = use_decomp
        self.fw_gm: Optional[fx.GraphModule] = None
        self.bw_gm: Optional[fx.GraphModule] = None

    def __call__(self, gm: fx.GraphModule, sample_inputs: list):
        from torch._functorch.aot_autograd import aot_module_simplified
        from torch._functorch._aot_autograd.utils import make_boxed_func
        from torch._guards import detect_fake_mode

        from ..core.partition import make_selective_partition_fn

        partition_fn = make_selective_partition_fn(self.strategy_config)

        def fw_compiler(gm, _sample_inputs):
            self.fw_gm = copy.deepcopy(gm)
            return make_boxed_func(gm.forward)

        def bw_compiler(gm, _sample_inputs):
            self.bw_gm = copy.deepcopy(gm)
            return make_boxed_func(gm.forward)

        fake_mode = detect_fake_mode(sample_inputs)
        if not fake_mode:
            fake_mode = torch._subclasses.FakeTensorMode(
                allow_non_fake_inputs=True
            )

        kwargs = dict(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=partition_fn,
        )
        if self.use_decomp:
            kwargs["decompositions"] = _safe_decomp_table()

        return aot_module_simplified(gm, sample_inputs, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
#  StaticEstimator
# ═══════════════════════════════════════════════════════════════════════════════

_LINE = "─" * 68
_BOLD = "═" * 68


class StaticEstimator:
    """
    基于 FX 图的静态峰值显存估算器。

    通过分析 AOT Autograd 编译后的 FW/BW GraphModule 中的 FakeTensor
    元信息，仿真 storage 生命周期，估算训练峰值显存。无需 GPU。
    """

    def __init__(self):
        self._results: Dict[str, Dict] = {}
        self._param_memory: Optional[Dict] = None

    # ── 单配置估算 ────────────────────────────────────────────────────────

    def estimate_from_graphs(
        self,
        fw_gm: fx.GraphModule,
        bw_gm: fx.GraphModule,
        model: nn.Module,
        optimizer_type: str = "adam",
        tag: str = "default",
    ) -> Dict:
        """
        从已编译的 FW/BW 图估算峰值显存。

        Parameters
        ----------
        fw_gm : 前向 GraphModule（带 meta['val']）
        bw_gm : 反向 GraphModule
        model : 原始模型（用于参数内存估算）
        optimizer_type : "adam"(2x) / "adamw"(2x) / "sgd"(0x) / "sgd_momentum"(1x)
        tag : 配置标签
        """
        # 1. 参数内存（所有配置共享，只算一次）
        if self._param_memory is None:
            param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
            self._param_memory = {
                "param_bytes": param_bytes,
                "buffer_bytes": buffer_bytes,
                "total_bytes": param_bytes + buffer_bytes,
            }

        param_bytes = self._param_memory["param_bytes"]

        # 2. FW 峰值仿真（pin_output_inputs=True：saved activations 不释放）
        fw_result = estimate_graph_peak_memory(fw_gm, pin_output_inputs=True)

        # 3. Saved activations（FW→BW 边界）
        saved_info = _saved_activation_bytes(fw_gm, bw_gm)
        saved_act_bytes = saved_info["activation_bytes"]

        # 4. BW 峰值仿真（含重计算节点的临时内存）
        bw_result = estimate_graph_peak_memory(bw_gm, pin_output_inputs=False)

        # 5. 梯度 ≈ 参数大小
        grad_bytes = param_bytes

        # 6. 优化器状态
        opt_mult = {"adam": 2, "adamw": 2, "sgd": 0, "sgd_momentum": 1}.get(
            optimizer_type, 2
        )
        optimizer_bytes = param_bytes * opt_mult

        # 7. 总峰值估算
        resident = param_bytes + grad_bytes + optimizer_bytes
        fw_act_peak = fw_result["peak_bytes"]
        bw_act_peak = saved_act_bytes + bw_result["peak_bytes"]
        estimated_peak = resident + max(fw_act_peak, bw_act_peak)

        fw_phase = resident + fw_act_peak
        bw_phase = resident + bw_act_peak
        opt_phase = resident

        result = {
            "tag": tag,
            "param_bytes": param_bytes,
            "buffer_bytes": self._param_memory["buffer_bytes"],
            "fw_peak_bytes": fw_result["peak_bytes"],
            "fw_total_allocated": fw_result["total_allocated"],
            "fw_num_nodes": fw_result["num_nodes"],
            "fw_num_alloc_nodes": fw_result["num_alloc_nodes"],
            "saved_act_bytes": saved_act_bytes,
            "saved_act_count": saved_info["num_activations"],
            "bw_peak_bytes": bw_result["peak_bytes"],
            "bw_total_allocated": bw_result["total_allocated"],
            "bw_num_nodes": bw_result["num_nodes"],
            "bw_num_alloc_nodes": bw_result["num_alloc_nodes"],
            "grad_bytes": grad_bytes,
            "optimizer_bytes": optimizer_bytes,
            "optimizer_type": optimizer_type,
            "fw_phase_peak": fw_phase,
            "bw_phase_peak": bw_phase,
            "opt_phase_peak": opt_phase,
            "estimated_peak": estimated_peak,
            "fw_timeline": fw_result["timeline"],
            "bw_timeline": bw_result["timeline"],
        }

        self._results[tag] = result
        return result

    # ── 多配置对比 ────────────────────────────────────────────────────────

    def compare_strategies(
        self,
        model: nn.Module,
        sample_inputs: tuple,
        strategies: Dict[str, Any],
        optimizer_type: str = "adam",
        module_lists: Optional[Callable] = None,
        loss_fn: Optional[Callable] = None,
    ) -> Dict[str, Dict]:
        """
        编译多种配置并静态对比峰值显存。

        Parameters
        ----------
        model : 原始模型（会被 deepcopy）
        sample_inputs : 模型输入元组
        strategies : {标签: 策略配置} 映射
            策略配置为 dict（如 {"6": 0}）或 "checkpoint"（PyTorch 原生 checkpoint）
        optimizer_type : 优化器类型
        module_lists : 函数 (model) -> [ModuleList, ...]，用于 checkpoint 包装
        loss_fn : 函数 (output) -> loss，用于触发 backward
        """
        import torch._dynamo

        from ..core.tag import inject_layer_tags

        # 先处理需要编译的策略，再推导 checkpoint
        checkpoint_tags = []
        compile_strategies = {}
        for tag, config in strategies.items():
            if config == "checkpoint":
                checkpoint_tags.append(tag)
            else:
                compile_strategies[tag] = config

        # ── 编译策略 ──────────────────────────────────────────────────────
        for tag, config in compile_strategies.items():
            torch._dynamo.reset()
            t0 = time.time()

            model_copy = copy.deepcopy(model)
            model_copy.train()

            # 注入层级标签
            if module_lists is not None:
                all_layers = []
                rank = 0
                for ml in module_lists(model_copy):
                    for layer in ml:
                        all_layers.append((layer, rank))
                        rank += 1
                inject_layer_tags(all_layers)

            backend = _StaticAnalysisBackend(strategy_config=config, use_decomp=True)
            compiled = torch.compile(model_copy, backend=backend, dynamic=True)

            try:
                output = compiled(*sample_inputs)
                if loss_fn is not None:
                    loss = loss_fn(output)
                else:
                    loss = output.sum()
                loss.backward()
            except Exception as _decomp_err:
                logger.warning(
                    "[StaticEstimator] '%s' 带分解表执行失败 (%s)，降级重试。",
                    tag, type(_decomp_err).__name__,
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

                backend = _StaticAnalysisBackend(
                    strategy_config=config, use_decomp=False,
                )
                compiled = torch.compile(model_copy, backend=backend, dynamic=True)
                output = compiled(*sample_inputs)
                if loss_fn is not None:
                    loss = loss_fn(output)
                else:
                    loss = output.sum()
                loss.backward()

            elapsed = time.time() - t0

            if backend.fw_gm is None or backend.bw_gm is None:
                logger.warning("[StaticEstimator] '%s' 编译未捕获 FW/BW 图，跳过。", tag)
                del model_copy, compiled, backend
                continue

            print(f"  [{tag}] 编译完成 ({elapsed:.1f}s)，"
                  f"FW {len(list(backend.fw_gm.graph.nodes))} 节点"
                  f" / BW {len(list(backend.bw_gm.graph.nodes))} 节点")

            self.estimate_from_graphs(
                backend.fw_gm, backend.bw_gm, model,
                optimizer_type=optimizer_type, tag=tag,
            )

            del model_copy, compiled, backend

        # ── Checkpoint 公式推导 ───────────────────────────────────────────
        for tag in checkpoint_tags:
            baseline_tag = next(
                (t for t in self._results if "no_recompute" in t),
                next(iter(self._results), None),
            )
            if baseline_tag is None:
                logger.warning(
                    "[StaticEstimator] 需先估算至少一个编译配置才能推导 checkpoint"
                )
                continue

            baseline = self._results[baseline_tag]

            num_layers = (
                sum(len(ml) for ml in module_lists(model))
                if module_lists else 1
            )

            boundary_bytes_per_layer = 0
            if module_lists is not None:
                lists = module_lists(model)
                hidden_dim = None
                for ml in lists:
                    if len(ml) > 0:
                        for p in ml[0].parameters():
                            if p.dim() >= 2:
                                hidden_dim = p.shape[-1]
                                break
                        break
                if hidden_dim and len(sample_inputs) > 0:
                    inp = sample_inputs[0]
                    batch = inp.shape[0] if inp.dim() >= 1 else 1
                    seq = inp.shape[1] if inp.dim() >= 2 else 1
                    elem_size = 4  # float32
                    boundary_bytes_per_layer = batch * seq * hidden_dim * elem_size

            if boundary_bytes_per_layer > 0:
                saved_act_bytes = num_layers * boundary_bytes_per_layer
            else:
                saved_act_bytes = baseline["saved_act_bytes"] // num_layers

            param_bytes = baseline["param_bytes"]
            grad_bytes = baseline["grad_bytes"]
            optimizer_bytes = baseline["optimizer_bytes"]

            # ── 基于 timeline 的逐层峰值分析 ──────────────────────────────
            fw_timeline = baseline.get("fw_timeline", [])
            bw_timeline = baseline.get("bw_timeline", [])

            def _per_layer_peak(timeline, num_layers):
                """将 timeline 按层数分段，返回最大单层增量峰值。"""
                if not timeline or num_layers <= 0:
                    return 0
                chunk_size = max(1, len(timeline) // num_layers)
                per_layer_peaks = []
                for i in range(num_layers):
                    start = i * chunk_size
                    end = (i + 1) * chunk_size if i < num_layers - 1 else len(timeline)
                    chunk = timeline[start:end]
                    if chunk:
                        base_val = chunk[0][1]
                        local_peak = max(v for _, v in chunk) - base_val
                        per_layer_peaks.append(max(local_peak, 0))
                return max(per_layer_peaks) if per_layer_peaks else 0

            if fw_timeline:
                fw_peak_bytes = _per_layer_peak(fw_timeline, num_layers)
            else:
                fw_peak_bytes = baseline["fw_peak_bytes"] // num_layers

            if bw_timeline:
                bw_layer_peak = _per_layer_peak(bw_timeline, num_layers)
            else:
                bw_layer_peak = baseline["bw_peak_bytes"] // num_layers

            bw_peak_bytes = bw_layer_peak + fw_peak_bytes

            resident = param_bytes + grad_bytes + optimizer_bytes
            fw_act_peak = fw_peak_bytes + saved_act_bytes
            bw_act_peak = saved_act_bytes + bw_peak_bytes
            estimated_peak = resident + max(fw_act_peak, bw_act_peak)

            fw_phase = resident + fw_act_peak
            bw_phase = resident + bw_act_peak
            opt_phase = resident

            result = {
                "tag": tag,
                "param_bytes": param_bytes,
                "buffer_bytes": baseline["buffer_bytes"],
                "fw_peak_bytes": fw_peak_bytes,
                "fw_total_allocated": baseline["fw_total_allocated"] // num_layers,
                "fw_num_nodes": baseline["fw_num_nodes"],
                "fw_num_alloc_nodes": baseline["fw_num_alloc_nodes"] // num_layers,
                "saved_act_bytes": saved_act_bytes,
                "saved_act_count": num_layers,
                "bw_peak_bytes": bw_peak_bytes,
                "bw_total_allocated": baseline["bw_total_allocated"] // num_layers,
                "bw_num_nodes": baseline["bw_num_nodes"],
                "bw_num_alloc_nodes": baseline["bw_num_alloc_nodes"] // num_layers,
                "grad_bytes": grad_bytes,
                "optimizer_bytes": optimizer_bytes,
                "optimizer_type": baseline["optimizer_type"],
                "fw_phase_peak": fw_phase,
                "bw_phase_peak": bw_phase,
                "opt_phase_peak": opt_phase,
                "estimated_peak": estimated_peak,
                "note": "基于公式推导（checkpoint 行为无法通过 torch.compile 捕获）",
            }
            self._results[tag] = result
            print(f"  [{tag}] 基于 {baseline_tag} 公式推导（跳过编译）")

        torch._dynamo.reset()
        return self._results

    # ── 报告输出 ──────────────────────────────────────────────────────────

    def report(self) -> None:
        """打印静态峰值显存估算对比表。"""
        if not self._results:
            return

        print(f"\n{_BOLD}")
        print("  静态峰值显存估算（基于 FX 图 FakeTensor 元信息，无需 GPU 运行）")
        print(_BOLD)

        if self._param_memory:
            pm = self._param_memory
            print(f"  模型参数: {_fmt_bytes(pm['param_bytes'])}  "
                  f"缓冲区: {_fmt_bytes(pm['buffer_bytes'])}")
            print(_LINE)

        print(f"\n  {'配置':<20} {'FW 峰值':>10} {'保存激活':>10} "
              f"{'BW 峰值':>10} {'估算总峰值':>14}")
        print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10} {'─'*14}")

        for tag, r in self._results.items():
            print(f"  {tag:<20} "
                  f"{_fmt_bytes(r['fw_peak_bytes']):>10} "
                  f"{_fmt_bytes(r['saved_act_bytes']):>10} "
                  f"{_fmt_bytes(r['bw_peak_bytes']):>10} "
                  f"{_fmt_bytes(r['estimated_peak']):>14}")

        print(f"\n  {'配置':<20} {'参数':>10} {'梯度':>10} "
              f"{'优化器':>10} {'峰值阶段':>14}")
        print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10} {'─'*14}")

        for tag, r in self._results.items():
            phases = {
                "FW": r["fw_phase_peak"],
                "BW": r["bw_phase_peak"],
                "Opt": r["opt_phase_peak"],
            }
            bottleneck = max(phases, key=phases.get)
            print(f"  {tag:<20} "
                  f"{_fmt_bytes(r['param_bytes']):>10} "
                  f"{_fmt_bytes(r['grad_bytes']):>10} "
                  f"{_fmt_bytes(r['optimizer_bytes']):>10} "
                  f"{'瓶颈: ' + bottleneck:>14}")

        tags = list(self._results.keys())
        if len(tags) >= 2:
            print(f"  {'─'*66}")
            baseline = self._results[tags[0]]
            for cmp_tag in tags[1:]:
                cmp = self._results[cmp_tag]
                delta = baseline["estimated_peak"] - cmp["estimated_peak"]
                direction = "节省" if delta >= 0 else "增加"
                pct = (delta / baseline["estimated_peak"] * 100
                       if baseline["estimated_peak"] > 0 else 0.0)
                print(f"  [{tags[0]} → {cmp_tag}]")
                print(f"    估算峰值{direction}: {_fmt_bytes(abs(delta))} ({pct:+.1f}%)")

                sa_delta = (baseline["saved_act_bytes"] - cmp["saved_act_bytes"])
                if sa_delta != 0:
                    sa_dir = "减少" if sa_delta > 0 else "增加"
                    print(f"    保存激活{sa_dir}: {_fmt_bytes(abs(sa_delta))} "
                          f"({baseline['saved_act_count']} → {cmp['saved_act_count']} 个)")

        print(_BOLD)

    # ── 可视化 ──────────────────────────────────────────────────────────

    def plot_peak_comparison(
        self,
        save_dir: Optional[str] = None,
        show: bool = False,
    ) -> Optional[str]:
        """生成静态峰值显存估算对比图（双子图）。"""
        if not _MPL_AVAILABLE:
            logger.warning("[StaticEstimator] matplotlib 未安装，跳过图表生成。")
            return None
        if not self._results:
            return None

        tags = list(self._results.keys())
        n = len(tags)
        mb = 1 << 20

        use_cjk = _CJK_FONT is not None
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(7, n * 2.5), 9))

        colors_bar = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974']

        # ── 上图：估算总峰值对比 ────────────────────────────────────────────
        peaks = [self._results[t]["estimated_peak"] / mb for t in tags]
        bars = ax1.bar(range(n), peaks, color=[colors_bar[i % len(colors_bar)] for i in range(n)],
                       alpha=0.88, edgecolor='white', linewidth=0.8)

        max_peak = max(peaks) if peaks else 1
        for i, (bar, val) in enumerate(zip(bars, peaks)):
            ax1.text(bar.get_x() + bar.get_width() / 2, val + max_peak * 0.015,
                     f'{val:.1f} MB', ha='center', va='bottom', fontsize=9, fontweight='bold')

        if n >= 2:
            baseline_peak = self._results[tags[0]]["estimated_peak"]
            for i in range(1, n):
                cmp_peak = self._results[tags[i]]["estimated_peak"]
                delta = baseline_peak - cmp_peak
                pct = delta / baseline_peak * 100 if baseline_peak > 0 else 0
                if abs(pct) > 0.1:
                    label = f'{"↓" if delta > 0 else "↑"}{abs(pct):.1f}%'
                    color = 'green' if delta > 0 else 'red'
                    ax1.text(i, peaks[i] / 2, label, ha='center', va='center',
                             fontsize=11, fontweight='bold', color=color)

        ax1.set_xticks(range(n))
        ax1.set_xticklabels(tags, fontsize=10)
        ax1.set_ylabel('Memory (MB)', fontsize=10)
        title1 = '静态峰值显存估算对比' if use_cjk else 'Static Peak Memory Estimation'
        ax1.set_title(title1, fontsize=12, fontweight='bold')
        ax1.grid(axis='y', linestyle=':', alpha=0.4)
        ax1.set_xlim(-0.6, n - 0.4)

        # ── 下图：内存分解堆叠柱 ────────────────────────────────────────────
        comp_keys = [
            ("param_bytes",    '#5975A4', '参数' if use_cjk else 'Params'),
            ("fw_peak_bytes",  '#5F9E6E', 'FW 激活' if use_cjk else 'FW Activation'),
            ("saved_act_bytes",'#B55D60', '保存激活' if use_cjk else 'Saved Act.'),
            ("bw_peak_bytes",  '#857AAB', 'BW 激活' if use_cjk else 'BW Activation'),
            ("grad_bytes",     '#C5B047', '梯度' if use_cjk else 'Gradients'),
            ("optimizer_bytes",'#D98880', '优化器' if use_cjk else 'Optimizer'),
        ]

        bottoms = [0.0] * n
        for key, color, label in comp_keys:
            vals = [self._results[t][key] / mb for t in tags]
            ax2.bar(range(n), vals, bottom=bottoms, color=color,
                    alpha=0.85, label=label, edgecolor='white', linewidth=0.5)
            bottoms = [b + v for b, v in zip(bottoms, vals)]

        ax2.set_xticks(range(n))
        ax2.set_xticklabels(tags, fontsize=10)
        ax2.set_ylabel('Memory (MB)', fontsize=10)
        title2 = '内存分解' if use_cjk else 'Memory Breakdown'
        ax2.set_title(title2, fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=8, ncol=2)
        ax2.grid(axis='y', linestyle=':', alpha=0.4)
        ax2.set_xlim(-0.6, n - 0.4)

        plt.tight_layout()

        path = None
        if save_dir:
            path = os.path.join(save_dir, 'static_estimation.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            logger.info("[StaticEstimator] 估算图表已保存至: %s", path)
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
        """将静态估算结果保存为 JSON，并可选生成对比图。"""
        from ..utils.save_ir import _default_ir_dir

        out_dir = _default_ir_dir(
            model_name or os.getenv("MODEL_NAME", "default_model"),
            subfolder=subfolder,
        )
        path = os.path.join(out_dir, "static_estimation.json")

        payload: Dict = {}

        if self._param_memory:
            payload["parameter_memory"] = self._param_memory

        payload["configurations"] = {}
        for tag, r in self._results.items():
            entry = {k: v for k, v in r.items()
                     if not k.endswith("_timeline") and k != "node_sizes"}
            payload["configurations"][tag] = entry

        tags = list(self._results.keys())
        if len(tags) >= 2:
            baseline = self._results[tags[0]]
            comparisons = []
            for cmp_tag in tags[1:]:
                cmp = self._results[cmp_tag]
                delta = baseline["estimated_peak"] - cmp["estimated_peak"]
                comparisons.append({
                    "baseline": tags[0],
                    "compared": cmp_tag,
                    "peak_delta_bytes": int(delta),
                    "peak_delta_pct": round(
                        delta / baseline["estimated_peak"] * 100, 2
                    ) if baseline["estimated_peak"] > 0 else 0.0,
                })
            payload["comparisons"] = comparisons

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        logger.info("[StaticEstimator] 静态估算报告已保存至: %s", path)

        if save_plots:
            self.plot_peak_comparison(save_dir=out_dir)

        return path


# 向后兼容别名
StaticMemoryEstimator = StaticEstimator
