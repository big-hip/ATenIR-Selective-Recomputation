"""
memory_analysis.py

提供两类内存分析能力：
  1. 静态估算 — 通过比较 Pass 前后的 FX 图，从 meta['val'] 推导激活内存节省量，
               无需实际运行，适合在训练启动前快速验证 Pass 效果。
  2. 运行时测量 — 对训练步骤进行峰值显存与耗时采样，量化重计算带来的实际
               显存收益和时间开销，支持 CUDA 和 CPU（CPU 端借助 psutil）。

典型用法::

    from aten_recompute.utils.memory_analysis import MemoryAnalyzer

    analyzer = MemoryAnalyzer(device='cuda')

    # 1. 静态估算（Pass 前后 FX 图对比）
    analyzer.estimate(
        fw_before=graph_capture.FW_gm, bw_before=graph_capture.BW_gm,
        fw_after=fw_gm_opt,            bw_after=bw_gm_opt,
    )

    # 2. 模型参数常驻显存估算（可选，作为对比参考）
    analyzer.estimate_parameter_memory(model)

    # 3. 运行时对比（传入封装好的训练步骤可调用对象）
    analyzer.profile_step("baseline",   fn=baseline_step_fn)
    analyzer.profile_step("recomputed", fn=recomputed_step_fn)
    analyzer.report()

    # 4. 保存报告 JSON 到 IR_artifacts
    analyzer.save_report(model_name="Transformer")
"""

import json
import os
import time
from typing import Callable, Dict, List, Optional

import torch
import torch.fx as fx
import torch.nn as nn

from .logger import get_logger
from .graph_utils import get_output_node, get_saved_activations
logger = get_logger(__name__)

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Agg')          # 非交互后端，服务器/无 GUI 环境安全
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    # 检测系统中是否存在可用的 CJK 字体；
    # 若无则不设置，避免 UserWarning: Glyph missing from font(s) DejaVu Sans
    def _find_cjk_font():
        from matplotlib import font_manager
        for name in (
            'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Noto Sans SC',
            'SimHei', 'Microsoft YaHei', 'AR PL UMing CN',
        ):
            try:
                path = font_manager.findfont(name, fallback_to_default=False)
                # findfont 找不到时返回默认字体路径；通过路径中是否含字体名关键词判断
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
    else:
        # 无 CJK 字体：保持 matplotlib 默认字体，图表标签使用英文
        logger.debug("[MemoryAnalyzer] 未检测到 CJK 字体，图表将使用英文标签。")

    _MPL_AVAILABLE = True
except ImportError:
    _CJK_FONT = None
    _MPL_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# 内部工具
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_bytes(n) -> str:
    """将字节数格式化为易读的 KB / MB / GB 字符串。
    兼容 SymInt / SymFloat：先用 float() 强制具化，若失败则返回占位符。
    """
    try:
        n = float(n)
    except (TypeError, RuntimeError):
        return "? B (symbolic)"
    if n == 0:
        return "0 B"
    for unit, threshold in [("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)]:
        if abs(n) >= threshold:
            return f"{n / threshold:.2f} {unit}"
    return f"{int(n)} B"


def _tensor_bytes(val) -> int:
    """从 FakeTensor / 真实 Tensor 估算字节数；非 Tensor 或符号形状无法具化时返回 0。

    dynamic=True 追踪时 numel() 返回 SymInt，但 SymInt 携带样本的具体值，
    int() 可以安全具化。若形状纯符号（无法具化）则捕获异常跳过。
    """
    if val is None or not isinstance(val, torch.Tensor):
        return 0
    try:
        numel = int(val.numel())   # 具化 SymInt → plain int
    except (TypeError, RuntimeError):
        return 0
    return numel * val.element_size()


def _saved_activation_bytes(
    fw_gm: fx.GraphModule,
    bw_gm: fx.GraphModule,
) -> Dict:
    """
    统计 FW→BW 边界上保存的张量，并将其分类为：
      - activations : FW 中间计算结果（op != 'placeholder'），是重计算真正要消除的对象
      - primals     : FW placeholder 节点直接透传给 BW（模型参数），本就长驻显存，
                      重计算无法"节省"这部分，但添加新的 primal 会让此类增加

    Returns:
        {
            'activation_bytes':  int,       中间激活的估算总字节数（重计算的真实节省来源）
            'primal_bytes':      int,       参数透传的估算总字节数（参考用）
            'total_bytes':       int,       两者之和
            'num_activations':   int,       中间激活数量
            'num_primals':       int,       参数透传数量
            'skipped':           int,       meta 缺失 / 非张量，跳过的节点数
            'activation_details': list,    中间激活的详情列表
            'primal_details':    list,      参数透传的详情列表
        }
    """
    # 构建 FW 图的 placeholder 名称集，用于判断一个 FW output 节点是否为 primal
    fw_placeholder_names = {
        n.name for n in fw_gm.graph.nodes if n.op == 'placeholder'
    }

    bw_ph_names = {
        n.name
        for n in bw_gm.graph.nodes
        if n.op == 'placeholder' and not n.name.startswith('tangents_')
    }

    output_node = get_output_node(fw_gm)   # 使用 graph_utils 工具
    if output_node is None:
        return {
            'activation_bytes': 0, 'primal_bytes': 0, 'total_bytes': 0,
            'num_activations': 0, 'num_primals': 0, 'skipped': 0,
            'activation_details': [], 'primal_details': [],
        }

    act_details:    List[Dict] = []
    primal_details: List[Dict] = []
    act_bytes    = 0
    primal_bytes = 0
    skipped      = 0

    for node in output_node.all_input_nodes:
        if node.name not in bw_ph_names:
            continue
        val = node.meta.get('val')
        if val is None or not isinstance(val, torch.Tensor):
            skipped += 1
            logger.debug(
                "[MemoryAnalyzer] 节点 '%s' meta['val'] 缺失或非 Tensor，跳过。",
                node.name,
            )
            continue

        nb = _tensor_bytes(val)
        entry = {
            'name':  node.name,
            'shape': [int(d) for d in val.shape],   # SymInt → plain int，保证 JSON 可序列化
            'dtype': str(val.dtype),
            'bytes': nb,
        }

        # 判断是 primal（参数透传）还是中间激活
        if node.name in fw_placeholder_names:
            primal_bytes += nb
            primal_details.append(entry)
        else:
            act_bytes += nb
            act_details.append(entry)

    if skipped:
        logger.debug(
            "[MemoryAnalyzer] %d 个节点（标量 / SymInt）无法估算，已跳过。",
            skipped,
        )

    return {
        'activation_bytes':  act_bytes,
        'primal_bytes':      primal_bytes,
        'total_bytes':       act_bytes + primal_bytes,
        'num_activations':   len(act_details),
        'num_primals':       len(primal_details),
        'skipped':           skipped,
        'activation_details': act_details,
        'primal_details':    primal_details,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 主类
# ─────────────────────────────────────────────────────────────────────────────

class MemoryAnalyzer:
    """
    重计算效果分析器。

    静态估算
    --------
    通过比较 Pass 前后的 FX 图，从 meta['val'] 推导激活内存节省量。
    无需真正运行，可在训练循环启动前调用。

    运行时测量
    ----------
    接受一个封装好训练步骤的可调用对象 fn()，预热若干次后进行多步采样，
    记录每步峰值显存（CUDA）和耗时，支持多组配置对比。
    CPU 端若安装了 psutil 则报告 RSS 峰值，否则显示 N/A。
    """

    def __init__(self, device: str = 'cuda'):
        self.device = device
        self._static_result: Optional[Dict] = None
        self._static_reported: bool = False        # Fix 1: 防止 report() 重复打印静态报告
        self._param_memory: Optional[Dict] = None  # Fix 7: 模型参数常驻显存估算结果
        self._runtime_results: Dict[str, Dict] = {}

    # ── 静态估算 ──────────────────────────────────────────────────────────────

    def estimate(
        self,
        fw_before: fx.GraphModule,
        bw_before: fx.GraphModule,
        fw_after: fx.GraphModule,
        bw_after: fx.GraphModule,
    ) -> Dict:
        """
        比较 Pass 前后的激活内存，打印报告并返回结果字典。

        Parameters
        ----------
        fw_before / bw_before : 重计算 Pass 执行前的 FW/BW 图
        fw_after  / bw_after  : 重计算 Pass 执行后的 FW/BW 图

        Returns
        -------
        {
            'before':       _saved_activation_bytes 的结果
            'after':        _saved_activation_bytes 的结果
            'saved_bytes':  节省的字节数（before - after）
            'saved_ratio':  节省比例
        }
        """
        before = _saved_activation_bytes(fw_before, bw_before)
        after  = _saved_activation_bytes(fw_after,  bw_after)

        # 真正有意义的节省量 = 中间激活内存的减少（参数透传本就长驻显存，不算节省）
        saved = before['activation_bytes'] - after['activation_bytes']
        ratio = saved / before['activation_bytes'] if before['activation_bytes'] > 0 else 0.0

        self._static_result = {
            'before': before,
            'after': after,
            'saved_bytes': saved,
            'saved_ratio': ratio,
        }
        self._static_reported = True  # Fix 1: 标记已打印，report() 不再重复打印
        self._print_static_report(self._static_result)
        return self._static_result

    def _print_static_report(self, r: Dict) -> None:
        line = "─" * 64
        before, after = r['before'], r['after']

        # 找出被消除/新增的中间激活
        after_act_names  = {d['name'] for d in after['activation_details']}
        eliminated = [d for d in before['activation_details'] if d['name'] not in after_act_names]
        remaining  = after['activation_details']

        # 参数透传变化
        before_primal_names = {d['name'] for d in before['primal_details']}
        added_primals = [d for d in after['primal_details'] if d['name'] not in before_primal_names]

        print(f"\n{line}")
        print("  [静态激活内存估算]  基于 FX 图 meta['val'] 推导，无需实际运行")
        print(f"  注：DOT 文件展示所有计算节点，此处仅统计跨越 FW→BW 边界的节点")
        print(line)

        # ── 汇总表 ──
        hdr = f"  {'配置':<8} {'中间激活':>8} {'激活内存':>14}  {'参数透传':>8} {'参数内存':>14}"
        print(hdr)
        print(f"  {'─'*8} {'─'*8} {'─'*14}  {'─'*8} {'─'*14}")
        print(f"  {'Pass 前':<8} {before['num_activations']:>8} "
              f"{_fmt_bytes(before['activation_bytes']):>14}  "
              f"{before['num_primals']:>8} {_fmt_bytes(before['primal_bytes']):>14}")
        print(f"  {'Pass 后':<8} {after['num_activations']:>8} "
              f"{_fmt_bytes(after['activation_bytes']):>14}  "
              f"{after['num_primals']:>8} {_fmt_bytes(after['primal_bytes']):>14}")
        print(f"  {'─'*62}")

        if r['saved_bytes'] >= 0:
            print(f"  ✓ 节省中间激活内存: {_fmt_bytes(r['saved_bytes'])}  "
                  f"({r['saved_ratio']:.1%})")
        else:
            print(f"  ✗ 中间激活内存增加: {_fmt_bytes(-r['saved_bytes'])}  "
                  f"（Pass 后保存了更多中间激活，请检查策略）")

        if added_primals:
            print(f"  ℹ Pass 后新增 {len(added_primals)} 个参数透传"
                  f"（重计算子图需要这些参数，它们本就在显存中）")

        if before['skipped'] or after['skipped']:
            print(f"  ℹ Pass 前/后各有 {before['skipped']}/{after['skipped']} 个"
                  f"标量节点无法估算，已跳过。")

        # ── 已消除的激活 ──
        if eliminated:
            print(f"\n  已重计算（从 FW→BW 传递链移除）的中间激活  [{len(eliminated)} 个]:")
            print(f"  {'名称':<42} {'形状':<22} {'dtype':<14} {'字节'}")
            print(f"  {'─'*42} {'─'*22} {'─'*14} {'─'*10}")
            for d in sorted(eliminated, key=lambda x: -x['bytes']):
                print(f"  {d['name']:<42} {str(d['shape']):<22} {d['dtype']:<14} "
                      f"{_fmt_bytes(d['bytes'])}")

        # ── 仍需保存的激活 ──
        if remaining:
            print(f"\n  仍需保存的中间激活  [{len(remaining)} 个]:")
            for d in sorted(remaining, key=lambda x: -x['bytes'])[:5]:
                print(f"    {d['name']:<42} {_fmt_bytes(d['bytes'])}")
            if len(remaining) > 5:
                print(f"    ... 共 {len(remaining)} 个，仅显示前 5")

        print(line + "\n")

    # ── 模型参数常驻显存估算（Fix 7）────────────────────────────────────────

    def estimate_parameter_memory(self, model: nn.Module) -> Dict:
        """
        统计模型所有参数与缓冲区的常驻显存，作为激活内存的参考基线。

        参数和缓冲区在整个训练过程中始终占据显存，重计算无法改变这部分。
        结果会缓存到 self._param_memory，供 save_report() 写入 JSON。
        调用后直接打印报告并返回结果字典。

        Returns
        -------
        {
            'param_bytes':    int,   所有可训练参数的字节数
            'buffer_bytes':   int,   所有缓冲区的字节数
            'total_bytes':    int,   两者之和
            'num_params':     int,   参数张量数量
            'num_buffers':    int,   缓冲区张量数量
            'num_elements':   int,   总参数量（元素数）
        }
        """
        param_bytes  = 0
        buffer_bytes = 0
        num_params   = 0
        num_buffers  = 0
        num_elements = 0

        for p in model.parameters():
            param_bytes  += p.numel() * p.element_size()
            num_elements += p.numel()
            num_params   += 1

        for b in model.buffers():
            buffer_bytes += b.numel() * b.element_size()
            num_buffers  += 1

        result = {
            'param_bytes':  param_bytes,
            'buffer_bytes': buffer_bytes,
            'total_bytes':  param_bytes + buffer_bytes,
            'num_params':   num_params,
            'num_buffers':  num_buffers,
            'num_elements': num_elements,
        }
        self._param_memory = result  # 缓存供 save_report() 使用

        line = "─" * 64
        print(f"\n{line}")
        print("  [模型参数常驻显存]  参数与缓冲区在训练全程占据，重计算无法节省")
        print(line)
        print(f"  可训练参数: {num_params:>6} 个张量  {num_elements:>12,} 元素  "
              f"{_fmt_bytes(param_bytes):>12}")
        print(f"  缓冲区:     {num_buffers:>6} 个张量                    "
              f"{_fmt_bytes(buffer_bytes):>12}")
        print(f"  合计:                                              "
              f"{_fmt_bytes(param_bytes + buffer_bytes):>12}")
        print(line + "\n")

        return result

    # ── 运行时测量 ────────────────────────────────────────────────────────────

    def profile_step(
        self,
        tag: str,
        fn: Callable,
        warmup: int = 2,
        steps: int = 5,
    ) -> Dict:
        """
        运行 fn() 若干次，统计每步峰值显存与耗时。

        Parameters
        ----------
        tag    : 标识符，用于报告中区分配置（如 "baseline" / "recomputed"）。
        fn     : 无参可调用对象，封装一个完整训练步骤（forward + backward + optimizer.step）。
        warmup : 预热次数，不纳入统计（让 CUDA graph / JIT 编译稳定）。
        steps  : 正式采样次数。

        Returns
        -------
        {
            'tag': str,
            'device': str,
            'peak_memory_bytes': list[int],   每步峰值显存（CUDA）或 RSS 字节（CPU + psutil）
            'elapsed_ms': list[float],         每步耗时（ms）
            'avg_peak_bytes': float,           -1.0 表示无法测量（CPU 且无 psutil）
            'max_peak_bytes': float,
            'min_peak_bytes': float,
            'avg_elapsed_ms': float,
            'max_elapsed_ms': float,
            'min_elapsed_ms': float,
        }
        """
        is_cuda = self.device.startswith('cuda') and torch.cuda.is_available()

        # Fix 2: CPU 端借助 psutil 读取 RSS（驻留集大小）
        proc = None
        if not is_cuda and _PSUTIL_AVAILABLE:
            proc = psutil.Process(os.getpid())

        print(f"[MemoryAnalyzer] 预热 '{tag}' ({warmup} 步)...")
        for _ in range(warmup):
            fn()
        if is_cuda:
            torch.cuda.synchronize(self.device)

        print(f"[MemoryAnalyzer] 正式采样 '{tag}' ({steps} 步)...")
        peak_mem_list: List[int]   = []
        elapsed_list:  List[float] = []

        for i in range(steps):
            if is_cuda:
                torch.cuda.reset_peak_memory_stats(self.device)
                torch.cuda.synchronize(self.device)
            elif proc is not None:
                # 采样前先读一次 RSS 作为基准（实际峰值在 fn 执行中产生，此处取执行后值作近似）
                pass

            t0 = time.perf_counter()
            fn()
            if is_cuda:
                torch.cuda.synchronize(self.device)
            elapsed_list.append((time.perf_counter() - t0) * 1000)

            if is_cuda:
                peak_mem_list.append(torch.cuda.max_memory_allocated(self.device))
            elif proc is not None:
                peak_mem_list.append(proc.memory_info().rss)

            # Fix 3: 改用 % 延迟格式，避免 debug 级别下多余的字符串拼接
            peak_str = _fmt_bytes(peak_mem_list[-1]) if peak_mem_list else "N/A"
            logger.debug(
                "[MemoryAnalyzer] '%s' step %d/%d: peak=%s, time=%.1f ms",
                tag, i + 1, steps, peak_str, elapsed_list[-1],
            )

        # Fix 6: 计算 min / max
        if peak_mem_list:
            avg_peak = sum(peak_mem_list) / len(peak_mem_list)
            max_peak = float(max(peak_mem_list))
            min_peak = float(min(peak_mem_list))
        else:
            # CPU 且无 psutil：用 -1.0 表示无法测量
            avg_peak = -1.0
            max_peak = -1.0
            min_peak = -1.0

        avg_time = sum(elapsed_list) / len(elapsed_list)
        max_time = max(elapsed_list)
        min_time = min(elapsed_list)

        result = {
            'tag': tag,
            'device': self.device,
            'peak_memory_bytes': peak_mem_list,
            'elapsed_ms': elapsed_list,
            'avg_peak_bytes': avg_peak,
            'max_peak_bytes': max_peak,
            'min_peak_bytes': min_peak,
            'avg_elapsed_ms': avg_time,
            'max_elapsed_ms': max_time,
            'min_elapsed_ms': min_time,
        }
        self._runtime_results[tag] = result

        # Fix 2: CPU 无显存数据时显示 N/A
        if avg_peak >= 0:
            peak_str = f"avg {_fmt_bytes(avg_peak)} [min {_fmt_bytes(min_peak)} / max {_fmt_bytes(max_peak)}]"
        else:
            peak_str = "N/A (CPU，请安装 psutil 以测量 RSS)" if not _PSUTIL_AVAILABLE else "N/A"

        print(
            f"[MemoryAnalyzer] '{tag}': {peak_str}, "
            f"avg time {avg_time:.1f} ms [min {min_time:.1f} / max {max_time:.1f}]\n"
        )
        return result

    # ── 综合报告 ──────────────────────────────────────────────────────────────

    def report(self) -> None:
        """打印静态估算结果（若有且未打印过）和所有运行时配置的对比报告。"""
        # Fix 1: 仅在 estimate() 未打印过时才打印静态报告
        if self._static_result and not self._static_reported:
            self._print_static_report(self._static_result)

        if not self._runtime_results:
            return

        line = "─" * 64
        print(f"\n{line}")
        print("  [运行时内存 & 耗时对比]")
        print(line)

        # Fix 6: 汇总表增加 min/max 列
        has_mem = any(r['avg_peak_bytes'] >= 0 for r in self._runtime_results.values())
        if has_mem:
            print(f"  {'配置':<20} {'平均峰值显存':>14} {'最小':>10} {'最大':>10} {'平均耗时':>12}")
            print(f"  {'─'*20} {'─'*14} {'─'*10} {'─'*10} {'─'*12}")
        else:
            print(f"  {'配置':<20} {'峰值显存':>14} {'平均耗时':>12}")
            print(f"  {'─'*20} {'─'*14} {'─'*12}")

        tags = list(self._runtime_results.keys())
        for tag, r in self._runtime_results.items():
            if has_mem and r['avg_peak_bytes'] >= 0:
                print(
                    f"  {tag:<20} {_fmt_bytes(r['avg_peak_bytes']):>14} "
                    f"{_fmt_bytes(r['min_peak_bytes']):>10} "
                    f"{_fmt_bytes(r['max_peak_bytes']):>10} "
                    f"{r['avg_elapsed_ms']:>10.1f} ms"
                )
            elif has_mem:
                print(f"  {tag:<20} {'N/A':>14} {'N/A':>10} {'N/A':>10} {r['avg_elapsed_ms']:>10.1f} ms")
            else:
                print(f"  {tag:<20} {'N/A':>14} {r['avg_elapsed_ms']:>10.1f} ms")

        # Fix 5 & 8: 对比区，支持多 tag（均与第一个 tag 作为基准对比）
        if len(tags) >= 2:
            print(f"  {'─'*62}")
            r_base = self._runtime_results[tags[0]]
            for cmp_tag in tags[1:]:
                r_cmp = self._runtime_results[cmp_tag]

                # 时间对比
                time_delta = r_cmp['avg_elapsed_ms'] - r_base['avg_elapsed_ms']
                time_pct   = (time_delta / r_base['avg_elapsed_ms'] * 100
                              if r_base['avg_elapsed_ms'] > 0 else 0.0)
                time_dir   = "slower↑" if time_delta > 0 else "faster↓"

                print(f"  [{tags[0]} → {cmp_tag}]")
                if has_mem and r_base['avg_peak_bytes'] >= 0 and r_cmp['avg_peak_bytes'] >= 0:
                    mem_delta = r_base['avg_peak_bytes'] - r_cmp['avg_peak_bytes']
                    mem_dir   = "saved" if mem_delta >= 0 else "increased"
                    print(f"    Peak memory {mem_dir}: {_fmt_bytes(abs(mem_delta))}")
                # Fix 5: 时间开销显示百分比
                print(f"    时间开销: {time_delta:+.1f} ms  ({time_pct:+.1f}%)  {time_dir}")

        print(line + "\n")

    # ── 可视化 ────────────────────────────────────────────────────────────────

    def plot_static_memory(
        self,
        save_dir: Optional[str] = None,
        show: bool = False,
    ) -> Optional[str]:
        """
        生成静态激活内存对比条形图（Pass 前 vs Pass 后）。

        Parameters
        ----------
        save_dir : 图片保存目录；为 None 时不保存文件（配合 show=True 使用）。
        show     : 是否调用 plt.show() 弹出窗口（服务器环境建议保持 False）。

        Returns
        -------
        保存的图片路径，未保存时返回 None。
        """
        if not _MPL_AVAILABLE:
            logger.warning("[MemoryAnalyzer] matplotlib 未安装，跳过静态内存图。")
            return None
        if not self._static_result:
            logger.warning("[MemoryAnalyzer] 尚未调用 estimate()，无静态数据可绘图。")
            return None

        r = self._static_result
        before, after = r['before'], r['after']

        mb = 1 << 20
        labels    = ['Before Pass', 'After Pass']
        act_mb    = [before['activation_bytes'] / mb, after['activation_bytes'] / mb]
        primal_mb = [before['primal_bytes'] / mb,     after['primal_bytes'] / mb]

        fig, ax = plt.subplots(figsize=(7, 5))
        x     = [0, 1]
        width = 0.35

        bars_act    = ax.bar([p - width / 2 for p in x], act_mb,    width,
                             label='Activation (recompute target)', color='steelblue', alpha=0.85)
        bars_primal = ax.bar([p + width / 2 for p in x], primal_mb, width,
                             label='Primal (always resident)',      color='coral',      alpha=0.85)

        # 在每根柱子上标注数值
        all_bars = list(bars_act) + list(bars_primal)
        max_h    = max(b.get_height() for b in all_bars) or 1.0
        for bar in all_bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + max_h * 0.01,
                        f'{h:.2f}', ha='center', va='bottom', fontsize=8)

        # 节省量标注
        saved_mb = r['saved_bytes'] / mb
        if saved_mb > 0:
            ax.annotate(
                f'Saved: {saved_mb:.2f} MB ({r["saved_ratio"]:.1%})',
                xy=(0.5, max_h * 0.92), xycoords=('data', 'data'),
                ha='center', color='green', fontsize=10, fontweight='bold',
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Memory (MB)')
        ax.set_title("Static Activation Memory: FW->BW Boundary\n(estimated from meta['val'])")
        ax.legend(loc='upper right')
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f MB'))
        ax.set_xlim(-0.6, 1.6)
        plt.tight_layout()

        path = None
        if save_dir:
            path = os.path.join(save_dir, 'static_memory.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            logger.info("[MemoryAnalyzer] 静态内存图已保存至: %s", path)
        if show:
            plt.show()
        plt.close(fig)
        return path

    def plot_runtime_memory(
        self,
        save_dir: Optional[str] = None,
        show: bool = False,
    ) -> Optional[str]:
        """
        生成运行时峰值显存折线图（上）与步骤耗时折线图（下）。

        Parameters
        ----------
        save_dir : 图片保存目录；为 None 时不保存文件。
        show     : 是否调用 plt.show()。

        Returns
        -------
        保存的图片路径，未保存时返回 None。
        """
        if not _MPL_AVAILABLE:
            logger.warning("[MemoryAnalyzer] matplotlib 未安装，跳过运行时内存图。")
            return None
        if not self._runtime_results:
            logger.warning("[MemoryAnalyzer] 尚未调用 profile_step()，无运行时数据可绘图。")
            return None

        has_mem = any(r['avg_peak_bytes'] >= 0 for r in self._runtime_results.values())
        n_rows  = 2 if has_mem else 1
        fig, axes = plt.subplots(n_rows, 1, figsize=(8, 4 * n_rows), sharex=False)
        if n_rows == 1:
            axes = [axes]

        colors = ['steelblue', 'coral', 'seagreen', 'orchid', 'goldenrod']
        mb = 1 << 20

        # ── 上图：峰值显存折线 ─────────────────────────────────────────────────
        if has_mem:
            ax_mem = axes[0]
            for idx, (tag, r) in enumerate(self._runtime_results.items()):
                if r['avg_peak_bytes'] < 0:
                    continue
                steps  = list(range(1, len(r['peak_memory_bytes']) + 1))
                values = [v / mb for v in r['peak_memory_bytes']]
                color  = colors[idx % len(colors)]
                ax_mem.plot(steps, values, marker='o', label=tag, color=color, linewidth=1.8)
                ax_mem.axhline(r['avg_peak_bytes'] / mb, linestyle='--',
                               color=color, alpha=0.45, linewidth=1)
            ax_mem.set_xlabel('Step')
            ax_mem.set_ylabel('Peak Memory (MB)')
            ax_mem.set_title('Runtime Peak Memory (per step)')
            ax_mem.legend()
            ax_mem.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f MB'))
            ax_mem.grid(axis='y', linestyle=':', alpha=0.5)

        # ── 下图（或唯一图）：步骤耗时折线 ────────────────────────────────────
        ax_time = axes[-1]
        for idx, (tag, r) in enumerate(self._runtime_results.items()):
            steps  = list(range(1, len(r['elapsed_ms']) + 1))
            color  = colors[idx % len(colors)]
            ax_time.plot(steps, r['elapsed_ms'], marker='s', label=tag,
                         color=color, linewidth=1.8)
            ax_time.axhline(r['avg_elapsed_ms'], linestyle='--',
                            color=color, alpha=0.45, linewidth=1)
        ax_time.set_xlabel('Step')
        ax_time.set_ylabel('Elapsed (ms)')
        ax_time.set_title('Runtime Step Elapsed Time (per step)')
        ax_time.legend()
        ax_time.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f ms'))
        ax_time.grid(axis='y', linestyle=':', alpha=0.5)

        plt.tight_layout()

        path = None
        if save_dir:
            path = os.path.join(save_dir, 'runtime_memory.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            logger.info("[MemoryAnalyzer] 运行时内存图已保存至: %s", path)
        if show:
            plt.show()
        plt.close(fig)
        return path

    # ── 持久化 ────────────────────────────────────────────────────────────────

    def save_report(
        self,
        model_name: Optional[str] = None,
        subfolder: str = "memory",
        save_plots: bool = True,
    ) -> str:
        """
        将分析结果以 JSON 格式保存到 IR_artifacts 目录，并可选生成内存图。

        Parameters
        ----------
        save_plots : 是否同时生成并保存 static_memory.png / runtime_memory.png（默认 True）。

        Returns
        -------
        JSON 报告的保存路径字符串。
        """
        from .save_ir import _default_ir_dir  # 延迟导入，避免循环依赖

        out_dir = _default_ir_dir(
            model_name or os.getenv("MODEL_NAME", "default_model"),
            subfolder=subfolder,
        )
        path = os.path.join(out_dir, "memory_report.json")

        payload: Dict = {}

        # Fix 4: 静态报告增加 remaining_activations 和 primal_details
        if self._static_result:
            r = self._static_result
            after_act_names = {d['name'] for d in r['after']['activation_details']}
            payload['static'] = {
                'before_activation_bytes': int(r['before']['activation_bytes']),
                'before_primal_bytes':     int(r['before']['primal_bytes']),
                'before_num_activations':  r['before']['num_activations'],
                'before_num_primals':      r['before']['num_primals'],
                'before_primal_details':   r['before']['primal_details'],
                'after_activation_bytes':  int(r['after']['activation_bytes']),
                'after_primal_bytes':      int(r['after']['primal_bytes']),
                'after_num_activations':   r['after']['num_activations'],
                'after_num_primals':       r['after']['num_primals'],
                'after_primal_details':    r['after']['primal_details'],
                'saved_bytes':             int(r['saved_bytes']),
                'saved_ratio':             float(r['saved_ratio']),
                'eliminated_activations': [
                    d for d in r['before']['activation_details']
                    if d['name'] not in after_act_names
                ],
                'remaining_activations':   r['after']['activation_details'],  # Fix 4
            }

        if self._param_memory:
            payload['parameter_memory'] = {
                k: int(v) if isinstance(v, int) else v
                for k, v in self._param_memory.items()
            }

        if self._runtime_results:
            payload['runtime'] = {
                tag: {
                    'avg_peak_bytes':   r['avg_peak_bytes'],
                    'max_peak_bytes':   r['max_peak_bytes'],   # Fix 6
                    'min_peak_bytes':   r['min_peak_bytes'],   # Fix 6
                    'avg_elapsed_ms':   r['avg_elapsed_ms'],
                    'max_elapsed_ms':   r['max_elapsed_ms'],   # Fix 6
                    'min_elapsed_ms':   r['min_elapsed_ms'],   # Fix 6
                    'peak_memory_bytes': r['peak_memory_bytes'],
                    'elapsed_ms':        r['elapsed_ms'],
                    'device':            r['device'],
                }
                for tag, r in self._runtime_results.items()
            }
            # Fix 5 & 8: 写入与基准的对比数据
            tags = list(self._runtime_results.keys())
            if len(tags) >= 2:
                r_base = self._runtime_results[tags[0]]
                comparisons = []
                for cmp_tag in tags[1:]:
                    r_cmp = self._runtime_results[cmp_tag]
                    time_delta = r_cmp['avg_elapsed_ms'] - r_base['avg_elapsed_ms']
                    time_pct   = (time_delta / r_base['avg_elapsed_ms'] * 100
                                  if r_base['avg_elapsed_ms'] > 0 else 0.0)
                    entry: Dict = {
                        'baseline': tags[0],
                        'compared': cmp_tag,
                        'time_delta_ms': time_delta,
                        'time_delta_pct': time_pct,
                    }
                    if r_base['avg_peak_bytes'] >= 0 and r_cmp['avg_peak_bytes'] >= 0:
                        mem_delta = r_base['avg_peak_bytes'] - r_cmp['avg_peak_bytes']
                        entry['mem_saved_bytes'] = mem_delta
                    comparisons.append(entry)
                payload['runtime_comparisons'] = comparisons

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        logger.info("[MemoryAnalyzer] 内存报告已保存至: %s", path)

        if save_plots:
            self.plot_static_memory(save_dir=out_dir)
            self.plot_runtime_memory(save_dir=out_dir)

        return path
