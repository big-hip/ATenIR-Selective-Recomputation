"""
profiler.py

运行时 GPU 内存采样 + 报告生成。

提供两类内存分析能力：
  1. 静态估算 — 通过比较 Pass 前后的 FX 图，从 meta['val'] 推导激活内存节省量，
               无需实际运行，适合在训练启动前快速验证 Pass 效果。
  2. 运行时测量 — 对训练步骤进行峰值显存与耗时采样，量化重计算带来的实际
               显存收益和时间开销，支持 CUDA 和 CPU（CPU 端借助 psutil）。

典型用法::

    from aten_recompute.analysis import MemoryProfiler

    profiler = MemoryProfiler(device='cuda')

    # 1. 模型参数常驻显存估算（可选，作为对比参考）
    profiler.estimate_parameter_memory(model)

    # 2. 运行时对比（推荐：分阶段模式，自动分离 FW/BW/Opt 计时）
    profiler.profile_step("baseline",
                          forward_fn=lambda: criterion(model(x), y),
                          optimizer=opt)
    profiler.profile_step("recomputed",
                          forward_fn=lambda: criterion(recomp_model(x), y),
                          optimizer=recomp_opt)
    profiler.report()

    # 3. 保存报告 JSON 到 IR_artifacts
    profiler.save_report(model_name="Transformer")
"""

import json
import os
import statistics
import time
from typing import Callable, Dict, List, Optional

import torch
import torch.fx as fx
import torch.nn as nn

from ..utils.logger import get_logger
from ..utils.graph_utils import get_output_node, get_saved_activations
from ._activations import _fmt_bytes, _tensor_bytes, _saved_activation_bytes

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
        logger.debug("[MemoryProfiler] 未检测到 CJK 字体，图表将使用英文标签。")

    _MPL_AVAILABLE = True
except ImportError:
    _CJK_FONT = None
    _MPL_AVAILABLE = False


def _median(seq):
    """计算中位数；空序列返回 0.0。"""
    return statistics.median(seq) if seq else 0.0


def _iqr_mean(seq):
    """
    IQR 修剪均值：去掉 Q1 以下和 Q3 以上的异常值后取均值。

    与 torch.utils.benchmark.Timer 使用相同的统计策略，
    对 GPU 热降频 / DVFS 等引起的离群点具有鲁棒性。
    空序列或样本不足（< 4）时回退到普通中位数。
    """
    n = len(seq)
    if n < 4:
        return _median(seq)
    s = sorted(seq)
    q1 = s[n // 4]
    q3 = s[3 * n // 4]
    trimmed = [x for x in s if q1 <= x <= q3]
    return sum(trimmed) / len(trimmed) if trimmed else _median(seq)


class MemoryProfiler:
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
        self._static_reported: bool = False
        self._param_memory: Optional[Dict] = None
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
        self._static_reported = True
        self._print_static_report(self._static_result)
        return self._static_result

    def _print_static_report(self, r: Dict) -> None:
        line = "─" * 68
        before, after = r['before'], r['after']

        # 找出被消除/新增的中间激活
        after_act_names  = {d['name'] for d in after['activation_details']}
        eliminated = [d for d in before['activation_details'] if d['name'] not in after_act_names]
        remaining  = after['activation_details']

        # 参数透传变化
        before_primal_names = {d['name'] for d in before['primal_details']}
        added_primals = [d for d in after['primal_details'] if d['name'] not in before_primal_names]

        print(f"\n{line}")
        print("  静态激活内存估算（基于 FX 图 meta['val'] 推导，无需实际运行）")
        print(f"  注: 此处仅统计跨越 FW→BW 边界的节点")
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
        print(f"  {'─'*66}")

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

        print(line)

    def estimate_parameter_memory(self, model: nn.Module) -> Dict:
        """
        统计模型所有参数与缓冲区的常驻显存，作为激活内存的参考基线。

        参数和缓冲区在整个训练过程中始终占据显存，重计算无法改变这部分。
        结果会缓存到 self._param_memory，供 save_report() 写入 JSON。
        调用后直接打印报告并返回结果字典。
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

        line = "─" * 68
        print(f"\n{line}")
        print("  模型参数常驻显存（参数与缓冲区在训练全程占据，重计算无法节省）")
        print(line)
        print(f"  {'类型':<14} {'张量数':>8} {'元素数':>14} {'内存':>12}")
        print(f"  {'─'*14} {'─'*8} {'─'*14} {'─'*12}")
        print(f"  {'可训练参数':<14} {num_params:>8} {num_elements:>14,} {_fmt_bytes(param_bytes):>12}")
        print(f"  {'缓冲区':<14} {num_buffers:>8} {'':>14} {_fmt_bytes(buffer_bytes):>12}")
        print(f"  {'─'*14} {'─'*8} {'─'*14} {'─'*12}")
        print(f"  {'合计':<14} {num_params + num_buffers:>8} {num_elements:>14,} "
              f"{_fmt_bytes(param_bytes + buffer_bytes):>12}")
        print(line)

        return result

    # ── 运行时测量 ────────────────────────────────────────────────────────────

    def profile_step(
        self,
        tag: str,
        fn: Optional[Callable] = None,
        *,
        forward_fn: Optional[Callable] = None,
        optimizer: Optional["torch.optim.Optimizer"] = None,
        warmup: int = 5,
        steps: int = 10,
    ) -> Dict:
        """
        运行训练步骤若干次，统计每步峰值显存与耗时。

        支持两种模式：

        **传统模式** — 传入 ``fn``（向后兼容）::

            profile_step("baseline", fn=step_fn)

        **分阶段模式** — 传入 ``forward_fn`` + ``optimizer``（推荐）::

            profile_step("recomputed",
                         forward_fn=lambda: criterion(model(x), y),
                         optimizer=opt)

        分阶段模式下，profiler 会自动分别计时前向、反向、优化器三阶段，
        并单独记录首次 warmup 调用的编译耗时。

        Parameters
        ----------
        tag        : 标识符，用于报告中区分配置。
        fn         : 传统模式 — 无参可调用对象，封装完整训练步骤。
        forward_fn : 分阶段模式 — 返回 loss tensor 的可调用对象。
        optimizer  : 分阶段模式 — torch.optim.Optimizer 实例。
        warmup     : 预热次数（默认 5，让编译/autotuning 稳定）。
        steps      : 正式采样次数。
        """
        # ── 参数校验 ──
        is_phased = forward_fn is not None
        if fn is not None and is_phased:
            raise ValueError("fn 与 forward_fn 互斥，请只传其中一个。")
        if fn is None and not is_phased:
            raise ValueError("请传入 fn 或 forward_fn。")
        if is_phased and optimizer is None:
            raise ValueError("分阶段模式下必须传入 optimizer。")

        is_cuda = self.device.startswith('cuda') and torch.cuda.is_available()

        # CPU 端借助 psutil 读取 RSS（驻留集大小）
        proc = None
        if not is_cuda and _PSUTIL_AVAILABLE:
            proc = psutil.Process(os.getpid())

        def _sync():
            if is_cuda:
                torch.cuda.synchronize(self.device)

        # ── Warmup（首步计时 → compile_ms）──
        print(f"  [{tag}] 预热 {warmup} 步...")
        compile_ms = 0.0

        for w in range(warmup):
            _sync()
            if w == 0:
                t_compile = time.perf_counter()

            if is_phased:
                optimizer.zero_grad()
                loss = forward_fn()
                loss.backward()
                optimizer.step()
            else:
                fn()

            _sync()
            if w == 0:
                compile_ms = (time.perf_counter() - t_compile) * 1000

        # ── 采样 ──
        print(f"  [{tag}] 采样 {steps} 步...")
        peak_mem_list: List[int]   = []
        elapsed_list:  List[float] = []
        forward_list:  List[float] = []
        backward_list: List[float] = []
        optim_list:    List[float] = []

        for i in range(steps):
            if is_cuda:
                torch.cuda.reset_peak_memory_stats(self.device)

            if is_phased:
                # ── 分阶段计时 ──
                optimizer.zero_grad()

                if is_cuda:
                    # CUDA Event 计时：直接在 GPU 端测量，避免 CPU 调度抖动
                    ev0 = torch.cuda.Event(enable_timing=True)
                    ev1 = torch.cuda.Event(enable_timing=True)
                    ev2 = torch.cuda.Event(enable_timing=True)
                    ev3 = torch.cuda.Event(enable_timing=True)

                    ev0.record()
                    loss = forward_fn()
                    ev1.record()
                    loss.backward()
                    ev2.record()
                    optimizer.step()
                    ev3.record()
                    torch.cuda.synchronize()

                    fw_ms  = ev0.elapsed_time(ev1)
                    bw_ms  = ev1.elapsed_time(ev2)
                    opt_ms = ev2.elapsed_time(ev3)
                else:
                    # CPU 回退：perf_counter
                    t_fw = time.perf_counter()
                    loss = forward_fn()
                    fw_ms = (time.perf_counter() - t_fw) * 1000

                    t_bw = time.perf_counter()
                    loss.backward()
                    bw_ms = (time.perf_counter() - t_bw) * 1000

                    t_opt = time.perf_counter()
                    optimizer.step()
                    opt_ms = (time.perf_counter() - t_opt) * 1000

                forward_list.append(fw_ms)
                backward_list.append(bw_ms)
                optim_list.append(opt_ms)
                elapsed_list.append(fw_ms + bw_ms + opt_ms)
            else:
                # ── 传统整体计时 ──
                if is_cuda:
                    ev_start = torch.cuda.Event(enable_timing=True)
                    ev_end   = torch.cuda.Event(enable_timing=True)
                    ev_start.record()
                    fn()
                    ev_end.record()
                    torch.cuda.synchronize()
                    elapsed_list.append(ev_start.elapsed_time(ev_end))
                else:
                    t0 = time.perf_counter()
                    fn()
                    elapsed_list.append((time.perf_counter() - t0) * 1000)

            # 峰值显存采集
            if is_cuda:
                peak_mem_list.append(torch.cuda.max_memory_allocated(self.device))
            elif proc is not None:
                peak_mem_list.append(proc.memory_info().rss)

            peak_str = _fmt_bytes(peak_mem_list[-1]) if peak_mem_list else "N/A"
            logger.debug(
                "[MemoryProfiler] '%s' step %d/%d: peak=%s, time=%.1f ms",
                tag, i + 1, steps, peak_str, elapsed_list[-1],
            )

        # ── 统计汇总 ──
        if peak_mem_list:
            avg_peak = sum(peak_mem_list) / len(peak_mem_list)
            max_peak = float(max(peak_mem_list))
            min_peak = float(min(peak_mem_list))
        else:
            avg_peak = -1.0
            max_peak = -1.0
            min_peak = -1.0

        avg_time = sum(elapsed_list) / len(elapsed_list)
        max_time = max(elapsed_list)
        min_time = min(elapsed_list)
        iqr_time = _iqr_mean(elapsed_list)

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
            'median_elapsed_ms': _median(elapsed_list),
            'iqr_elapsed_ms': iqr_time,
            'compile_ms': compile_ms,
        }

        # ── 分阶段字段（仅 phased 模式）──
        if is_phased:
            result['forward_ms']  = forward_list
            result['backward_ms'] = backward_list
            result['optimizer_ms'] = optim_list
            result['avg_forward_ms']  = sum(forward_list) / len(forward_list)
            result['avg_backward_ms'] = sum(backward_list) / len(backward_list)
            result['avg_optimizer_ms'] = sum(optim_list) / len(optim_list)
            result['median_forward_ms']  = _median(forward_list)
            result['median_backward_ms'] = _median(backward_list)
            result['median_optimizer_ms'] = _median(optim_list)
            result['iqr_forward_ms']  = _iqr_mean(forward_list)
            result['iqr_backward_ms'] = _iqr_mean(backward_list)
            result['iqr_optimizer_ms'] = _iqr_mean(optim_list)

        self._runtime_results[tag] = result

        # ── 打印摘要 ──
        if avg_peak >= 0:
            peak_str = f"平均 {_fmt_bytes(avg_peak)} [最小 {_fmt_bytes(min_peak)} / 最大 {_fmt_bytes(max_peak)}]"
        else:
            peak_str = "N/A（CPU 环境，请安装 psutil 以测量 RSS）" if not _PSUTIL_AVAILABLE else "N/A"

        summary = f"  [{tag}] 峰值显存: {peak_str}, IQR步时: {iqr_time:.1f} ms"
        if is_phased:
            summary += (
                f" (FW {result['iqr_forward_ms']:.1f}"
                f" / BW {result['iqr_backward_ms']:.1f}"
                f" / Opt {result['iqr_optimizer_ms']:.1f})"
            )
        if compile_ms > 0:
            summary += f", 编译: {compile_ms:.0f} ms"
        print(summary)

        return result

    # ── 综合报告 ──────────────────────────────────────────────────────────────

    def report(self) -> None:
        """打印静态估算结果（若有且未打印过）和所有运行时配置的对比报告。"""
        # 仅在 estimate() 未打印过时才打印静态报告
        if self._static_result and not self._static_reported:
            self._print_static_report(self._static_result)

        if not self._runtime_results:
            return

        line = "─" * 100
        print(f"\n{line}")
        print("  运行时内存 & 耗时对比")
        print(line)

        has_mem   = any(r['avg_peak_bytes'] >= 0 for r in self._runtime_results.values())
        has_phase = any('forward_ms' in r for r in self._runtime_results.values())

        # ── 表头 ──
        hdr  = f"  {'配置':<22}"
        sep  = f"  {'─'*22}"
        if has_mem:
            hdr += f" {'峰值显存':>12}"
            sep += f" {'─'*12}"
        hdr += f" {'编译耗时':>10} {'IQR步时':>10}"
        sep += f" {'─'*10} {'─'*10}"
        if has_phase:
            hdr += f" {'前向':>9} {'反向':>9} {'优化器':>9}"
            sep += f" {'─'*9} {'─'*9} {'─'*9}"
        print(hdr)
        print(sep)

        # ── 数据行 ──
        tags = list(self._runtime_results.keys())
        for tag, r in self._runtime_results.items():
            row = f"  {tag:<22}"
            if has_mem:
                if r['avg_peak_bytes'] >= 0:
                    row += f" {_fmt_bytes(r['avg_peak_bytes']):>12}"
                else:
                    row += f" {'N/A':>12}"
            compile_str = f"{r['compile_ms']:.0f} ms" if r.get('compile_ms', 0) > 0 else "N/A"
            row += f" {compile_str:>10} {r['iqr_elapsed_ms']:>8.1f} ms"
            if has_phase:
                if 'forward_ms' in r:
                    row += (f" {r['iqr_forward_ms']:>7.1f} ms"
                            f" {r['iqr_backward_ms']:>7.1f} ms"
                            f" {r['iqr_optimizer_ms']:>7.1f} ms")
                else:
                    row += f" {'-':>9} {'-':>9} {'-':>9}"
            print(row)

        # ── 对比区 ──
        if len(tags) >= 2:
            print(f"  {'─'*98}")
            r_base = self._runtime_results[tags[0]]
            for cmp_tag in tags[1:]:
                r_cmp = self._runtime_results[cmp_tag]

                # 时间对比（基于 IQR 修剪均值）
                time_delta = r_cmp['iqr_elapsed_ms'] - r_base['iqr_elapsed_ms']
                time_pct   = (time_delta / r_base['iqr_elapsed_ms'] * 100
                              if r_base['iqr_elapsed_ms'] > 0 else 0.0)
                time_dir   = "更慢 ↑" if time_delta > 0 else "更快 ↓"

                print(f"  [{tags[0]} → {cmp_tag}]")

                # 显存对比
                mem_saved_bytes = 0.0
                if has_mem and r_base['avg_peak_bytes'] >= 0 and r_cmp['avg_peak_bytes'] >= 0:
                    mem_delta = r_base['avg_peak_bytes'] - r_cmp['avg_peak_bytes']
                    mem_saved_bytes = mem_delta
                    mem_dir = "节省" if mem_delta >= 0 else "增加"
                    print(f"    峰值显存{mem_dir}: {_fmt_bytes(abs(mem_delta))}")

                print(f"    耗时变化: {time_delta:+.1f} ms ({time_pct:+.1f}%) {time_dir}")

                # 反向阶段对比
                if 'forward_ms' in r_base and 'forward_ms' in r_cmp:
                    bw_delta = r_cmp['iqr_backward_ms'] - r_base['iqr_backward_ms']
                    if abs(bw_delta) > 0.1:
                        bw_dir = "更慢 ↑" if bw_delta > 0 else "更快 ↓"
                        print(f"    反向阶段: {bw_delta:+.1f} ms {bw_dir}")

                # 显存-时间权衡
                if mem_saved_bytes > 0 and time_delta > 0:
                    ratio = (mem_saved_bytes / (1 << 20)) / time_delta
                    print(f"    显存-时间权衡: 每多花 1ms 可节省 {ratio:.1f} MB 显存")
                elif mem_saved_bytes > 0 and time_delta <= 0:
                    print(f"    显存-时间权衡: 节省 {_fmt_bytes(mem_saved_bytes)} 且无时间开销")

        print(line)

    # ── 可视化 ────────────────────────────────────────────────────────────────

    def plot_static_memory(
        self,
        save_dir: Optional[str] = None,
        show: bool = False,
    ) -> Optional[str]:
        """生成静态激活内存对比条形图（Pass 前 vs Pass 后）。"""
        if not _MPL_AVAILABLE:
            logger.warning("[MemoryProfiler] matplotlib 未安装，跳过静态内存图。")
            return None
        if not self._static_result:
            logger.warning("[MemoryProfiler] 尚未调用 estimate()，无静态数据可绘图。")
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
            logger.info("[MemoryProfiler] 静态内存图已保存至: %s", path)
        if show:
            plt.show()
        plt.close(fig)
        return path

    def plot_runtime_memory(
        self,
        save_dir: Optional[str] = None,
        show: bool = False,
    ) -> Optional[str]:
        """生成运行时峰值显存折线图（上）与步骤耗时折线图（下）。"""
        if not _MPL_AVAILABLE:
            logger.warning("[MemoryProfiler] matplotlib 未安装，跳过运行时内存图。")
            return None
        if not self._runtime_results:
            logger.warning("[MemoryProfiler] 尚未调用 profile_step()，无运行时数据可绘图。")
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
            median_val = r.get('median_elapsed_ms', r['avg_elapsed_ms'])
            ax_time.axhline(median_val, linestyle='--',
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
            logger.info("[MemoryProfiler] 运行时内存图已保存至: %s", path)
        if show:
            plt.show()
        plt.close(fig)
        return path

    def plot_phase_breakdown(
        self,
        save_dir: Optional[str] = None,
        show: bool = False,
    ) -> Optional[str]:
        """生成前向/反向/优化器中位耗时分组柱状图（仅含分阶段数据的 tag）。"""
        if not _MPL_AVAILABLE:
            logger.warning("[MemoryProfiler] matplotlib 未安装，跳过分阶段柱状图。")
            return None

        # 筛选有分阶段数据的 tag
        phased = {tag: r for tag, r in self._runtime_results.items() if 'forward_ms' in r}
        if not phased:
            logger.debug("[MemoryProfiler] 无分阶段数据，跳过 phase_breakdown 图。")
            return None

        tags = list(phased.keys())
        fw_vals  = [phased[t]['median_forward_ms']  for t in tags]
        bw_vals  = [phased[t]['median_backward_ms']  for t in tags]
        opt_vals = [phased[t]['median_optimizer_ms'] for t in tags]

        import numpy as np
        x = np.arange(len(tags))
        width = 0.22

        fig, ax = plt.subplots(figsize=(max(8, len(tags) * 2), 5))
        bars_fw  = ax.bar(x - width, fw_vals,  width, label='Forward',   color='steelblue', alpha=0.85)
        bars_bw  = ax.bar(x,         bw_vals,  width, label='Backward',  color='coral',     alpha=0.85)
        bars_opt = ax.bar(x + width, opt_vals, width, label='Optimizer', color='seagreen',  alpha=0.85)

        # 数值标注
        for bars in (bars_fw, bars_bw, bars_opt):
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                            f'{h:.1f}', ha='center', va='bottom', fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(tags, rotation=15, ha='right')
        ax.set_ylabel('Median Time (ms)')
        ax.set_title('Phase Breakdown: Forward / Backward / Optimizer')
        ax.legend()
        ax.grid(axis='y', linestyle=':', alpha=0.5)
        plt.tight_layout()

        path = None
        if save_dir:
            path = os.path.join(save_dir, 'phase_breakdown.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            logger.info("[MemoryProfiler] 分阶段柱状图已保存至: %s", path)
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
        """将分析结果以 JSON 格式保存到 IR_artifacts 目录，并可选生成内存图。"""
        from ..utils.save_ir import _default_ir_dir  # 延迟导入，避免循环依赖

        out_dir = _default_ir_dir(
            model_name or os.getenv("MODEL_NAME", "default_model"),
            subfolder=subfolder,
        )
        path = os.path.join(out_dir, "memory_report.json")

        payload: Dict = {}

        # 静态报告增加 remaining_activations 和 primal_details
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
                'remaining_activations':   r['after']['activation_details'],
            }

        if self._param_memory:
            payload['parameter_memory'] = {
                k: int(v) if isinstance(v, int) else v
                for k, v in self._param_memory.items()
            }

        if self._runtime_results:
            runtime_payload = {}
            for tag, r in self._runtime_results.items():
                entry = {
                    'avg_peak_bytes':    r['avg_peak_bytes'],
                    'max_peak_bytes':    r['max_peak_bytes'],
                    'min_peak_bytes':    r['min_peak_bytes'],
                    'avg_elapsed_ms':    r['avg_elapsed_ms'],
                    'max_elapsed_ms':    r['max_elapsed_ms'],
                    'min_elapsed_ms':    r['min_elapsed_ms'],
                    'median_elapsed_ms': r['median_elapsed_ms'],
                    'iqr_elapsed_ms':    r['iqr_elapsed_ms'],
                    'compile_ms':        r['compile_ms'],
                    'peak_memory_bytes': r['peak_memory_bytes'],
                    'elapsed_ms':        r['elapsed_ms'],
                    'device':            r['device'],
                }
                # 分阶段字段（仅 phased 模式产生）
                if 'forward_ms' in r:
                    entry.update({
                        'forward_ms':          r['forward_ms'],
                        'backward_ms':         r['backward_ms'],
                        'optimizer_ms':        r['optimizer_ms'],
                        'median_forward_ms':   r['median_forward_ms'],
                        'median_backward_ms':  r['median_backward_ms'],
                        'median_optimizer_ms': r['median_optimizer_ms'],
                        'iqr_forward_ms':      r['iqr_forward_ms'],
                        'iqr_backward_ms':     r['iqr_backward_ms'],
                        'iqr_optimizer_ms':    r['iqr_optimizer_ms'],
                    })
                runtime_payload[tag] = entry
            payload['runtime'] = runtime_payload

            # 写入与基准的对比数据
            tags = list(self._runtime_results.keys())
            if len(tags) >= 2:
                r_base = self._runtime_results[tags[0]]
                comparisons = []
                for cmp_tag in tags[1:]:
                    r_cmp = self._runtime_results[cmp_tag]
                    iqr_delta = r_cmp['iqr_elapsed_ms'] - r_base['iqr_elapsed_ms']
                    iqr_pct   = (iqr_delta / r_base['iqr_elapsed_ms'] * 100
                                 if r_base['iqr_elapsed_ms'] > 0 else 0.0)
                    entry: Dict = {
                        'baseline': tags[0],
                        'compared': cmp_tag,
                        'iqr_time_delta_ms': iqr_delta,
                        'iqr_time_delta_pct': iqr_pct,
                    }
                    if r_base['avg_peak_bytes'] >= 0 and r_cmp['avg_peak_bytes'] >= 0:
                        mem_delta = r_base['avg_peak_bytes'] - r_cmp['avg_peak_bytes']
                        entry['mem_saved_bytes'] = mem_delta
                        if iqr_delta > 0 and mem_delta > 0:
                            entry['trade_off_mb_per_ms'] = (
                                mem_delta / (1 << 20)
                            ) / iqr_delta
                    comparisons.append(entry)
                payload['runtime_comparisons'] = comparisons

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        logger.info("[MemoryProfiler] 内存报告已保存至: %s", path)

        if save_plots:
            self.plot_static_memory(save_dir=out_dir)
            self.plot_runtime_memory(save_dir=out_dir)
            self.plot_phase_breakdown(save_dir=out_dir)

        return path


# 向后兼容别名
MemoryAnalyzer = MemoryProfiler
