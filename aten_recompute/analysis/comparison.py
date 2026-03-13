"""
comparison.py

SOTA 重计算/重实体化方法的理论对比表。
覆盖 PyTorch 内建方法、ATenIR 策略、以及学术界代表性方法。
"""


_METHODS = [
    # (方法名, 粒度, 最优性, 额外开销, 决策时机, 参考文献)
    ("Eager (no recompute)",  "N/A",    "N/A",             "无",           "N/A",    "—"),
    ("PyTorch Checkpoint",    "按层",   "手动",            "低",           "静态",   "PyTorch docs"),
    ("PyTorch SAC",           "按算子", "策略函数",        "低",           "编译时", "PyTorch 2.x"),
    ("ATenIR Strategy 6",     "按算子", "启发式(链深度)",  "低",           "编译时", "本项目"),
    ("ATenIR Strategy 7",     "按算子", "最优(min-cut)",   "中(NetworkX)", "编译时", "本项目"),
    ("DTR",                   "按张量", "贪心(在线)",      "运行时开销",   "动态",   "ICLR 2021"),
    ("Checkmate",             "按算子", "最优(ILP)",       "高(求解器)",   "离线",   "MLSys 2020"),
    ("Rockmate",              "按块",   "近最优(层次)",    "中",           "离线",   "ICML 2023"),
]


def print_method_comparison():
    """打印重计算/重实体化方法的理论对比表。"""
    headers = ("方法", "粒度", "最优性", "额外开销", "决策时机", "参考")
    widths  = (26, 8, 18, 14, 8, 14)

    bold = "═" * 68
    line = "─" * 68
    print(f"\n{bold}")
    print("  重计算方法理论对比")
    print(bold)

    header_line = "  "
    for h, w in zip(headers, widths):
        header_line += f"{h:<{w}}"
    print(header_line)
    print(f"  {line}")

    for row in _METHODS:
        row_line = "  "
        for val, w in zip(row, widths):
            row_line += f"{val:<{w}}"
        print(row_line)

    print(bold)

    print("  ATenIR 特点：")
    print("    ✓ 在 AOT Autograd joint graph 级别控制 saved_values")
    print("    ✓ 与 torch.compile 深度集成，无运行时额外开销")
    print("    ✓ 多种策略可选（启发式 → 最优）")
    print("    × 静态决策，无法根据运行时内存压力自适应（vs DTR）")
    print("    × 不支持 CPU offloading（vs Capuchin/ROTOR）")
    print(f"{bold}\n")


def get_method_comparison_data() -> list:
    """返回方法对比数据，用于 JSON 报告。"""
    headers = ("method", "granularity", "optimality", "overhead", "timing", "reference")
    return [dict(zip(headers, row)) for row in _METHODS]
