"""
comparison.py

SOTA 重计算/重实体化方法的理论对比表。
覆盖 PyTorch 内建方法、工业框架方法、ATenIR 策略、经典学术方法，
以及压缩/卸载等正交方向。
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  重计算方法数据
# ═══════════════════════════════════════════════════════════════════════════════

# (方法名, 粒度, 最优性, 额外开销, 决策时机, 参考文献)
_METHODS_RECOMPUTE = [
    # ── 基线 ──────────────────────────────────────────────────────────────────
    ("Eager (no recompute)",    "N/A",    "N/A",             "无",           "N/A",    "—"),

    # ── PyTorch 内建 ──────────────────────────────────────────────────────────
    ("PyTorch Checkpoint",      "按层",   "手动",            "低",           "静态",   "PyTorch docs"),
    ("PyTorch SAC",             "按算子", "策略函数",        "低",           "编译时", "PyTorch 2.x"),
    ("Memory Budget API",       "按算子", "knapsack 求解",   "低",           "编译时", "PyTorch 2.4+"),

    # ── 工业框架 ──────────────────────────────────────────────────────────────
    ("Megatron Selective",      "按子模块", "经验规则",      "低",           "静态",   "MLSys 2023"),
    ("DeepSpeed Checkpoint",    "按层",   "手动",            "低",           "静态",   "DeepSpeed"),

    # ── 本项目 ────────────────────────────────────────────────────────────────
    ("ATenIR Strategy 6",       "按算子", "启发式(链深度)",  "低",           "编译时", "本项目"),
    ("ATenIR Strategy 7",       "按算子", "最优(min-cut)",   "中(NetworkX)", "编译时", "本项目"),

    # ── 经典学术 ──────────────────────────────────────────────────────────────
    ("Chen et al.",             "按段",   "均匀分段 O(√N)",  "低",           "静态",   "arXiv 2016"),
    ("Rotor",                   "按层",   "最优(DP)",        "低(profiling)","离线",   "INRIA 2019"),
    ("Checkmate",               "按算子", "最优(ILP)",       "高(求解器)",   "离线",   "MLSys 2020"),
    ("MONeT",                   "按算子", "最优(ILP 联合)",  "高(求解器)",   "离线",   "ICLR 2021"),
    ("DTR",                     "按张量", "贪心(在线)",      "运行时开销",   "动态",   "ICLR 2021"),
    ("Rockmate",                "按块",   "近最优(层次DP)",  "中",           "离线",   "ICML 2023"),
]

# ═══════════════════════════════════════════════════════════════════════════════
#  正交 / 互补方向
# ═══════════════════════════════════════════════════════════════════════════════

# (方法名, 类别, 核心思路, 代表性效果, 参考文献)
_METHODS_ORTHOGONAL = [
    # ── 压缩（不丢弃激活，量化保存）────────────────────────────────────────
    ("ActNN",       "压缩", "2-bit 混合精度量化激活",              "12x 激活显存压缩",     "ICML 2021"),
    ("GACT",        "压缩", "架构无关压缩 + 运行时敏感度估计",    "最高 8.1x 压缩",       "ICML 2022"),
    ("COAT",        "压缩", "FP8 压缩激活 + 优化器状态",          "1.54x 显存, 1.43x 速度", "ICLR 2025"),

    # ── 卸载（CPU / SSD）──────────────────────────────────────────────────
    ("vDNN",        "卸载", "逐层激活 swap 到 CPU",                "开创性 GPU-CPU swap",  "2016"),
    ("POET",        "卸载", "ILP 最优重计算 + paging 联合调度",    "边缘设备微调 BERT",    "ICML 2022"),
    ("SSDTrain",    "卸载", "自适应激活 offload 到 NVMe SSD",      "数据传输与计算完全重叠","2024"),

    # ── 混合（重计算 + 卸载 + 压缩）─────────────────────────────────────
    ("Capuchin",    "混合", "运行时 profiling 逐 tensor 决策重计算 vs swap", "BERT 7x batch", "ASPLOS 2020"),
    ("DELTA",       "混合", "动态细粒度重计算 + swap",             "GPT2-XL 6x batch",     "TACO 2024"),
    ("Adacc",       "混合", "自适应混合压缩 + 重计算",             "逐 tensor 最优策略",   "2025"),

    # ── 流水线并行感知 ───────────────────────────────────────────────────
    ("AdaPipe",     "PP感知", "自适应逐 stage 重计算策略(DP)",      "异构 stage 最优调度",  "ASPLOS 2024"),
    ("Obscura",     "PP感知", "重计算隐藏在流水线气泡 + 集成 swap", "气泡利用率提升",       "ATC 2025"),
]


# ═══════════════════════════════════════════════════════════════════════════════
#  打印函数
# ═══════════════════════════════════════════════════════════════════════════════

def print_method_comparison():
    """打印重计算/重实体化方法的理论对比表。"""
    bold = "═" * 92
    line = "─" * 92

    # ── 表 1: 重计算方法 ──────────────────────────────────────────────────
    headers = ("方法", "粒度", "最优性", "额外开销", "决策时机", "参考")
    widths  = (26, 10, 18, 16, 10, 16)

    print(f"\n{bold}")
    print("  表 1  重计算 (Recomputation) 方法对比")
    print(bold)

    header_line = "  "
    for h, w in zip(headers, widths):
        header_line += f"{h:<{w}}"
    print(header_line)
    print(f"  {line}")

    for row in _METHODS_RECOMPUTE:
        row_line = "  "
        for val, w in zip(row, widths):
            row_line += f"{val:<{w}}"
        print(row_line)

    print(bold)

    # ── 表 2: 正交 / 互补方向 ────────────────────────────────────────────
    headers2 = ("方法", "类别", "核心思路", "代表性效果", "参考")
    widths2  = (14, 8, 40, 22, 14)

    print(f"\n{bold}")
    print("  表 2  正交 / 互补方向（压缩、卸载、混合、流水线感知）")
    print(bold)

    header_line2 = "  "
    for h, w in zip(headers2, widths2):
        header_line2 += f"{h:<{w}}"
    print(header_line2)
    print(f"  {line}")

    for row in _METHODS_ORTHOGONAL:
        row_line = "  "
        for val, w in zip(row, widths2):
            row_line += f"{val:<{w}}"
        print(row_line)

    print(bold)

    # ── ATenIR 定位 ──────────────────────────────────────────────────────
    print("\n  ATenIR 定位与特点：")
    print("    ✓ 在 AOT Autograd joint graph 级别控制 saved_values")
    print("    ✓ 与 torch.compile 深度集成，无运行时额外开销")
    print("    ✓ 多种策略可选（启发式 → 最优），无需用户手写 policy")
    print("    ✓ 自动化程度高于 Megatron 经验规则和 PyTorch SAC 手动 policy")
    print("    △ 与 Memory Budget API 同赛道，优势在于策略多样性和可解释性")
    print("    × 静态决策，无法根据运行时内存压力自适应（vs DTR/Capuchin）")
    print("    × 不支持激活压缩（vs ActNN/COAT）和 CPU/SSD offloading（vs DELTA）")
    print("    × 单体 BW 函数无法逐层释放激活（vs eager checkpoint 的逐层 GC）")
    print(f"{bold}\n")


def get_method_comparison_data() -> dict:
    """返回方法对比数据，用于 JSON 报告。"""
    recompute_headers = (
        "method", "granularity", "optimality", "overhead", "timing", "reference",
    )
    orthogonal_headers = (
        "method", "category", "approach", "result", "reference",
    )
    return {
        "recomputation_methods": [
            dict(zip(recompute_headers, row)) for row in _METHODS_RECOMPUTE
        ],
        "orthogonal_methods": [
            dict(zip(orthogonal_headers, row)) for row in _METHODS_ORTHOGONAL
        ],
    }
