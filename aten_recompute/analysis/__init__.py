"""
aten_recompute.analysis — 内存分析与估算子包。

提供运行时 GPU 内存采样、静态峰值显存估算、SOTA 方法对比等能力。
"""

from .profiler import MemoryProfiler
from .static import StaticEstimator
from .comparison import print_method_comparison, get_method_comparison_data
from ._activations import _fmt_bytes, _saved_activation_bytes

# 向后兼容别名
MemoryAnalyzer = MemoryProfiler
StaticMemoryEstimator = StaticEstimator

__all__ = [
    "MemoryProfiler",
    "MemoryAnalyzer",
    "StaticEstimator",
    "StaticMemoryEstimator",
    "print_method_comparison",
    "get_method_comparison_data",
    "_fmt_bytes",
    "_saved_activation_bytes",
]
