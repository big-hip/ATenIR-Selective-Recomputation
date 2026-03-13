"""ATenIR Selective Recomputation — 基于 ATen IR 的选择性重计算框架。"""

import os
from .utils.logger import get_logger

# 默认 logger，整个 aten_recompute 包内统一使用
_log_file = os.getenv("RECOMPUTE_LOG_FILE", "aten_recompute.log")
logger = get_logger(__name__, _log_file)

from .core import (
    CompilerBackend,
    inject_layer_tags,
    make_selective_partition_fn,
    selective_recompute_partition,
)
from .analysis import (
    MemoryProfiler,
    StaticEstimator,
    print_method_comparison,
)
from .utils import apply_activation_checkpoint

__all__ = [
    "CompilerBackend",
    "inject_layer_tags",
    "make_selective_partition_fn",
    "selective_recompute_partition",
    "MemoryProfiler",
    "StaticEstimator",
    "print_method_comparison",
    "apply_activation_checkpoint",
    "get_logger",
    "logger",
]
