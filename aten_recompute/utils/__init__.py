from .logger import get_logger
from .save_ir import save_fx_module_code_and_graph, save_graphmodule_readable, save_ir_and_dot
from .graph_utils import get_output_node, get_saved_activations
from .checkpoint import (
    apply_activation_checkpoint,
    remove_activation_checkpoint,
    wrap_layer_with_checkpoint,
)

# 向后兼容：从旧路径导入的代码仍可通过 utils 访问
from ..analysis import (
    MemoryProfiler,
    MemoryAnalyzer,
    StaticEstimator,
    StaticMemoryEstimator,
    print_method_comparison,
    get_method_comparison_data,
)

__all__ = [
    "get_logger",
    "save_fx_module_code_and_graph",
    "save_graphmodule_readable",
    "save_ir_and_dot",
    "get_output_node",
    "get_saved_activations",
    "apply_activation_checkpoint",
    "remove_activation_checkpoint",
    "wrap_layer_with_checkpoint",
    # 向后兼容（推荐从 aten_recompute.analysis 导入）
    "MemoryProfiler",
    "MemoryAnalyzer",
    "StaticEstimator",
    "StaticMemoryEstimator",
    "print_method_comparison",
    "get_method_comparison_data",
]
