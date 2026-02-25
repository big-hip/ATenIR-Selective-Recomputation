from .utils.logger import get_logger

# 默认 logger，整个 aten_recompute 包内统一使用
logger = get_logger(__name__, "aten_recompute.log")

__all__ = ["logger"]

