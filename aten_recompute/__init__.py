import os
from .utils.logger import get_logger

# 默认 logger，整个 aten_recompute 包内统一使用
_log_file = os.getenv("RECOMPUTE_LOG_FILE", "aten_recompute.log")
logger = get_logger(__name__, _log_file)

__all__ = ["logger"]
