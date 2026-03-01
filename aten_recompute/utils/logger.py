import logging
import os
from logging.handlers import RotatingFileHandler

# 统一将日志也放到 IR_artifacts 大目录下
# 注意：使用延迟创建策略——只在真正需要文件 handler 时才创建目录，
# 避免在测试 / CI 中导入包时污染当前工作目录。
_PROJECT_ROOT = os.getenv("PROJECT_ROOT", os.getcwd())
LOGS_DIR = os.path.join(_PROJECT_ROOT, "IR_artifacts", "logs")


class _ModulePathFormatter(logging.Formatter):
    """
    将 %(pathname)s 替换为相对于项目根目录的 Python 模块路径。

    例如：/project/aten_recompute/core/recompute.py → aten_recompute.core.recompute
    """

    def format(self, record: logging.LogRecord) -> str:
        record.module_line = f"{self._to_module_path(record.pathname)}:{record.lineno}"
        return super().format(record)

    @staticmethod
    def _to_module_path(abs_path: str) -> str:
        """将绝对路径转换为点分隔的模块路径（不含 .py 后缀）。"""
        project_root = os.getenv("PROJECT_ROOT", os.getcwd())
        try:
            rel = os.path.relpath(abs_path, start=project_root)
        except ValueError:
            # Windows 上跨盘符时 relpath 会抛出 ValueError
            return os.path.basename(abs_path).replace(".py", "")
        # 去掉 .py，路径分隔符换成点
        return rel.replace(os.sep, ".").removesuffix(".py")


def get_logger(module_name: str, log_file_name: str = None) -> logging.Logger:
    """
    获取（或复用）一个配置好的 Logger。

    Parameters
    ----------
    module_name   : 通常传入 __name__，决定 Logger 层级。
    log_file_name : 可选。若提供，则同时输出到
                    IR_artifacts/logs/<log_file_name>（带滚动）。

    日志级别通过环境变量 RECOMPUTE_LOG_LEVEL 控制，默认 INFO。
    """
    logger = logging.getLogger(module_name)

    # 从环境变量读取日志级别
    level_str = os.environ.get("RECOMPUTE_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    logger.setLevel(level)

    # 避免重复添加 Handler（同一 Logger 被多次调用时）
    if not logger.handlers:
        fmt = "%(asctime)s - %(module_line)s - %(levelname)s - %(message)s"
        formatter = _ModulePathFormatter(fmt)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file_name:
            os.makedirs(LOGS_DIR, exist_ok=True)          # 延迟创建目录
            log_file_path = os.path.join(LOGS_DIR, log_file_name)
            file_handler = RotatingFileHandler(
                log_file_path,
                maxBytes=10 * 1024 * 1024,   # 单文件上限 10 MB
                backupCount=3,               # 保留最近 3 个滚动备份
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    logger.propagate = False
    return logger
