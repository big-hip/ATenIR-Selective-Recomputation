import logging
import os

LOGS_DIR = "dist_logs"
os.makedirs(LOGS_DIR, exist_ok=True)

class CustomFormatter(logging.Formatter):
    # ... 保持你现有的 CustomFormatter 代码不变 ...
    def format(self, record):
        record.pathname = self.convert_to_module_path(record.pathname)
        record.module_line = f"{record.pathname}:{record.lineno}"
        return super().format(record)

    def convert_to_module_path(self, path):
        # ... 保持原样 ...
        base_path = os.path.dirname(__file__)
        relative_path = os.path.relpath(path, start=base_path)
        module_path = relative_path.replace(os.sep, '.').replace('.py', '')
        parts = module_path.split('.')
        if len(parts) > 1 and parts[-2] in ['Data_Parallelism', 'Pipeline_Parallelism', 'Tensor_Parallelism']:
            module_path = '.'.join(parts[-2:])
        else:
            module_path = parts[-1]
        module_path = 'Distributed_Parallelism.'+ module_path
        return module_path

def get_logger(module_name, log_file_name=None):
    logger = logging.getLogger(module_name)
    
    # 核心修改：从环境变量读取日志级别，默认为 INFO
    log_level_str = os.environ.get("RECOMPUTE_LOG_LEVEL", "INFO").upper()
    
    # 将字符串映射为 logging 的内置级别
    level_mapping = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    
    log_level = level_mapping.get(log_level_str, logging.INFO)
    logger.setLevel(log_level)

    # 避免重复添加 Handler（防止在多次调用 get_logger 时打印重复日志）
    if not logger.handlers:
        log_format = "%(asctime)s - %(module_line)s - %(levelname)s - %(message)s"
        formatter = CustomFormatter(log_format)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file_name:
            log_file_path = os.path.join(LOGS_DIR, log_file_name)
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    logger.propagate = False
    return logger