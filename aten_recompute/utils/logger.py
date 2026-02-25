import logging
import os

# 确保 logs 目录存在
LOGS_DIR = "dist_logs"  # 日志目录名称
os.makedirs(LOGS_DIR, exist_ok=True)  # 如果目录不存在，则创建

class CustomFormatter(logging.Formatter):
    def format(self, record):
        # 获取相对路径并转换为模块路径
        record.pathname = self.convert_to_module_path(record.pathname)
        record.module_line = f"{record.pathname}:{record.lineno}"
        return super().format(record)

    def convert_to_module_path(self, path):
        # 将文件路径转换为模块路径
        base_path = os.path.dirname(__file__)  # 获取当前文件的目录
        relative_path = os.path.relpath(path, start=base_path)
        module_path = relative_path.replace(os.sep, '.').replace('.py', '')
        # 只保留最后一个文件名部分，并判断文件名前的那一级目录是否是指定的目录
        parts = module_path.split('.')
        if len(parts) > 1 and parts[-2] in ['Data_Parallelism', 'Pipeline_Parallelism', 'Tensor_Parallelism']:
            module_path = '.'.join(parts[-2:])
        else:
            module_path = parts[-1]

        module_path = 'Distributed_Parallelism.'+ module_path
        return module_path

def get_logger(module_name, log_file_name=None):

    # 创建日志记录器
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)

    # 设置日志格式，包含模块路径和行号
    log_format = "%(asctime)s - %(module_line)s - %(levelname)s - %(message)s"
    formatter = CustomFormatter(log_format)

    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 如果指定了日志文件名，添加文件处理器
    if log_file_name:
        # 确保日志文件保存在 logs 目录下
        log_file_path = os.path.join(LOGS_DIR, log_file_name)
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # 避免日志重复输出
    logger.propagate = False

    return logger