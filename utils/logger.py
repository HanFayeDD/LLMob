import logging
import os

import logging
from logging.handlers import RotatingFileHandler


def init_log():
    # 创建日志目录（如果不存在）
    log_dir = "log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 设置日志文件路径
    log_file_path = os.path.join(log_dir, "app.log")

    # 创建Logger实例
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # 设置全局日志级别

    # 创建文件处理器（按文件大小轮转）
    file_handler = RotatingFileHandler(
        log_file_path, maxBytes=50*1024*1024, backupCount=1, encoding="utf-8"  # 每个文件5MB，保留3个备份
    )
    file_handler.setLevel(logging.DEBUG)  # 文件记录DEBUG及以上级别

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # 控制台记录INFO及以上级别

    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s|%(levelname)s|%(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器到Logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logging.info("初始化日志模块")
    

