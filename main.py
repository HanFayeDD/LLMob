from configmanager.configloader import ConfigLoader
from utils.logger import init_log
import os
import logging





if __name__ == "__main__":
    # 优先从环境变量读取日志级别（例如：LOG_LEVEL=DEBUG），否则使用 INFO
    init_log()
    llmapiconf = ConfigLoader().load()
    

    # print(llmapiconf)