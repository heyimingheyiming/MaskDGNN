#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description :
import logging  # 导入 logging 模块，用于记录日志
import os  # 导入 os 模块，用于操作系统相关功能
import time  # 导入 time 模块，用于处理时间相关功能

def getLogger(log_path):
    logger = logging.getLogger()  # 获取一个 logger 对象，默认名称为空字符串
    logger.setLevel(logging.INFO)  # 设置日志记录的级别为 INFO，低于这个级别的日志将不会被记录
    formatter = logging.Formatter(fmt="[%(asctime)s|%(filename)s|%(levelname)s] %(message)s",
                                  datefmt="%a %b %d %H:%M:%S %Y")  # 创建一个格式化器，指定日志的输出格式和时间格式

    # StreamHandler
    sHandler = logging.StreamHandler()  # 创建一个流处理器，用于将日志输出到控制台
    sHandler.setFormatter(formatter)  # 为流处理器设置格式化器
    logger.addHandler(sHandler)  # 将流处理器添加到 logger 对象中

    # FileHandler
    work_dir = os.path.join(log_path,
                            time.strftime("%Y-%m-%d-%H.%M", time.localtime()))  # 生成日志文件存放的目录路径，包含当前时间
    if not os.path.exists(work_dir):  # 如果目录不存在
        os.makedirs(work_dir)  # 创建目录
    fHandler = logging.FileHandler(work_dir + '/log.txt', mode='w')  # 创建一个文件处理器，指定日志文件路径和写入模式
    fHandler.setLevel(logging.DEBUG)  # 设置文件处理器的日志记录级别为 DEBUG
    fHandler.setFormatter(formatter)  # 为文件处理器设置格式化器
    logger.addHandler(fHandler)  # 将文件处理器添加到 logger 对象中

    return logger  # 返回配置好的 logger 对象

