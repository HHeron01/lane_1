# Copyright (c) OpenMMLab. All rights reserved.
import logging

from mmcv.utils import get_logger


def get_root_logger(log_file=None, log_level=logging.INFO, name='mmdet3d'):
    """获取根日志记录器并添加一个关键字过滤器。

    如果根日志记录器尚未初始化，则将对其进行初始化。默认情况下，将添加一个StreamHandler。
    如果指定了`log_file`，还将添加一个FileHandler。根日志记录器的名称是顶级包的名称，例如"mmdet3d"。
    Args:
        log_file (str, optional): 日志文件的路径。默认为None。
        log_level (int, optional): 日志记录器的级别。默认为logging.INFO。
        name (str, optional): 根日志记录器的名称，也用作过滤关键字。默认为'mmdet3d'。
    Returns:
        :obj:`logging.Logger`: 获取到的日志记录器。
    """
    logger = get_logger(name=name, log_file=log_file, log_level=log_level)

    # 添加一个日志记录过滤器
    logging_filter = logging.Filter(name)
    # class Filter(object):
    # """
    # 过滤器实例用于执行LogRecords的任意过滤。

    # Loggers和Handlers可以选择使用Filter实例来根据需要过滤记录。
    # 基础过滤器类仅允许位于记录器层次结构中某一点下面的事件。
    # 例如，使用"A.B"初始化的过滤器将允许由记录器"A.B"、"A.B.C"、"A.B.C.D"、"A.B.D"等记录器记录的事件，
    # 但不允许"A.BB"、"B.A.B"等事件。如果使用空字符串初始化，将允许所有事件通过。
    # """

    # def __init__(self, name=''):
    #     """
    #     初始化过滤器。

    #     使用记录器的名称进行初始化，该名称与其子记录器一起将其事件通过过滤器。
    #     如果未指定名称，将允许每个事件。
    #     Args:
    #         name (str, optional): 要过滤的记录器的名称。默认为''（空字符串）。
    #     """
    #     self.name = name
    #     self.nlen = len(name)

    # def filter(self, record):
    #     """
    #     确定是否应记录指定的记录。

    #     如果应记录记录，则返回True，否则返回False。
    #     如果适当，可以原地修改记录。
    #     Args:
    #         record (LogRecord): 要过滤的日志记录。

    #     Returns:
    #         bool: 如果应记录记录，则为True；否则为False。
    #     """
    #     if self.nlen == 0:
    #         return True
    #     elif self.name == record.name:
    #         return True
    #     elif record.name.find(self.name, 0, self.nlen) != 0:
    #         return False
    #     return (record.name[self.nlen] == ".")

    
    logging_filter.filter = lambda record: record.find(name) != -1

    return logger
