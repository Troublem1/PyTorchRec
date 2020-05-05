"""
系统相关模块
"""
import logging

import os


def init_console_logger() -> None:
    """
    初始化终端日志打印
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(filename)s %(funcName)s :: %(message)s')


def check_dir_and_mkdir(path: str) -> None:
    """
    检查路径是否存在，如果不存在则创建路劲，如果输入是文件名则自动提取路径。
    """
    if os.path.basename(path).find('.') == -1 or path.endswith('/'):
        dirname = path
    else:
        dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        logging.info(f'新建文件夹：{dirname}')
        os.makedirs(dirname)
    return
