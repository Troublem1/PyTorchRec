"""
任务模式
"""
from enum import Enum, unique


@unique
class TaskMode(Enum):
    """任务模式"""
    RERANK = "rerank"
    TOPK = "topk"
