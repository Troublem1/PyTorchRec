"""
数据集模式
"""
from enum import Enum, unique


@unique
class DatasetMode(Enum):
    """数据集模式"""
    SEQUENTIAL_SPLIT = "sequential_split"
    LEAVE_K_OUT = "leave_k_out"
