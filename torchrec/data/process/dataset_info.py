"""
检查已有数据集列表
"""
from typing import List

from torchrec.utils.const import *


def check_dataset_info() -> List[str]:
    """检查已有数据集列表"""
    dataset_list = list()
    for filename in os.listdir(DATASET_DIR):
        dataset_list.append(filename)
    dataset_list.sort()
    return dataset_list
