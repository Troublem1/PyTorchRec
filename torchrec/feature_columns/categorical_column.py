"""
分类特征类
"""
from typing import Dict, Any

from torch import LongTensor


class CategoricalColumn:
    """分类特征类，输入必须是整数，需要提前处理成密集分类，等价于生成[0-max]的one-hot向量"""

    def __init__(self, feature_name: str, max_value: int):
        self.feature_name = feature_name
        self.category_num = max_value + 1

    def get_feature_data(self, batch: Dict[str, Any]) -> LongTensor:
        """获取特征数据"""
        return batch[self.feature_name].long()
