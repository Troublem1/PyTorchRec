"""
分类特征类
"""
from typing import Dict, Any

from torch import LongTensor

from torchrec.utils.const import *


class CategoricalColumn:
    """分类特征类，输入必须是整数，需要提前处理成密集分类，等价于生成[0-max]的one-hot向量"""

    def __init__(self, feature_name: str, max_value: int):
        self.feature_name = feature_name
        self.category_num = max_value + 1

    def get_feature_data(self, batch: Dict[str, Any]) -> LongTensor:
        """获取特征数据"""
        return batch[self.feature_name].long()

    @staticmethod
    def from_description_dict(description: Dict):
        """从词典中构造"""
        assert description[FEATURE_TYPE] == CATEGORICAL_COLUMN
        return CategoricalColumn(
            feature_name=description[FEATURE_NAME],
            max_value=description[MAX]
        )
