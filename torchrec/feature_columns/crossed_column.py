"""
分类特征交叉列
"""
from typing import Dict, Any, List

from torch import LongTensor

from .categorical_column import CategoricalColumn


class CrossedColumn:
    """分类特征交叉列，输入必须是已经创建好的分类特征列"""

    def __init__(self, categorical_columns: List[CategoricalColumn]):
        self.categorical_columns = categorical_columns
        self.coefficients = [1] * len(categorical_columns)
        for i in range(len(categorical_columns) - 1, 0, -1):
            self.coefficients[i - 1] = self.coefficients[i] * self.categorical_columns[i].category_num

    def get_feature_data(self, batch: Dict[str, Any]) -> LongTensor:
        """获取特征数据"""
        return sum((self.coefficients[i] * batch[self.categorical_columns[i].feature_name].long()
                    for i in range(len(self.coefficients))))
