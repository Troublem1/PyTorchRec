"""
数值列
"""
from typing import Dict, Any

from torch import FloatTensor

from .normalization_mode import NormalizationMode


class NumericColumn:
    """数值特征列"""

    def __init__(self, feature_name: str, min_value: float, max_value: float, mean_value: float, std_value: float):
        self.feature_name = feature_name
        self.min_value = min_value
        self.max_value = max_value
        self.mean_value = mean_value
        self.std_value = std_value

    def get_feature_data(self, batch: Dict[str, Any], normalization_mode: NormalizationMode = NormalizationMode.NOP) \
            -> FloatTensor:
        """获取特征数据"""
        if normalization_mode == NormalizationMode.NOP:
            return batch[self.feature_name]
        if normalization_mode == NormalizationMode.MAX_MIN:
            return (batch[self.feature_name] - self.min_value) / (self.max_value - self.min_value)
        if normalization_mode == NormalizationMode.Z_SCORE:
            return (batch[self.feature_name] - self.mean_value) / self.std_value
        raise Exception("NormalizationMode is wrong!")
