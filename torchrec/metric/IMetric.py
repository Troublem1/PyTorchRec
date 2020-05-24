"""
评价指标接口类
"""
from abc import ABC, abstractmethod


class IMetric(ABC):
    """评价指标接口类"""

    def __init__(self):
        self.name = "IMetric"

    @abstractmethod
    def __call__(self, prediction, target, *args, **kwargs):
        pass
