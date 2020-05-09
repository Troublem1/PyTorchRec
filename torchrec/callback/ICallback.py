"""
回调函数接口类
"""
from abc import ABC
from typing import Optional, Dict


class ICallback(ABC):
    """回调函数接口类"""

    def __init__(self):
        self.recommender = None
        self.params: Optional[Dict] = None

    def set_params(self, params):
        """设置参数"""
        self.params = params

    def set_recommender(self, recommender):
        """引用推荐模型"""
        self.recommender = recommender

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        """轮次开始时调用"""
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """轮次结束时调用"""
        pass

    def on_train_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        """训练批次开始时调用"""
        pass

    def on_train_batch_end(self, batch: int, logs: Optional[Dict] = None):
        """训练批次结束后调用"""
        pass

    def on_test_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        """验证/测试批次开始的时候"""
        pass

    def on_test_batch_end(self, batch: int, logs: Optional[Dict] = None):
        """验证/测试批次结束的时候"""
        pass

    def on_predict_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        """预测批次开始的时候"""
        pass

    def on_predict_batch_end(self, batch: int, logs: Optional[Dict] = None):
        """预测批次结束的时候"""
        pass

    def on_train_begin(self, logs: Optional[Dict] = None):
        """训练开始的时候"""
        pass

    def on_train_end(self, logs: Optional[Dict] = None):
        """训练结束的时候"""
        pass

    def on_test_begin(self, logs: Optional[Dict] = None):
        """验证/测试开始的时候"""
        pass

    def on_test_end(self, logs: Optional[Dict] = None):
        """验证/测试结束的时候"""
        pass

    def on_predict_begin(self, logs: Optional[Dict] = None):
        """预测开始的时候"""
        pass

    def on_predict_end(self, logs: Optional[Dict] = None):
        """预测结束的时候"""
        pass
