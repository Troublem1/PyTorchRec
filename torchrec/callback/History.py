"""
训练历史
"""
from typing import List, Optional, Dict

from torchrec.callback.ICallback import ICallback


class History(ICallback):
    """
    训练历史记录
    该回调会自动应用于IRecommender.fit函数，并返回
    """

    def __init__(self):
        super().__init__()
        self.epoch: List[int] = []
        self.history = {}

    def on_train_begin(self, logs: Optional[Dict] = None):
        """训练开始的时候"""
        self.epoch: List[int] = []

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """轮次结束时调用"""
        logs: Dict = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        self.recommender.history = self
