"""
回调列表
"""
from typing import List, Optional, Dict

from torchrec.callback.History import History
from torchrec.callback.ICallback import ICallback
from torchrec.callback.ProgbarLogger import ProgbarLogger
from torchrec.utils.data_structure import tensor_to_numpy_or_python_type


class CallbackList:
    """回调列表"""

    def __init__(self,
                 callbacks: Optional[List[ICallback]] = None,
                 add_history: bool = False,
                 add_progbar: bool = False,
                 recommender=None,
                 **params):
        """
        创建回调函数列表
        :param callbacks: 回调对象列表
        :param add_history: 是否添加History
        :param add_progbar: 是否添加ProgbarLogger
        :param recommender: 推荐算法引用
        :param params: 其他要传递给ICallback.set_params的参数
        """
        self.callbacks: List[ICallback] = callbacks or []
        self._progbar = None
        self._history = None

        for callback in self.callbacks:
            if isinstance(callback, ProgbarLogger):
                self._progbar = callback
            elif isinstance(callback, History):
                self._history = callback

        if self._progbar is None and add_progbar:
            self._progbar = ProgbarLogger()
            self.callbacks.append(self._progbar)

        if self._history is None and add_history:
            self._history = History()
            self.callbacks.append(self._history)

        if recommender:
            self.recommender = recommender
            if self._history:
                recommender.history = self._history
            for callback in self.callbacks:
                callback.set_recommender(recommender)
        if params:
            self.params = params
            for callback in self.callbacks:
                callback.set_params(params)

        # todo 根据回调是否实现了batch级别的回调，来决定是否调用（优化性能）
        # todo 根据回调时间判断回调占时与batch占时比例，避免回调严重拖慢速度

    @staticmethod
    def _process_logs(logs: Optional[Dict]):
        """将tensor转化为ndarray或者标量数据"""
        if logs:
            return tensor_to_numpy_or_python_type(logs)
        return {}

    def append(self, callback: ICallback):
        """添加回调"""
        self.callbacks.append(callback)

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        """轮次开始"""
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """轮次结束"""
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_train_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        """训练批次开始"""
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_train_batch_begin(batch, logs)

    def on_train_batch_end(self, batch: int, logs: Optional[Dict] = None):
        """训练批次结束"""
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_train_batch_end(batch, logs)

    def on_test_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        """验证/测试批次开始"""
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_test_batch_begin(batch, logs)

    def on_test_batch_end(self, batch: int, logs: Optional[Dict] = None):
        """验证/测试批次结束"""
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_test_batch_end(batch, logs)

    def on_predict_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        """预测批次开始"""
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_predict_batch_begin(batch, logs)

    def on_predict_batch_end(self, batch: int, logs: Optional[Dict] = None):
        """预测批次结束"""
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_predict_batch_end(batch, logs)

    def on_train_begin(self, logs: Optional[Dict] = None):
        """训练开始"""
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs: Optional[Dict] = None):
        """训练结束"""
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_test_begin(self, logs: Optional[Dict] = None):
        """验证/测试开始"""
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_test_begin(logs)

    def on_test_end(self, logs: Optional[Dict] = None):
        """验证/测试结束"""
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_test_end(logs)

    def on_predict_begin(self, logs: Optional[Dict] = None):
        """预测开始"""
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_predict_begin(logs)

    def on_predict_end(self, logs: Optional[Dict] = None):
        """预测结束"""
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_predict_end(logs)

    def __iter__(self):
        return iter(self.callbacks)
