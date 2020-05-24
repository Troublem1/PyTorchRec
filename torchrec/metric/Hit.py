"""
Hit
"""
import numpy as np
from numpy import ndarray

from torchrec.metric.IMetric import IMetric


class Hit(IMetric):
    """Hit"""

    def __init__(self, user_sample_n: int, k: int):
        self.user_sample_n = user_sample_n
        self.k = k
        super().__init__()
        self.name = f"hit@{self.k}"

    def __call__(self, prediction, target, *args, **kwargs):
        # TopK 验证集与测试集不随机打乱的情况下，每 user_sample_n 个数据是一个用户
        prediction_array: ndarray = prediction.reshape((-1, self.user_sample_n))  # [user_num, user_sample_n]

        # 按照预测值获得正例预测值排序
        # [user_num, user_sample_n]
        sort_idx_array: ndarray = (-prediction_array).argsort()
        # 这里用到一个技巧，每行只有一个正例并且位于第一个，因此只要确认 sort_idx 每行下标 0 的位置即可，然后 +1 确保下标从 1 开始
        pos_ranks: ndarray = np.argwhere(sort_idx_array == 0)[:, 1] + 1  # [user_num]

        hit_pos_ranks = pos_ranks[pos_ranks <= self.k]
        return len(hit_pos_ranks) / len(pos_ranks)
