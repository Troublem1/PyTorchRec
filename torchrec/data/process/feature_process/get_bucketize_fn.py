"""
获取分桶函数
"""
import math
from typing import Sequence, Optional, Callable


def get_bucketize_fn(boundaries: Sequence, log_base: Optional[int] = None) -> Callable:
    """
    获取分桶函数
    :param boundaries: 分桶边界(左闭右开)，桶的数量等于边界+1（两侧开区间）
    :param log_base: 如果为None，线性分桶；如果大于1，取对数后分桶
    :return: 能够用于map参数的函数
    """

    def bucketize_fn(value) -> int:
        """要返回的函数"""
        if log_base:
            assert log_base > 1
            value = math.log(value, log_base)
        category = 0
        for boundary in boundaries:
            if value < boundary:
                break
            category += 1
        return category

    return bucketize_fn
