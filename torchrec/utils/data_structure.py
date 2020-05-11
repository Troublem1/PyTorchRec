"""
与数据结构相关的辅助功能
"""
from typing import Callable

import numpy as np
from torch import Tensor


def map_structure(func: Callable, structure):
    """
    对数据结构的每一个元素进行遍历，并可以递归执行

    目前支持的数据类型：
    List: 对每个元素执行函数
    Dict：对每个值执行函数
    其他：视为一个值，直接执行
    # todo：其他类型的支持
    """
    if not callable(func):
        raise TypeError("func must be callable, got: %s" % func)

    if isinstance(structure, list):
        return [map_structure(func, item) for item in structure]

    if isinstance(structure, dict):
        return {key: map_structure(func, structure[key]) for key in structure}

    return func(structure)


def tensor_to_numpy_or_python_type(structure):
    """将数据结构中的tensor转化为ndarray或者标量，可以递归转化"""

    def _to_numpy_or_python_type(t):
        if isinstance(t, Tensor):
            x = t.numpy()
            return x.item() if np.ndim(x) == 0 else x
        return t

    return map_structure(_to_numpy_or_python_type, structure)
