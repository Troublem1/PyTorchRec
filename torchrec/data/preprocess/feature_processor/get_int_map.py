"""
获取整数映射
"""
from typing import Union, Sequence, Mapping, AbstractSet, Dict, Any

from pandas import Series


def get_int_map(collection: Union[Sequence, Series, Mapping, AbstractSet], start: int = 0) -> Dict[Any, int]:
    """
    将输入序列的元素或者输入集合的元素或者输入映射的Key重新映射到整数
    :param collection: 输入序列/集合/映射
    :param start: 整数下标初始值，默认为0，如果后续需要pad可以设置为1
    :return: 映射字典
    """
    assert start >= 0, start
    keys = sorted(list(set(collection)))
    values = range(start, len(keys) + start)
    return dict(zip(keys, values))
