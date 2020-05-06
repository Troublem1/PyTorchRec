"""
生成每条交互记录前k个历史信息
"""
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from tqdm import tqdm

from torchrec.utils.const import *
from torchrec.utils.system import check_dir_and_mkdir


def pad_or_cut_array(array: ndarray, array_len: int, pad: int = 0) -> ndarray:
    """
    将 ndarray 填充（后侧补指定值）或者裁剪（从后侧）到指定长度
    """
    if len(array) < array_len:
        if pad == 0:
            pad = np.zeros(array_len - len(array), dtype=array.dtype)
        else:
            pad = np.full(shape=array_len - len(array), fill_value=pad, dtype=array.dtype)
        return np.concatenate([array, pad])
    if len(array) > array_len:
        return array[-array_len:]
    return array


def create_last_k_history(dataset_name: str, k: int) -> None:
    """
    生成每条交互记录前k个历史信息，直接输出ndarray矩阵，形状为(DF_LEN, k + 1)，按照数据集顺序排序，第一列是历史信息长度，后面是历史信息
    :param dataset_name: 数据集名称
    :param k: 历史长度（填充/截取）
    """
    dataset_dir = os.path.join(DATASET_DIR, dataset_name)
    history_dir = os.path.join(dataset_dir, HISTORY_DIR)
    check_dir_and_mkdir(history_dir)

    logging.info(f'读取数据集{dataset_name}...')
    interaction_df: DataFrame = pd.read_pickle(os.path.join(dataset_dir, BASE_INTERACTION_PKL))

    need_neg = (interaction_df[LABEL] == 0).any()

    his_dict: Dict[int, List[int]] = dict()
    neg_dict: Optional[Dict[int, List[int]]] = None
    his_array_list: List[ndarray] = []
    neg_array_list: Optional[List[ndarray]] = None
    if need_neg:
        neg_dict = dict()
        neg_array_list: List[ndarray] = []
    for row in tqdm(interaction_df.itertuples(index=False), total=len(interaction_df),
                    desc=f'生成历史信息，长度为{k}：'):
        uid: int = getattr(row, UID)
        iid: int = getattr(row, IID)
        label: int = getattr(row, LABEL)
        # 正向交互
        user_his_list: List[int] = his_dict.setdefault(uid, [])
        user_his_len: int = min(len(user_his_list), k)
        user_his_array: ndarray = pad_or_cut_array(np.array([user_his_len] + user_his_list[-k:],
                                                            dtype=np.int32), k + 1)
        his_array_list.append(user_his_array)
        if label > 0:
            user_his_list.append(iid)
        if need_neg:
            # 负向交互
            user_neg_list: List[int] = neg_dict.setdefault(uid, [])
            user_neg_len: int = min(len(user_neg_list), k)
            user_neg_array: ndarray = pad_or_cut_array(np.array([user_neg_len] + user_neg_list[-k:],
                                                                dtype=np.int32), k + 1)
            neg_array_list.append(user_neg_array)
            if label <= 0:
                user_neg_list.append(iid)
    all_his_array: ndarray = np.vstack(his_array_list)
    assert all_his_array.dtype == np.int32, all_his_array.dtype
    all_neg_array: Optional[ndarray] = None
    if need_neg:
        all_neg_array = np.vstack(neg_array_list)
        assert all_neg_array.dtype == np.int32, all_neg_array.dtype
    np.save(os.path.join(history_dir, POS_HIS_NPY_TEMPLATE % k), all_his_array)
    # noinspection PyTypeChecker
    np.savetxt(os.path.join(history_dir, POS_HIS_CSV_TEMPLATE % k), all_his_array, delimiter=SEP, fmt='%d')
    if need_neg:
        np.save(os.path.join(history_dir, NEG_HIS_NPY_TEMPLATE % k), all_neg_array)
        # noinspection PyTypeChecker
        np.savetxt(os.path.join(history_dir, NEG_HIS_CSV_TEMPLATE % k), all_neg_array, delimiter=SEP, fmt='%d')
