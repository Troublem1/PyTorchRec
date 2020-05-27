"""
生成RL的下一次状态，即每条交互记录前（包括自身）k个信息
"""
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from tqdm import tqdm

from torchrec.data.process.interaction_history_list import pad_or_cut_array
from torchrec.utils.const import *
from torchrec.utils.system import check_dir_and_mkdir


def generate_interaction_next_state_list(dataset_name: str, k: int) -> None:
    """
    生成每条交互记录前k个历史信息，直接输出ndarray矩阵，形状为(DF_LEN, k + 1)，按照数据集顺序排序，第一列是历史信息长度，后面是历史信息
    :param dataset_name: 数据集名称
    :param k: 历史长度（填充/截取）
    """
    dataset_dir = os.path.join(DATASET_DIR, dataset_name)
    next_state_dir = os.path.join(dataset_dir, NEXT_STATE_DIR)
    check_dir_and_mkdir(next_state_dir)

    logging.info(f'读取数据集{dataset_name}...')
    interaction_df: DataFrame = pd.read_feather(os.path.join(dataset_dir, BASE_INTERACTION_FEATHER))

    need_neg = (interaction_df[LABEL] == 0).any()

    next_state_dict: Dict[int, List[int]] = dict()
    neg_dict: Optional[Dict[int, List[int]]] = None
    next_state_array_list: List[ndarray] = []
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
        user_next_state_list: List[int] = next_state_dict.setdefault(uid, [])
        if label > 0:
            user_next_state_list.append(iid)
        user_next_state_len: int = min(len(user_next_state_list), k)
        user_next_state_array: ndarray = pad_or_cut_array(np.array([user_next_state_len] + user_next_state_list[-k:],
                                                                   dtype=np.int32), k + 1)
        next_state_array_list.append(user_next_state_array)
        if need_neg:
            # 负向交互
            user_neg_list: List[int] = neg_dict.setdefault(uid, [])
            if label <= 0:
                user_neg_list.append(iid)
            user_neg_len: int = min(len(user_neg_list), k)
            user_neg_array: ndarray = pad_or_cut_array(np.array([user_neg_len] + user_neg_list[-k:],
                                                                dtype=np.int32), k + 1)
            neg_array_list.append(user_neg_array)
    all_next_state_array: ndarray = np.vstack(next_state_array_list)
    assert all_next_state_array.dtype == np.int32, all_next_state_array.dtype
    all_neg_array: Optional[ndarray] = None
    if need_neg:
        all_neg_array = np.vstack(neg_array_list)
        assert all_neg_array.dtype == np.int32, all_neg_array.dtype
    np.save(os.path.join(next_state_dir, POS_NEXT_STATE_NPY_TEMPLATE % k), all_next_state_array)
    # noinspection PyTypeChecker
    np.savetxt(os.path.join(next_state_dir, POS_NEXT_STATE_CSV_TEMPLATE % k),
               all_next_state_array, delimiter=SEP, fmt='%d')
    if need_neg:
        np.save(os.path.join(next_state_dir, NEG_NEXT_STATE_NPY_TEMPLATE % k), all_neg_array)
        # noinspection PyTypeChecker
        np.savetxt(os.path.join(next_state_dir, NEG_NEXT_STATE_CSV_TEMPLATE % k),
                   all_neg_array, delimiter=SEP, fmt='%d')


def check_interaction_next_state_list(dataset_name: str) -> List[int]:
    """检查每个已经生成过的历史信息的长度"""
    import re
    next_state_dir = os.path.join(DATASET_DIR, dataset_name, NEXT_STATE_DIR)
    pattern = re.compile(r"^pos_next_state_(\d+).npy$")
    len_list = list()
    for filename in os.listdir(next_state_dir):
        match_result = pattern.match(filename)
        if match_result:
            len_list.append(int(match_result.group(1)))
    len_list.sort()
    return len_list
