import logging

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from tqdm import tqdm
from typing import Dict, List

from torchrec.data.process.interaction_history_list import pad_or_cut_array
from torchrec.utils.const import *
from torchrec.utils.system import check_dir_and_mkdir


def generate_rl_next_item_sample(dataset_name: str, sample_len: int) -> None:
    """
    生成Value-Based RL模型所需要的候选集，这里的采样方式是以当前为基准，前max_size/2个正例与后max_size/2个正例，只需要对训练集生成
    直接输出ndarray矩阵，形状为(DF_LEN, SAMPLE_LEN)，按照数据集顺序排序
    Args:
        dataset_name: 划分后的数据集名称
        sample_len: 最大采样数（填充自身，不能填充0/截取）
    """
    l_size = sample_len // 2 if sample_len % 2 == 0 else (sample_len - 1) // 2
    r_size = sample_len // 2 if sample_len % 2 == 0 else (sample_len + 1) // 2
    dataset_dir = os.path.join(DATASET_DIR, dataset_name)
    rl_sample_dir = os.path.join(dataset_dir, RL_SAMPLE_DIR)
    check_dir_and_mkdir(rl_sample_dir)

    logging.info(f'读取数据集{dataset_name}...')
    interaction_df: DataFrame = pd.read_feather(os.path.join(dataset_dir, BASE_INTERACTION_FEATHER))

    train_pos_his_dict: Dict[int, ndarray] = {}
    for uid, uid_df in interaction_df[interaction_df[LABEL] == 1].groupby(UID):
        train_pos_his = uid_df[IID].values
        train_pos_his_dict[uid] = train_pos_his

    for uid in interaction_df[UID].unique():
        if uid not in train_pos_his_dict:
            train_pos_his_dict[uid] = np.array([0], dtype=np.int32)

    print('生成RL相关数据...')
    rl_sample_list: List[ndarray] = []
    for uid, user_df in tqdm(interaction_df.groupby(UID)):
        pos_his_list: ndarray = train_pos_his_dict[uid]
        pos_his_list_len = len(pos_his_list)
        his_list_pos: int = 0
        for row in user_df.itertuples():
            label: int = getattr(row, LABEL)
            if label == 1:
                his_list_pos += 1
            sample_list = pos_his_list[max(0, his_list_pos - l_size):min(his_list_pos + r_size, pos_his_list_len)]
            # sample_list = pos_his_list[max(0, his_list_pos - sample_len):his_list_pos]
            rl_sample_list.append(pad_or_cut_array(sample_list, sample_len, pad=sample_list[-1]))
    rl_sample_array = np.vstack(rl_sample_list)
    assert rl_sample_array.dtype == np.int32, rl_sample_array.dtype
    np.save(os.path.join(dataset_dir, RL_SAMPLE_DIR, RL_SAMPLE_NPY_TEMPLATE % sample_len), rl_sample_array)
    # noinspection PyTypeChecker
    np.savetxt(os.path.join(dataset_dir, RL_SAMPLE_DIR, RL_SAMPLE_CSV_TEMPLATE % sample_len), rl_sample_array,
               delimiter=SEP, fmt='%d')

# todo check函数
