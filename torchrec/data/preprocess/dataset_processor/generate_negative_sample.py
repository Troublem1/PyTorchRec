"""
负采样模块
"""
import logging
import pickle as pkl
from typing import List, Set

import numpy as np
import pandas as pd
from numpy import ndarray
from numpy.random import default_rng
from pandas import DataFrame
from tqdm import tqdm

from torchrec.utils.const import *
from torchrec.utils.system import check_dir_and_mkdir


def generate_negative_sample(seed: int, dataset_name: str, sample_n: int) -> None:
    """
    为验证集与测试集负采样，输出ndarray矩阵，形状为(USER_NUM, SAMPLE_NEG)，按照UID排序
    :param seed: 随机数种子
    :param dataset_name: 划分后的数据集名称
    :param sample_n: 负采样数量
    """
    rng = default_rng(seed)
    dataset_dir = os.path.join(DATASET_DIR, dataset_name)
    logging.info(f'读取数据集{dataset_name}...')
    interaction_df: DataFrame = pd.read_pickle(os.path.join(dataset_dir, BASE_INTERACTION_PKL))
    user_df: DataFrame = pd.read_pickle(os.path.join(dataset_dir, USER_PKL))
    item_df: DataFrame = pd.read_pickle(os.path.join(dataset_dir, ITEM_PKL))
    logging.info('获得验证集用户集用户ID列表...')
    uid_list: ndarray = interaction_df[UID].unique()
    assert len(uid_list) == len(user_df[UID].unique())
    logging.info('用户数：%d' % len(uid_list))
    logging.info('获得所有交互过的物品ID集合...')
    iid_list: ndarray = interaction_df[IID].unique()
    assert len(iid_list) == len(item_df[IID].unique())
    max_iid_index = len(iid_list)
    logging.info(f'交互过的物品数：{max_iid_index}')

    logging.info('读入交互历史统计信息...')
    statistic_dir = os.path.join(dataset_dir, STATISTIC_DIR)
    with open(os.path.join(statistic_dir, USER_POS_HIS_SET_DICT_PKL), 'rb') as user_pos_his_set_dict_pkl:
        user_pos_his_set_dict = pkl.load(user_pos_his_set_dict_pkl)

    validation_neg_sample_iid_list: List[ndarray] = []
    test_neg_sample_iid_list: List[ndarray] = []

    logging.info('负采样...')
    for uid in tqdm(uid_list):
        # 该用户正向交互过的物品 ID 集合
        inter_iid_set = user_pos_his_set_dict[uid]
        # 确保剩余物品数量足够
        assert max_iid_index - len(inter_iid_set) >= sample_n * 2
        # 采样
        sample_iid_set: Set[int] = set()
        for i in range(sample_n * 2):
            iid: int = iid_list[rng.integers(max_iid_index)]
            while iid in inter_iid_set or iid in sample_iid_set:
                iid: int = iid_list[rng.integers(max_iid_index)]
            sample_iid_set.add(iid)
        sample_iid_list: ndarray = np.array(sorted(list(sample_iid_set))).astype(np.int32)
        rng.shuffle(sample_iid_list)
        validation_neg_sample_iid_list.append(sample_iid_list[:sample_n])
        test_neg_sample_iid_list.append(sample_iid_list[sample_n:])
    validation_neg_sample_iid_array: ndarray = np.vstack(validation_neg_sample_iid_list)
    test_neg_sample_iid_array: ndarray = np.vstack(test_neg_sample_iid_list)

    logging.info('输出负采样结果...')
    negsam_dir = os.path.join(dataset_dir, NEGATIVE_SAMPLE_DIR)
    check_dir_and_mkdir(negsam_dir)

    assert validation_neg_sample_iid_array.dtype == np.int32, validation_neg_sample_iid_array.dtype
    np.save(os.path.join(negsam_dir, VALIDATION_NEG_NPY_TEMPLATE % sample_n), validation_neg_sample_iid_array)
    # noinspection PyTypeChecker
    np.savetxt(os.path.join(negsam_dir, VALIDATION_NEG_CSV_TEMPLATE % sample_n), validation_neg_sample_iid_array,
               delimiter=SEP, fmt='%d')

    assert test_neg_sample_iid_array.dtype == np.int32, test_neg_sample_iid_array.dtype
    np.save(os.path.join(negsam_dir, TEST_NEG_NPY_TEMPLATE % sample_n), test_neg_sample_iid_array)
    # noinspection PyTypeChecker
    np.savetxt(os.path.join(negsam_dir, TEST_NEG_CSV_TEMPLATE % sample_n), test_neg_sample_iid_array,
               delimiter=SEP, fmt='%d')
