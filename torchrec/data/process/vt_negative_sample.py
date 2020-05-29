"""
负采样模块
"""
import logging
import pickle as pkl

import numpy as np
import pandas as pd
from numpy import ndarray
from numpy.random import default_rng
from pandas import DataFrame
from tqdm import tqdm
from typing import List, Set, Dict

from torchrec.utils.const import *
from torchrec.utils.system import check_dir_and_mkdir


def __generate_user_history_statistic(dataset_name: str) -> None:
    """
    生成以用户为单位的历史统计信息
    :param dataset_name: 数据集名称
    """

    def create_user_his_set_dict(interaction_df: DataFrame, interaction_type: str) -> Dict[int, Set[int]]:
        """
        以用户为单位统计数据帧中指定类型的交互物品ID列表
        :param interaction_df: 数据帧，通常是数据集交互信息
        :param interaction_type: 交互类型，'all'：所有，'positive'：正向，'negative'：负向
        :return 含有UID与IIDS两列的数据帧
        """
        user_his_set_dict = dict()
        for row in tqdm(interaction_df.itertuples(index=False), total=len(interaction_df)):
            uid: int = getattr(row, UID)
            iid: int = getattr(row, IID)
            label: int = getattr(row, LABEL)
            user_his_set = user_his_set_dict.setdefault(uid, set())
            if label == 1 and (interaction_type == ALL or interaction_type == POSITIVE):
                user_his_set.add(iid)
            if label == 0 and (interaction_type == ALL or interaction_type == NEGATIVE):
                user_his_set.add(iid)
        return user_his_set_dict

    dataset_dir = os.path.join(DATASET_DIR, dataset_name)
    logging.info(f'读取数据集{dataset_name}...')
    interaction_df = pd.read_feather(os.path.join(dataset_dir, BASE_INTERACTION_FEATHER))

    neg_sample_dir = os.path.join(dataset_dir, NEGATIVE_SAMPLE_DIR)
    check_dir_and_mkdir(neg_sample_dir)

    logging.info("生成用户正向交互历史统计信息...")
    user_pos_his_set_dict = create_user_his_set_dict(interaction_df, POSITIVE)
    with open(os.path.join(neg_sample_dir, USER_POS_HIS_SET_DICT_PKL), 'wb') as user_pos_his_set_dict_pkl:
        pkl.dump(user_pos_his_set_dict, user_pos_his_set_dict_pkl, pkl.HIGHEST_PROTOCOL)


def generate_vt_negative_sample(seed: int, dataset_name: str, sample_n: int) -> None:
    """
    为验证集与测试集负采样，输出ndarray矩阵，形状为(USER_NUM, SAMPLE_NEG)，按照UID排序
    :param seed: 随机数种子
    :param dataset_name: 划分后的数据集名称
    :param sample_n: 负采样数量
    """
    dataset_dir = os.path.join(DATASET_DIR, dataset_name)
    neg_sample_dir = os.path.join(dataset_dir, NEGATIVE_SAMPLE_DIR)
    check_dir_and_mkdir(neg_sample_dir)

    rng = default_rng(seed)
    logging.info(f'读取数据集{dataset_name}...')
    interaction_df: DataFrame = pd.read_feather(os.path.join(dataset_dir, BASE_INTERACTION_FEATHER))
    logging.info('获得验证集用户集用户ID列表...')
    uid_list: ndarray = interaction_df[UID].unique()
    logging.info('用户数：%d' % len(uid_list))
    logging.info('获得所有交互过的物品ID集合...')
    min_iid = 1  # 0: PAD
    max_iid = interaction_df[IID].max() + 1
    logging.info(f'物品ID范围：[{min_iid}, {max_iid})')

    logging.info('读入交互历史统计信息...')
    user_pos_his_set_dict_filename = os.path.join(neg_sample_dir, USER_POS_HIS_SET_DICT_PKL)
    if not os.path.exists(user_pos_his_set_dict_filename):
        __generate_user_history_statistic(dataset_name)
    with open(os.path.join(user_pos_his_set_dict_filename), 'rb') as user_pos_his_set_dict_pkl:
        user_pos_his_set_dict = pkl.load(user_pos_his_set_dict_pkl)

    validation_neg_sample_iid_list: List[ndarray] = []
    test_neg_sample_iid_list: List[ndarray] = []

    logging.info('负采样...')
    for uid in tqdm(uid_list):
        # 该用户正向交互过的物品 ID 集合
        inter_iid_set = user_pos_his_set_dict[uid]
        # 确保剩余物品数量足够
        assert max_iid - min_iid - len(inter_iid_set) >= sample_n * 2
        # 采样
        sample_iid_set: Set[int] = set()
        for i in range(sample_n * 2):
            iid: int = rng.integers(min_iid, max_iid)
            while iid in inter_iid_set or iid in sample_iid_set:
                iid: int = rng.integers(min_iid, max_iid)
            sample_iid_set.add(iid)
        sample_iid_list: ndarray = np.array(sorted(list(sample_iid_set))).astype(np.int32)
        rng.shuffle(sample_iid_list)
        validation_neg_sample_iid_list.append(sample_iid_list[:sample_n])
        test_neg_sample_iid_list.append(sample_iid_list[sample_n:])
    validation_neg_sample_iid_array: ndarray = np.vstack(validation_neg_sample_iid_list)
    test_neg_sample_iid_array: ndarray = np.vstack(test_neg_sample_iid_list)

    logging.info('输出负采样结果...')

    assert validation_neg_sample_iid_array.dtype == np.int32, validation_neg_sample_iid_array.dtype
    print(validation_neg_sample_iid_array.shape)
    np.save(os.path.join(neg_sample_dir, DEV_NEG_NPY_TEMPLATE % (seed, sample_n)), validation_neg_sample_iid_array)
    # noinspection PyTypeChecker
    np.savetxt(os.path.join(neg_sample_dir, DEV_NEG_CSV_TEMPLATE % (seed, sample_n)), validation_neg_sample_iid_array,
               delimiter=SEP, fmt='%d')

    assert test_neg_sample_iid_array.dtype == np.int32, test_neg_sample_iid_array.dtype
    np.save(os.path.join(neg_sample_dir, TEST_NEG_NPY_TEMPLATE % (seed, sample_n)), test_neg_sample_iid_array)
    # noinspection PyTypeChecker
    np.savetxt(os.path.join(neg_sample_dir, TEST_NEG_CSV_TEMPLATE % (seed, sample_n)), test_neg_sample_iid_array,
               delimiter=SEP, fmt='%d')


def check_vt_negative_sample(dataset_name: str) -> List[int]:
    """检查验证集与测试集负采样的采样长度列表"""
    import re
    sample_dir = os.path.join(DATASET_DIR, dataset_name, NEGATIVE_SAMPLE_DIR)
    test_len_set, dev_len_set = set(), set()
    for (type, len_set) in [("test", test_len_set), ("dev", dev_len_set)]:
        pattern = re.compile(rf"^{type}_neg_(\d+).npy$")
        for filename in os.listdir(sample_dir):
            match_result = pattern.match(filename)
            if match_result:
                len_set.add(int(match_result.group(1)))
    len_list = sorted(list(test_len_set & dev_len_set))
    return len_list
