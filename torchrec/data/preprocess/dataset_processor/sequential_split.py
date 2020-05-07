"""
顺序划分
"""
import logging
import math
from typing import Set, List, Tuple

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from tqdm import tqdm

from torchrec.utils.const import *
from torchrec.utils.system import check_dir_and_mkdir


def __get_warm_interaction_df(interaction_df: DataFrame, warm_n: int) -> DataFrame:
    """删除掉冷用户"""
    logging.info(f'获取正向交互次数大于等于{warm_n}的用户交互数据...')
    pos_df = interaction_df[interaction_df[LABEL] == 1]
    uid_array: ndarray = pos_df[pos_df.groupby(UID)[UID].transform('count').ge(warm_n)][UID].unique()
    uid_set: Set[int] = set(uid_array)
    new_interaction_df = interaction_df[interaction_df[UID].isin(uid_set)]
    logging.info(f'正向交互次数大于等于{warm_n}的用户交互总数：{len(interaction_df)}')
    logging.info(f'正向交互次数大于等于{warm_n}的用户总数：{len(uid_array)}')
    return new_interaction_df


def sequential_split(dataset_name: str, warm_n: int, vt_ratio: float) -> None:
    """
    顺序划分，只保留暖用户，训练集、验证集与测试集按照比例分别从每个用户中划分，保存index信息
    :param dataset_name: 数据集名称
    :param warm_n: 暖用户限制
    :param vt_ratio: 验证集/测试集比例
    """
    dataset_dir = os.path.join(DATASET_DIR, dataset_name)
    split_index_dir = os.path.join(dataset_dir, SPLIT_INDEX_DIR)
    split_name = SEQUENTIAL_SPLIT_NAME_TEMPLATE % (warm_n, vt_ratio)
    check_dir_and_mkdir(split_index_dir)

    interaction_pkl = os.path.join(dataset_dir, BASE_INTERACTION_PKL)

    logging.info('读取数据集交互数据...')
    interaction_df: DataFrame = pd.read_pickle(interaction_pkl)
    logging.info(f'交互总数：{len(interaction_df)}')
    logging.info(f'用户总数：{len(interaction_df[UID].unique())}')
    if warm_n == 0:
        warm_n = 1  # 至少有一个正向交互信息
    assert warm_n > 0, warm_n
    interaction_df = __get_warm_interaction_df(interaction_df, warm_n)

    train_index_array_list: List[ndarray] = list()
    dev_index_array_list: List[ndarray] = list()
    test_index_array_list: List[ndarray] = list()

    for uid, user_df in tqdm(interaction_df.groupby(UID)):
        interaction_num = len(user_df)
        vt_num = int(math.floor(interaction_num * vt_ratio))
        train_num = interaction_num - 2 * vt_num
        index_array = user_df.index.to_numpy().astype(np.int32)
        train_index_array_list.append(index_array[:train_num])
        if vt_num > 0:
            dev_index_array_list.append(index_array[train_num: train_num + vt_num])
            test_index_array_list.append(index_array[train_num + vt_num: train_num + 2 * vt_num])

    train_index_array: ndarray = np.concatenate(train_index_array_list)
    dev_index_array: ndarray = np.concatenate(dev_index_array_list)
    test_index_array: ndarray = np.concatenate(test_index_array_list)

    logging.info(f"训练集大小：{len(train_index_array)}")
    logging.info(f"验证集大小：{len(dev_index_array)}")
    logging.info(f"测试集大小：{len(test_index_array)}")
    print(train_index_array)
    print(dev_index_array)
    print(test_index_array)

    logging.info('保存生成的划分索引信息...')
    for array, npy_template, csv_template in [
        (train_index_array, TRAIN_INDEX_NPY_TEMPLATE, TRAIN_INDEX_CSV_TEMPLATE),
        (dev_index_array, DEV_INDEX_NPY_TEMPLATE, DEV_INDEX_CSV_TEMPLATE),
        (test_index_array, TEST_INDEX_NPY_TEMPLATE, TEST_INDEX_CSV_TEMPLATE)
    ]:
        assert array.dtype == np.int32, array.dtype
        np.save(os.path.join(split_index_dir, npy_template % split_name), array)
        # noinspection PyTypeChecker
        np.savetxt(os.path.join(split_index_dir, csv_template % split_name),
                   array, delimiter=SEP, fmt='%d')


def check_sequential_split(dataset_name: str) -> List[Tuple[int, float]]:
    """检查顺序划分参数"""
    import re
    split_dir = os.path.join(DATASET_DIR, dataset_name, SPLIT_INDEX_DIR)
    train_var_set, dev_var_set, test_var_set = set(), set(), set()
    for (type, var_set) in [("train", train_var_set), ("dev", dev_var_set), ("test", test_var_set)]:
        pattern = re.compile(rf"^seq_split_(\d+)_(0.\d+).{type}_index.npy$")
        for filename in os.listdir(split_dir):
            match_result = pattern.match(filename)
            if match_result:
                var_set.add((int(match_result.group(1)), float(match_result.group(2))))
    var_list = sorted(list(train_var_set & dev_var_set & test_var_set))
    return var_list
