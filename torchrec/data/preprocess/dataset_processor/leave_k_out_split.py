"""
留下最后K个划分
"""
import logging
from typing import Set, List, Tuple

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from tqdm import tqdm

from torchrec.data.preprocess.dataset_processor.sequential_split import __get_warm_interaction_df
from torchrec.utils.const import *
from torchrec.utils.system import check_dir_and_mkdir


def leave_k_out_split(dataset_name: str, warm_n: int, k: int):
    """
    留下最后K个划分，只保留暖用户，验证集与测试集按照k大小分别从每个用户最后的正向交互中划分，保存index信息
    :param dataset_name: 数据集名称
    :param warm_n: 暖用户限制
    :param k: 验证集/测试集中正例个数
    """
    dataset_dir = os.path.join(DATASET_DIR, dataset_name)
    split_index_dir = os.path.join(dataset_dir, SPLIT_INDEX_DIR)
    split_name = LEAVE_K_OUT_SPLIT_NAME_TEMPLATE % (warm_n, k)
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
    train_uid_array: ndarray = interaction_df[UID].unique()
    train_user_n = len(train_uid_array)

    logging.info(f'获取正向交互次数大于等于{warm_n + 2 * k}的用户数量...')  # 训练集必须有正例
    vt_uid_set: Set[int] = set(__get_warm_interaction_df(interaction_df, warm_n + 2 * k)[UID].unique())
    vt_user_n = len(vt_uid_set)

    logging.info(f'训练集用户总数：{train_user_n}，验证测试集用户总数：{vt_user_n}')

    logging.info('拆分数据集...')
    vt_indexes = []
    for i in range(2):
        k_vt_index_list: List[ndarray] = list()
        for j in range(k):
            split_index_list: List[ndarray] = list()
            vt_index_list = list()
            for uid, group in tqdm(interaction_df.groupby(UID), desc='拆分数据集'):
                if uid not in vt_uid_set:
                    continue
                last_pos_index = group[group[LABEL] == 1].index[-1]
                last_index = group.index[-1]
                vt_index_list.append(last_pos_index)
                split_index_list.append(np.arange(last_pos_index, last_index + 1, dtype=np.int32))
            split_index: ndarray = np.concatenate(split_index_list)
            interaction_df.drop(index=split_index, inplace=True)
            k_vt_index_list.append(np.array(vt_index_list, dtype=np.int32))
        vt_indexes.append(np.concatenate(k_vt_index_list))

    train_index_array: ndarray = interaction_df.index.to_numpy().astype(np.int32)
    train_index_array.sort()
    test_index_array: ndarray = vt_indexes[0]
    test_index_array.sort()
    dev_index_array: ndarray = vt_indexes[1]
    dev_index_array.sort()

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


def check_leave_k_out_split(dataset_name: str) -> List[Tuple[int, int]]:
    """检查留下最后K个划分参数"""
    import re
    split_dir = os.path.join(DATASET_DIR, dataset_name, SPLIT_INDEX_DIR)
    train_var_set, dev_var_set, test_var_set = set(), set(), set()
    for (type, var_set) in [("train", train_var_set), ("dev", dev_var_set), ("test", test_var_set)]:
        pattern = re.compile(rf"^leave_k_out_(\d+)_(\d+).{type}_index.npy$")
        for filename in os.listdir(split_dir):
            match_result = pattern.match(filename)
            if match_result:
                var_set.add((int(match_result.group(1)), int(match_result.group(2))))
    var_list = sorted(list(train_var_set & dev_var_set & test_var_set))
    return var_list
