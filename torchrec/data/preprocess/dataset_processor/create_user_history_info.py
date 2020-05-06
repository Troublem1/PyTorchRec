"""
生成以用户为单位的历史统计信息
"""
import logging
import pickle as pkl
from typing import Dict, Set

import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

from torchrec.utils.const import *
from torchrec.utils.system import check_dir_and_mkdir


def create_user_history_info(dataset_name: str) -> None:
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
        for row in tqdm(interaction_df.itertuples(index=False), total=len(interaction_df),
                        desc=f'生成用户{interaction_type}历史交互集合：'):
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
    interaction_df = pd.read_pickle(os.path.join(dataset_dir, BASE_INTERACTION_PKL))

    statistic_dir = os.path.join(dataset_dir, STATISTIC_DIR)
    check_dir_and_mkdir(statistic_dir)

    logging.info("生成用户正向交互历史统计信息...")
    user_pos_his_set_dict = create_user_his_set_dict(interaction_df, POSITIVE)
    with open(os.path.join(statistic_dir, USER_POS_HIS_SET_DICT_PKL), 'wb') as user_pos_his_set_dict_pkl:
        pkl.dump(user_pos_his_set_dict, user_pos_his_set_dict_pkl, pkl.HIGHEST_PROTOCOL)

    if (interaction_df[LABEL] == 0).any():
        logging.info("生成用户负向交互历史统计信息...")
        user_neg_his_set_dict = create_user_his_set_dict(interaction_df, NEGATIVE)
        with open(os.path.join(statistic_dir, USER_NEG_HIS_SET_DICT_PKL), 'wb') as user_neg_his_set_dict_pkl:
            pkl.dump(user_neg_his_set_dict, user_neg_his_set_dict_pkl, pkl.HIGHEST_PROTOCOL)


def check_user_history_info(dataset_name: str) -> bool:
    """检查是否存在以用户为单位的历史统计信息"""
    statistic_dir = os.path.join(DATASET_DIR, dataset_name, STATISTIC_DIR)
    return os.path.exists(os.path.join(statistic_dir, USER_POS_HIS_SET_DICT_PKL))
