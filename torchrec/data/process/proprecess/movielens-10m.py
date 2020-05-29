"""
MovieLens-1M数据集预处理
"""
import logging
import pickle as pkl

import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Callable

from torchrec.data.dataset import DatasetDescription
from torchrec.data.process import generate_interaction_history_list, generate_interaction_next_state_list, \
    generate_rl_next_item_sample
from torchrec.data.process import generate_leave_k_out_split
from torchrec.data.process import generate_sequential_split
from torchrec.data.process import generate_vt_negative_sample
from torchrec.data.process.feature_process import get_int_map
from torchrec.feature_column import CategoricalColumnWithIdentity
from torchrec.utils.const import *
from torchrec.utils.system import init_console_logger, check_dir_and_mkdir

pd.set_option('display.max_colwidth', 20)
pd.set_option('display.max_columns', None)

SEED = 2020
RAW_DATA_NAME = 'MovieLens-10M'
RAW_INTERACTION_NAME = 'ratings.dat'


def format_data(dataset_name: str, rank_to_label: Callable, info: str) -> None:
    """
    过滤并格式化原始数据集中用户ID、物品ID、评分、标签、时间戳五项基本信息、上下文/物品/用户特征信息（可选），并统计相关信息
    标签是二值化的，评分如果不存在，与标签一致
    """
    description = DatasetDescription(info)

    logging.info('读入评分数据...')
    interaction_df: DataFrame = pd.read_csv(os.path.join(RAW_DATA_DIR, RAW_DATA_NAME, RAW_INTERACTION_NAME), sep='::',
                                            header=None, engine='python', dtype=np.int32)
    interaction_df.columns = [UID, IID, RATE, TIME]
    assert not any(interaction_df.isnull().any()), interaction_df.isnull().any()
    interaction_df[UID] = interaction_df[UID].map(get_int_map(interaction_df[UID], start=1)).astype(np.int32)
    interaction_df[IID] = interaction_df[IID].map(get_int_map(interaction_df[IID], start=1)).astype(np.int32)
    interaction_df[LABEL] = interaction_df[RATE].map(rank_to_label).astype(np.int32)
    interaction_df = interaction_df[[UID, IID, RATE, LABEL, TIME]]
    logging.debug('排序评分数据...')
    interaction_df.drop_duplicates([UID, IID], inplace=True)
    interaction_df.sort_values(by=[UID, TIME], kind='mergesort', inplace=True)
    interaction_df.reset_index(inplace=True, drop=True)
    logging.debug(interaction_df)
    logging.debug(interaction_df.info())
    # 基本特征信息
    description.uid_column = CategoricalColumnWithIdentity.from_series(
        feature_name=UID,
        series=interaction_df[UID])
    description.iid_column = CategoricalColumnWithIdentity.from_series(
        feature_name=IID,
        series=interaction_df[IID])
    description.rate_column = CategoricalColumnWithIdentity.from_series(
        feature_name=RATE,
        series=interaction_df[RATE])
    description.label_column = CategoricalColumnWithIdentity.from_series(
        feature_name=LABEL,
        series=interaction_df[LABEL])
    description.time_column = CategoricalColumnWithIdentity.from_series(
        feature_name=TIME,
        series=interaction_df[TIME])
    # 统计交互信息
    description.get_user_interaction_statistic(interaction_df)

    user_df = interaction_df[[UID]].copy(deep=True)
    user_df.sort_values(UID, kind='mergesort', inplace=True)
    user_df.drop_duplicates(UID, inplace=True)
    user_df.reset_index(inplace=True, drop=True)
    item_df = interaction_df[[IID]].copy(deep=True)
    item_df.sort_values(IID, kind='mergesort', inplace=True)
    item_df.drop_duplicates(IID, inplace=True)
    item_df.reset_index(inplace=True, drop=True)

    dataset_dir = os.path.join(DATASET_DIR, dataset_name)
    check_dir_and_mkdir(dataset_dir)

    logging.info('保存数据...')
    assert (interaction_df.dtypes == np.int32).all(), interaction_df.dtypes
    base_interaction_df = interaction_df
    base_interaction_df.to_csv(os.path.join(dataset_dir, BASE_INTERACTION_CSV), index=False, sep=SEP)
    base_interaction_df.to_feather(os.path.join(dataset_dir, BASE_INTERACTION_FEATHER))
    interaction_df.to_csv(os.path.join(dataset_dir, INTERACTION_CSV), index=False, sep=SEP)
    interaction_df.to_feather(os.path.join(dataset_dir, INTERACTION_FEATHER))
    item_df.to_csv(os.path.join(dataset_dir, ITEM_CSV), index=False, sep=SEP)
    item_df.to_feather(os.path.join(dataset_dir, ITEM_FEATHER))
    user_df.to_csv(os.path.join(dataset_dir, USER_CSV), index=False, sep=SEP)
    user_df.to_feather(os.path.join(dataset_dir, USER_FEATHER))

    logging.info('保存数据集信息...')
    with open(os.path.join(dataset_dir, DESCRIPTION_PKL), "wb") as description_pkl:
        pkl.dump(description, description_pkl, pkl.HIGHEST_PROTOCOL)
    description.to_txt_file(os.path.join(dataset_dir, DESCRIPTION_TXT))


if __name__ == '__main__':
    init_console_logger()

    dataset_name = RAW_DATA_NAME + "-PN"
    format_data(
        dataset_name=dataset_name,
        rank_to_label=lambda x: 1 if x >= 4.0 else 0,
        info="正负例化的MovieLens-10M数据集，评分为[4,5]为正例，评分为[1,4)为负例"
    )
    generate_sequential_split(dataset_name=dataset_name, warm_n=5, vt_ratio=0.1)
    generate_leave_k_out_split(dataset_name=dataset_name, warm_n=5, k=1)
    for i in range(5):
        generate_vt_negative_sample(seed=SEED + i, dataset_name=dataset_name, sample_n=99)
    generate_interaction_history_list(dataset_name=dataset_name, k=10)
    generate_interaction_next_state_list(dataset_name=dataset_name, k=10)
    generate_rl_next_item_sample(dataset_name=dataset_name, sample_len=32)

    # dataset_name = RAW_DATA_NAME + "-P"
    # format_data(
    #     dataset_name=dataset_name,
    #     rank_to_label={1: 1, 2: 1, 3: 1, 4: 1, 5: 1},
    #     info="全部视为正例的MovieLens-1M数据集，评分为1/2/3/4/5为正例"
    # )
    # generate_sequential_split(dataset_name=dataset_name, warm_n=5, vt_ratio=0.1)
    # generate_leave_k_out_split(dataset_name=dataset_name, warm_n=5, k=1)
    # generate_vt_negative_sample(seed=SEED, dataset_name=dataset_name, sample_n=99)
    # generate_interaction_history_list(dataset_name=dataset_name, k=10)
