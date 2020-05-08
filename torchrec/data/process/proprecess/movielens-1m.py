"""
MovieLens-1M数据集预处理
"""
import logging
import pickle as pkl
from typing import Dict

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from torchrec.data.dataset import DatasetDescription
from torchrec.data.process import generate_interaction_history_list
from torchrec.data.process import generate_leave_k_out_split
from torchrec.data.process import generate_sequential_split
from torchrec.data.process import generate_vt_negative_sample
from torchrec.data.process.feature_process import get_int_map, get_bucketize_fn
from torchrec.feature_column import CategoricalColumnWithIdentity
from torchrec.utils.const import *
from torchrec.utils.system import init_console_logger, check_dir_and_mkdir

pd.set_option('display.max_colwidth', 20)
pd.set_option('display.max_columns', None)

SEED = 2020
RAW_DATA_NAME = 'MovieLens-1M'
RAW_INTERACTION_NAME = 'ratings.dat'
RAW_USER_NAME = 'users.dat'
RAW_ITEM_NAME = 'movies.dat'


def format_data(dataset_name: str, rank_to_label: Dict, info: str) -> None:
    """
    过滤并格式化原始数据集中用户ID、物品ID、评分、标签、时间戳五项基本信息、上下文/物品/用户特征信息（可选），并统计相关信息
    标签是二值化的，评分如果不存在，与标签一致
    """
    description = DatasetDescription(info)

    logging.info('读入用户数据...')
    U_AGE, U_GENDER, U_OCCUPATION = "u_c_age", "u_c_gender", "u_c_occupation"
    user_usecols = [0, 1, 2, 3]
    user_dtype_dict = {0: np.int32, 1: np.str, 2: np.int32, 3: np.int32}
    user_df: DataFrame = pd.read_csv(os.path.join(RAW_DATA_DIR, RAW_DATA_NAME, RAW_USER_NAME), sep='::', header=None,
                                     usecols=user_usecols, dtype=user_dtype_dict,
                                     engine='python')
    user_df.columns = [UID, U_GENDER, U_AGE, U_OCCUPATION]
    assert not any(user_df.isnull().any()), user_df.isnull().any()
    # 性别
    u_gender_int_map = {"M": 0, "F": 1}
    user_df[U_GENDER] = user_df[U_GENDER].map(u_gender_int_map).astype(np.int32)
    description.user_columns.append(CategoricalColumnWithIdentity.from_series(
        feature_name=U_GENDER,
        series=user_df[U_GENDER],
        other_info={INT_MAP: u_gender_int_map}
    ))
    # 年龄
    u_age_int_map = get_int_map(user_df[U_AGE])
    user_df[U_AGE] = user_df[U_AGE].map(u_age_int_map).astype(np.int32)
    description.user_columns.append(CategoricalColumnWithIdentity.from_series(
        feature_name=U_AGE,
        series=user_df[U_AGE],
        other_info={INT_MAP: u_age_int_map}
    ))
    # 职业
    description.user_columns.append(CategoricalColumnWithIdentity.from_series(
        feature_name=U_OCCUPATION,
        series=user_df[U_OCCUPATION]
    ))
    logging.debug(user_df)
    logging.debug(user_df.info())

    logging.info('读入物品数据...')
    item_usecols = [0, 1, 2]
    item_dtype_dict = {0: np.int32, 1: np.str, 2: np.str}
    item_df: DataFrame = pd.read_csv(os.path.join(RAW_DATA_DIR, RAW_DATA_NAME, RAW_ITEM_NAME), sep='::', header=None,
                                     usecols=item_usecols, dtype=item_dtype_dict, engine='python')
    item_df.columns = [IID, "i_c_year", "type"]
    # 年份
    item_df["i_c_year"] = item_df["i_c_year"].map(lambda s: int(s[-5:-1])).astype(np.int32)
    i_year_boundaries = [1940, 1950, 1960, 1970, 1980, 1985] + list(range(1990, int(item_df['i_c_year'].max() + 1)))
    item_df["i_c_year"] = item_df["i_c_year"].map(get_bucketize_fn(i_year_boundaries)).astype(np.int32)
    description.item_columns.append(CategoricalColumnWithIdentity.from_series(
        feature_name="i_c_year",
        series=item_df["i_c_year"],
        other_info={BUCKET_BOUNDARIES: i_year_boundaries}
    ))
    # 其他类型特征
    type_name_dict = {
        "Action": "i_c_action",
        "Adventure": "i_c_adventure",
        "Animation": "i_c_animation",
        "Children's": "i_c_children",
        "Comedy": "i_c_comedy",
        "Crime": "i_c_crime",
        "Documentary": "i_c_documentary",
        "Drama": "i_c_drama",
        "Fantasy": "i_c_fantasy",
        "Film-Noir": "i_c_file_noir",
        "Horror": "i_c_horror",
        "Musical": "i_c_musical",
        "Mystery": "i_c_mystery",
        "Romance": "i_c_romance",
        "Sci-Fi": "i_c_sci_fi",
        "Thriller": "i_c_thriller",
        "War": "i_c_war",
        "Western": "i_western"
    }
    type_data_dict = {}
    for key in type_name_dict:
        type_data_dict[type_name_dict[key]] = []
    for types in item_df["type"]:
        type_list = types.split("|")
        for key in type_name_dict:
            type_name = type_name_dict[key]
            type_data_dict[type_name].append(1 if key in type_list else 0)
    item_df.drop(columns="type", inplace=True)
    for key in type_data_dict:
        item_df[key] = Series(type_data_dict[key]).astype(np.int32)
        description.item_columns.append(CategoricalColumnWithIdentity.from_series(
            feature_name=key,
            series=item_df[key]
        ))
    assert not any(item_df.isnull().any()), item_df.isnull().any()
    logging.debug(item_df)
    logging.debug(item_df.info())

    logging.info('读入评分数据...')
    interaction_df: DataFrame = pd.read_csv(os.path.join(RAW_DATA_DIR, RAW_DATA_NAME, RAW_INTERACTION_NAME), sep='::',
                                            header=None,
                                            engine='python', dtype=np.int32)
    interaction_df.columns = [UID, IID, RATE, TIME]
    assert not any(interaction_df.isnull().any()), interaction_df.isnull().any()
    interaction_df[LABEL] = interaction_df[RATE].map(rank_to_label).astype(np.int32)
    interaction_df = interaction_df[[UID, IID, RATE, LABEL, TIME]]
    logging.debug('排序评分数据...')
    interaction_df.sort_values(by=[UID, TIME], kind='mergesort', inplace=True)
    interaction_df.reset_index(inplace=True, drop=True)
    logging.debug('重映射物品ID...')
    iid_int_map = get_int_map(set(interaction_df[IID]) & set(item_df[IID]), start=1)
    interaction_df[IID] = interaction_df[IID].map(iid_int_map).astype(np.int32)
    item_df = item_df[item_df[IID].isin(iid_int_map)].reset_index(drop=True)
    item_df[IID] = item_df[IID].map(iid_int_map).astype(np.int32)
    logging.debug(interaction_df)
    logging.debug(interaction_df.info())
    logging.debug(item_df)
    logging.debug(item_df.info())
    # 基本特征信息
    description.uid_column = CategoricalColumnWithIdentity.from_series(
        feature_name=UID,
        series=interaction_df[UID])
    description.iid_column = CategoricalColumnWithIdentity.from_series(
        feature_name=IID,
        series=interaction_df[IID],
        other_info={INT_MAP: iid_int_map})
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

    merge_interaction_df = pd.merge(
        left=pd.merge(left=interaction_df, right=user_df, on=UID, how="left", validate="many_to_one"),
        right=item_df, on=IID, how="left", validate="many_to_one")
    logging.debug(merge_interaction_df)
    logging.debug(merge_interaction_df.info())

    dataset_dir = os.path.join(DATASET_DIR, dataset_name)
    check_dir_and_mkdir(dataset_dir)

    logging.info('保存数据...')
    assert (interaction_df.dtypes == np.int32).all(), interaction_df.dtypes
    base_interaction_df = merge_interaction_df[[UID, IID, RATE, LABEL, TIME]]
    base_interaction_df.to_csv(os.path.join(dataset_dir, BASE_INTERACTION_CSV), index=False, sep=SEP)
    base_interaction_df.to_feather(os.path.join(dataset_dir, BASE_INTERACTION_FEATHER))
    merge_interaction_df.to_csv(os.path.join(dataset_dir, INTERACTION_CSV), index=False, sep=SEP)
    merge_interaction_df.to_feather(os.path.join(dataset_dir, INTERACTION_FEATHER))
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
        rank_to_label={1: 0, 2: 0, 3: 0, 4: 1, 5: 1},
        info="正负例化的MovieLens-1M数据集，评分为4/5为正例，评分为1/2/3为负例"
    )
    generate_sequential_split(dataset_name=dataset_name, warm_n=5, vt_ratio=0.1)
    generate_leave_k_out_split(dataset_name=dataset_name, warm_n=5, k=1)
    generate_vt_negative_sample(seed=SEED, dataset_name=dataset_name, sample_n=99)
    generate_interaction_history_list(dataset_name=dataset_name, k=10)

    dataset_name = RAW_DATA_NAME + "-P"
    format_data(
        dataset_name=dataset_name,
        rank_to_label={1: 1, 2: 1, 3: 1, 4: 1, 5: 1},
        info="全部视为正例的MovieLens-1M数据集，评分为1/2/3/4/5为正例"
    )
    generate_sequential_split(dataset_name=dataset_name, warm_n=5, vt_ratio=0.1)
    generate_leave_k_out_split(dataset_name=dataset_name, warm_n=5, k=1)
    generate_vt_negative_sample(seed=SEED, dataset_name=dataset_name, sample_n=99)
    generate_interaction_history_list(dataset_name=dataset_name, k=10)
