"""
MovieLens-100K数据预处理
"""
import logging
import pickle as pkl
from typing import Dict

import numpy as np
import pandas as pd
from pandas import DataFrame

from torchrec.data.dataset import DatasetDescription
from torchrec.data.preprocess.dataset_processor import create_user_history_info, create_history_info, \
    generate_negative_sample, sequential_split, leave_k_out_split
from torchrec.data.preprocess.feature_processor import get_int_map, get_bucketize_fn
from torchrec.feature_columns import CategoricalColumnWithIdentity
from torchrec.utils.const import *
from torchrec.utils.system import init_console_logger, check_dir_and_mkdir

pd.set_option('display.max_colwidth', 20)
pd.set_option('display.max_columns', None)

SEED = 2020
RAW_DATA_NAME = 'MovieLens-100K'
RAW_INTERACTION_NAME = 'u.data'
RAW_USER_NAME = 'u.user'
RAW_ITEM_NAME = 'u.item.utf8'


def format_data(dataset_name: str, rank_to_label: Dict, info: str) -> None:
    """
    过滤并格式化原始数据集中用户ID、物品ID、评分、标签、时间戳四项基本信息、物品特征信息（可选）、用户特征信息（可选）、上下文特征信息（可选），并统计相关信息
    标签是二值化的，评分如果不存在，与标签一致
    生成如下文件：
        interaction.csv/pkl
        user.csv/pkl
        item.csv/pkl
        description.json
    """
    description = DatasetDescription(info)

    logging.info('读入用户数据...')
    U_AGE, U_GENDER, U_OCCUPATION = "u_age", "u_gender", "u_occupation"
    user_usecols = [0, 1, 2, 3]
    user_dtype_dict = {0: np.int32, 1: np.int32, 2: np.str, 3: np.str}
    user_df: DataFrame = pd.read_csv(os.path.join(RAW_DATA_DIR, RAW_DATA_NAME, RAW_USER_NAME), sep='|', header=None,
                                     usecols=user_usecols, na_filter=False, dtype=user_dtype_dict)
    user_df.columns = [UID, U_AGE, U_GENDER, U_OCCUPATION]
    # 处理年龄特征
    # *  0:  "Under 18"
    # *  1:  "18-24"
    # *  2:  "25-34"
    # *  3:  "35-44"
    # *  4:  "45-49"
    # *  5:  "50-55"
    # *  6:  "56+"
    # Movielens-1M的划分方法
    u_age_bucket_boundaries = [18, 25, 35, 45, 50, 56]
    user_df[U_AGE] = user_df[U_AGE].map(get_bucketize_fn(u_age_bucket_boundaries)).astype(np.int32)
    description.user_categorical_columns.append(CategoricalColumnWithIdentity.from_series(
        feature_name=U_AGE,
        series=user_df[U_AGE],
        other_info={BUCKET_BOUNDARIES: u_age_bucket_boundaries}
    ))
    # 处理性别特征
    u_gender_int_map = {"M": 0, "F": 1}
    user_df[U_GENDER] = user_df[U_GENDER].map(u_gender_int_map).astype(np.int32)
    description.user_categorical_columns.append(CategoricalColumnWithIdentity.from_series(
        feature_name=U_GENDER,
        series=user_df[U_GENDER],
        other_info={INT_MAP: u_gender_int_map}
    ))
    # 处理职业特征
    u_occupation_int_map = get_int_map(user_df[U_OCCUPATION])
    user_df[U_OCCUPATION] = user_df[U_OCCUPATION].map(u_occupation_int_map).astype(np.int32)
    description.user_categorical_columns.append(CategoricalColumnWithIdentity.from_series(
        feature_name=U_OCCUPATION,
        series=user_df[U_OCCUPATION],
        other_info={INT_MAP: u_occupation_int_map}
    ))
    print(user_df)
    print(user_df.dtypes)

    logging.info('读入物品数据...')
    item_usecols = [0, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    item_dtype_dict = {0: np.int32, 2: np.str}
    for i in range(5, 24):
        item_dtype_dict[i] = np.int32
    item_feature_names = ["i_year", "i_unknown", "i_action", "i_adventure", "i_animation", "i_children", "i_comedy",
                          "i_crime", "i_documentary", "i_drama", "i_fantasy", "i_film_noir", "i_horror", "i_musical",
                          "i_mystery", "i_romance", "i_sci_fi", "i_thriller", "i_war", "i_western"]
    item_df: DataFrame = pd.read_csv(os.path.join(RAW_DATA_DIR, RAW_DATA_NAME, RAW_ITEM_NAME), sep='|', header=None,
                                     usecols=item_usecols, na_filter=False, dtype=item_dtype_dict)
    item_df.columns = [IID] + item_feature_names
    # 处理年份信息
    item_df["i_year"] = item_df["i_year"].map(lambda s: int(s[-4:]) if s != "" else -1)
    i_year_boundaries = [1940, 1950, 1960, 1970, 1980, 1985] + list(range(1990, item_df['i_year'].max() + 1))
    item_df["i_year"] = item_df["i_year"].map(get_bucketize_fn(i_year_boundaries)).astype(np.int32)
    description.item_categorical_columns.append(CategoricalColumnWithIdentity.from_series(
        feature_name="i_year",
        series=item_df["i_year"],
        other_info={BUCKET_BOUNDARIES: i_year_boundaries}
    ))
    # 处理其他二值信息
    for feature_name in item_feature_names[1:]:
        item_df[feature_name] = item_df[feature_name].astype(np.int32)
        description.item_categorical_columns.append(CategoricalColumnWithIdentity.from_series(
            feature_name=feature_name,
            series=item_df[feature_name],
            other_info=dict()
        ))
    print(item_df)
    print(item_df.dtypes)

    logging.info('读入评分数据...')
    interaction_df: DataFrame = pd.read_csv(os.path.join(RAW_DATA_DIR, RAW_DATA_NAME, RAW_INTERACTION_NAME), sep='\t',
                                            header=None, na_filter=False, dtype=np.int32)
    interaction_df.columns = [UID, IID, RATE, TIME]
    interaction_df[LABEL] = interaction_df[RATE].map(rank_to_label).astype(np.int32)
    interaction_df = interaction_df[[UID, IID, RATE, LABEL, TIME]]
    logging.info('排序评分数据...')
    interaction_df.sort_values(by=[UID, TIME], kind='mergesort', inplace=True)
    interaction_df.reset_index(inplace=True, drop=True)
    # 处理基本特征信息
    description.uid_column = CategoricalColumnWithIdentity.from_series(UID, interaction_df[UID], dict())
    description.iid_column = CategoricalColumnWithIdentity.from_series(IID, interaction_df[IID], dict())
    description.rate_column = CategoricalColumnWithIdentity.from_series(RATE, interaction_df[RATE], dict())
    description.label_column = CategoricalColumnWithIdentity.from_series(LABEL, interaction_df[LABEL], dict())
    description.time_column = CategoricalColumnWithIdentity.from_series(TIME, interaction_df[TIME], dict())
    # 统计交互信息
    description.get_user_interaction_statistic(interaction_df)
    print(interaction_df)
    print(interaction_df.dtypes)

    dataset_dir = os.path.join(DATASET_DIR, dataset_name)
    check_dir_and_mkdir(dataset_dir)

    logging.info('保存评分数据...')
    assert (interaction_df.dtypes == np.int32).all(), interaction_df.dtypes
    base_interaction_df = interaction_df[[UID, IID, RATE, LABEL, TIME]]
    base_interaction_df.to_csv(os.path.join(dataset_dir, BASE_INTERACTION_CSV), index=False, sep=SEP)
    base_interaction_df.to_pickle(os.path.join(dataset_dir, BASE_INTERACTION_PKL))
    interaction_df.to_csv(os.path.join(dataset_dir, INTERACTION_CSV), index=False, sep=SEP)
    interaction_df.to_pickle(os.path.join(dataset_dir, INTERACTION_PKL))

    logging.info('保存用户数据...')
    assert (user_df.dtypes == np.int32).all(), user_df.dtypes
    user_df.to_csv(os.path.join(dataset_dir, USER_CSV), index=False, sep=SEP)
    user_df.to_pickle(os.path.join(dataset_dir, USER_PKL))

    logging.info('保存物品数据...')
    assert (item_df.dtypes == np.int32).all(), item_df.dtypes
    item_df.to_csv(os.path.join(dataset_dir, ITEM_CSV), index=False, sep=SEP)
    item_df.to_pickle(os.path.join(dataset_dir, ITEM_PKL))

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
        info="正负例化的MovieLens-100K数据集，评分为4/5为正例，评分为1/2/3为负例"
    )
    create_user_history_info(dataset_name=dataset_name)
    for i in range(1, 11):
        create_history_info(dataset_name=dataset_name, k=i)
    generate_negative_sample(seed=SEED, dataset_name=dataset_name, sample_n=99)
    sequential_split(dataset_name=dataset_name, warm_n=5, vt_ratio=0.1)
    leave_k_out_split(dataset_name=dataset_name, warm_n=5, k=1)

    dataset_name = RAW_DATA_NAME + "-P"
    format_data(
        dataset_name=dataset_name,
        rank_to_label={1: 1, 2: 1, 3: 1, 4: 1, 5: 1},
        info="全部视为正例的MovieLens-100K数据集，评分为1/2/3/4/5为正例"
    )
    create_user_history_info(dataset_name=dataset_name)
    for i in range(1, 11):
        create_history_info(dataset_name=dataset_name, k=i)
    generate_negative_sample(seed=SEED, dataset_name=dataset_name, sample_n=99)
    sequential_split(dataset_name=dataset_name, warm_n=5, vt_ratio=0.1)
    leave_k_out_split(dataset_name=dataset_name, warm_n=5, k=1)
