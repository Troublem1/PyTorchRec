"""
Xing数据集预处理
"""
import logging
import pickle as pkl

import numpy as np
import pandas as pd
from pandas import DataFrame

from torchrec.data.dataset import DatasetDescription
from torchrec.data.process import generate_interaction_history_list
from torchrec.data.process import generate_leave_k_out_split
from torchrec.data.process import generate_sequential_split
from torchrec.data.process import generate_vt_negative_sample
from torchrec.data.process.feature_process import get_int_map
from torchrec.data.process.sequential_split import __get_warm_interaction_df
from torchrec.feature_column import CategoricalColumnWithIdentity
from torchrec.utils.const import *
from torchrec.utils.system import init_console_logger, check_dir_and_mkdir

pd.set_option('display.max_colwidth', 20)
pd.set_option('display.max_columns', None)

SEED = 2020
RAW_DATA_NAME = 'Xing'
RAW_INTERACTION_NAME = 'interactions.csv'
RAW_USER_NAME = 'users.csv'
RAW_ITEM_NAME = 'items.csv'

PRE_USER_FEATHER = 'pre_user.feather'
PRE_ITEM_FEATHER = 'pre_item.feather'
PRE_INTERACTION_FEATHER = 'pre_interaction.feather'


def prepare_user_data() -> None:
    """数据量较大，预先处理user信息"""
    logging.info('读入用户数据...')
    user_df: DataFrame = pd.read_csv(os.path.join(RAW_DATA_DIR, RAW_DATA_NAME, RAW_USER_NAME), sep=SEP)
    user_df.columns = [
        UID,
        'jobroles',
        'u_c_career_level',
        'u_c_discipline_id',
        'u_c_industry_id',
        'u_c_country',
        'u_c_region',
        'u_c_experience_n_entries_class',
        'u_c_experience_years_experience',
        'u_c_experience_years_in_current',
        'u_c_edu_degree',
        'edu_fieldofstudies',
        'u_c_wtcj',
        'u_c_premium']
    user_df.drop(columns=['jobroles', 'edu_fieldofstudies'], inplace=True)
    assert not any(user_df.isnull().any()), user_df.isnull().any()
    u_country_int_map = {'non_dach': 0, 'de': 1, 'at': 2, 'ch': 3}
    user_df['u_c_country'] = user_df['u_c_country'].map(u_country_int_map)
    user_df[UID] = user_df[UID].astype(np.int32)
    for feature_name in ['u_c_career_level',
                         'u_c_discipline_id',
                         'u_c_industry_id',
                         'u_c_country',
                         'u_c_region',
                         'u_c_experience_n_entries_class',
                         'u_c_experience_years_experience',
                         'u_c_experience_years_in_current',
                         'u_c_edu_degree',
                         'u_c_wtcj',
                         'u_c_premium']:
        user_df[feature_name] = user_df[feature_name].astype(np.int8)
    user_df.sort_values(by=UID, kind="mergesort", inplace=True)
    user_df.reset_index(drop=True, inplace=True)
    logging.debug(user_df.info())
    user_df.to_feather(os.path.join(RAW_DATA_DIR, RAW_DATA_NAME, PRE_USER_FEATHER))


def prepare_item_data() -> None:
    """数据量较大，预先处理item信息"""
    item_df: DataFrame = pd.read_csv(os.path.join(RAW_DATA_DIR, RAW_DATA_NAME, RAW_ITEM_NAME), sep=SEP)
    item_df.columns = [
        IID,
        'title',
        'i_c_career_level',
        'i_c_discipline_id',
        'i_c_industry_id',
        'i_c_country',
        'i_c_is_paid',
        'i_c_region',
        'i_c_latitude',
        'i_c_longitude',
        'i_c_employment',
        'tags',
        'i_c_created_at']
    item_df.drop(columns=['title', 'tags'], inplace=True)
    i_country_int_map = {'non_dach': 0, 'de': 1, 'at': 2, 'ch': 3}
    item_df['i_c_country'] = item_df['i_c_country'].map(i_country_int_map)
    item_df['i_c_latitude'] = item_df['i_c_latitude'].map(
        lambda x: 0 if np.isnan(x) else int((int(x + 90) / 10)) + 1)
    item_df['i_c_longitude'] = item_df['i_c_longitude'].map(
        lambda x: 0 if np.isnan(x) else int((int(x + 180) / 10)) + 1)
    item_df['i_c_created_at'] = pd.to_datetime(item_df['i_c_created_at'], unit='s')
    item_year = item_df['i_c_created_at'].map(lambda x: x.year)
    min_year = item_year.min()
    item_month = item_df['i_c_created_at'].map(lambda x: x.month)
    item_df['i_c_created_at'] = (item_year.fillna(-1) - min_year) * 12 + item_month.fillna(-1)
    item_df['i_c_created_at'] = item_df['i_c_created_at'].map(lambda x: int(x) if x > 0 else 0)
    item_df[IID] = item_df[IID].astype(np.int32)
    for feature_name in ['i_c_career_level',
                         'i_c_discipline_id',
                         'i_c_industry_id',
                         'i_c_country',
                         'i_c_is_paid',
                         'i_c_region',
                         'i_c_latitude',
                         'i_c_longitude',
                         'i_c_employment',
                         'i_c_created_at']:
        item_df[feature_name] = item_df[feature_name].astype(np.int8)
    item_df.sort_values(IID, kind="mergesort", inplace=True)
    item_df.reset_index(drop=True, inplace=True)
    logging.debug(item_df.info())
    item_df.to_feather(os.path.join(RAW_DATA_DIR, RAW_DATA_NAME, PRE_ITEM_FEATHER))


def prepare_interaction_data() -> None:
    """预处理交互信息"""
    dtype = {
        "recsyschallenge_v2017_interactions_final_anonym_training_export.user_id": np.int32,
        "recsyschallenge_v2017_interactions_final_anonym_training_export.item_id": np.int32,
        "recsyschallenge_v2017_interactions_final_anonym_training_export.interaction_type": np.int8,
        "recsyschallenge_v2017_interactions_final_anonym_training_export.created_at": np.int32}
    interaction_df: DataFrame = pd.read_csv(os.path.join(RAW_DATA_DIR, RAW_DATA_NAME, RAW_INTERACTION_NAME), sep=SEP,
                                            dtype=dtype, na_filter=True)
    interaction_df.columns = [UID, IID, LABEL, TIME]
    logging.debug('按照标签排序...')
    interaction_df.sort_values(by=[UID, LABEL], kind='mergesort', inplace=True)
    logging.debug('去重[UID, IID]...')
    interaction_df.drop_duplicates([UID, IID], keep='last', inplace=True)
    logging.debug('排序评分数据...')
    interaction_df.sort_values(by=[UID, TIME], kind='mergesort', inplace=True)
    logging.debug('生成评分与标签...')
    label_to_rate_map = {0: 0, 1: 1, 2: 5, 3: 5, 4: -10, 5: 20}
    interaction_df[RATE] = interaction_df[LABEL].map(label_to_rate_map).astype(np.int8)
    label_to_label_map = {0: 0, 1: 1, 2: 1, 3: 1, 4: 0, 5: 1}
    interaction_df[LABEL] = interaction_df[LABEL].map(label_to_label_map).astype(np.int8)
    interaction_df = interaction_df[[UID, IID, RATE, LABEL, TIME]]
    interaction_df.reset_index(drop=True, inplace=True)
    logging.debug(interaction_df.info())
    interaction_df.to_feather(os.path.join(RAW_DATA_DIR, RAW_DATA_NAME, PRE_INTERACTION_FEATHER))


def format_data(dataset_name: str, info: str) -> None:
    """
    过滤并格式化原始数据集中用户ID、物品ID、评分、标签、时间戳五项基本信息、上下文/物品/用户特征信息（可选），并统计相关信息
    标签是二值化的，评分与标签一致
    """
    description = DatasetDescription(info)

    logging.info('读入预处理的用户数据...')
    user_df: DataFrame = pd.read_feather(os.path.join(RAW_DATA_DIR, RAW_DATA_NAME, PRE_USER_FEATHER))
    description.user_columns.append(CategoricalColumnWithIdentity.from_series(
        feature_name='u_c_career_level',
        series=user_df['u_c_career_level']
    ))
    description.user_columns.append(CategoricalColumnWithIdentity.from_series(
        feature_name='u_c_discipline_id',
        series=user_df['u_c_discipline_id']
    ))
    description.user_columns.append(CategoricalColumnWithIdentity.from_series(
        feature_name='u_c_industry_id',
        series=user_df['u_c_industry_id']
    ))
    u_country_int_map = {'non_dach': 0, 'de': 1, 'at': 2, 'ch': 3}
    description.user_columns.append(CategoricalColumnWithIdentity.from_series(
        feature_name='u_c_country',
        series=user_df['u_c_country'],
        other_info={INT_MAP: u_country_int_map}
    ))
    description.user_columns.append(CategoricalColumnWithIdentity.from_series(
        feature_name='u_c_region',
        series=user_df['u_c_region']
    ))
    description.user_columns.append(CategoricalColumnWithIdentity.from_series(
        feature_name='u_c_experience_n_entries_class',
        series=user_df['u_c_experience_n_entries_class']
    ))
    description.user_columns.append(CategoricalColumnWithIdentity.from_series(
        feature_name='u_c_experience_years_experience',
        series=user_df['u_c_experience_years_experience']
    ))
    description.user_columns.append(CategoricalColumnWithIdentity.from_series(
        feature_name='u_c_experience_years_in_current',
        series=user_df['u_c_experience_years_in_current']
    ))
    description.user_columns.append(CategoricalColumnWithIdentity.from_series(
        feature_name='u_c_edu_degree',
        series=user_df['u_c_edu_degree']
    ))
    description.user_columns.append(CategoricalColumnWithIdentity.from_series(
        feature_name='u_c_wtcj',
        series=user_df['u_c_wtcj']
    ))
    description.user_columns.append(CategoricalColumnWithIdentity.from_series(
        feature_name='u_c_premium',
        series=user_df['u_c_premium']
    ))
    logging.debug(user_df)
    logging.debug(user_df.info())
    for column in description.user_columns:
        logging.debug(column)

    logging.info('读入物品数据...')
    item_df: DataFrame = pd.read_feather(os.path.join(RAW_DATA_DIR, RAW_DATA_NAME, PRE_ITEM_FEATHER))
    description.item_columns.append(CategoricalColumnWithIdentity.from_series(
        feature_name='i_c_career_level',
        series=item_df['i_c_career_level']
    ))
    description.item_columns.append(CategoricalColumnWithIdentity.from_series(
        feature_name='i_c_discipline_id',
        series=item_df['i_c_discipline_id']
    ))
    description.item_columns.append(CategoricalColumnWithIdentity.from_series(
        feature_name='i_c_industry_id',
        series=item_df['i_c_industry_id']
    ))
    i_country_int_map = {'non_dach': 0, 'de': 1, 'at': 2, 'ch': 3}
    description.item_columns.append(CategoricalColumnWithIdentity.from_series(
        feature_name='i_c_country',
        series=item_df['i_c_country'],
        other_info={INT_MAP: i_country_int_map}
    ))
    description.item_columns.append(CategoricalColumnWithIdentity.from_series(
        feature_name='i_c_is_paid',
        series=item_df['i_c_is_paid']
    ))
    description.item_columns.append(CategoricalColumnWithIdentity.from_series(
        feature_name='i_c_region',
        series=item_df['i_c_region']
    ))
    description.item_columns.append(CategoricalColumnWithIdentity.from_series(
        feature_name='i_c_latitude',
        series=item_df['i_c_latitude']
    ))
    description.item_columns.append(CategoricalColumnWithIdentity.from_series(
        feature_name='i_c_longitude',
        series=item_df['i_c_longitude']
    ))
    description.item_columns.append(CategoricalColumnWithIdentity.from_series(
        feature_name='i_c_employment',
        series=item_df['i_c_employment']
    ))
    description.item_columns.append(CategoricalColumnWithIdentity.from_series(
        feature_name='i_c_created_at',
        series=item_df['i_c_created_at']
    ))
    assert not any(item_df.isnull().any()), item_df.isnull().any()
    logging.debug(item_df)
    logging.debug(item_df.info())
    for column in description.item_columns:
        logging.debug(column)

    logging.info('读入评分数据...')
    interaction_df: DataFrame = pd.read_feather(os.path.join(RAW_DATA_DIR, RAW_DATA_NAME, PRE_INTERACTION_FEATHER))
    logging.debug('删除没有正向操作的用户...')
    interaction_df = __get_warm_interaction_df(interaction_df, 1)
    logging.debug('重映射物品ID...')
    iid_int_map = get_int_map(set(interaction_df[IID]) & set(item_df[IID]), start=1)
    interaction_df[IID] = interaction_df[IID].map(iid_int_map).astype(np.int32)
    item_df = item_df[item_df[IID].isin(iid_int_map)].reset_index(drop=True)
    item_df[IID] = item_df[IID].map(iid_int_map).astype(np.int32)
    logging.debug('重映射用户ID...')
    uid_int_map = get_int_map(set(interaction_df[UID]) & set(user_df[UID]), start=1)
    interaction_df[UID] = interaction_df[UID].map(uid_int_map).astype(np.int32)
    user_df = user_df[user_df[UID].isin(uid_int_map)].reset_index(drop=True)
    user_df[UID] = user_df[UID].map(uid_int_map).astype(np.int32)
    interaction_df.reset_index(drop=True, inplace=True)
    logging.debug(interaction_df)
    logging.debug(interaction_df.info())
    logging.debug(item_df)
    logging.debug(item_df.info())
    logging.debug(user_df)
    logging.debug(user_df.info())
    # 基本特征信息
    description.uid_column = CategoricalColumnWithIdentity.from_series(
        feature_name=UID,
        series=interaction_df[UID],
        other_info={INT_MAP: uid_int_map})
    description.iid_column = CategoricalColumnWithIdentity.from_series(
        feature_name=IID,
        series=interaction_df[IID],
        other_info={INT_MAP: iid_int_map})
    label_to_rate_map = {0: 0, 1: 1, 2: 5, 3: 5, 4: -10, 5: 20}
    description.rate_column = CategoricalColumnWithIdentity.from_series(
        feature_name=RATE,
        series=interaction_df[RATE],
        other_info={INT_MAP: label_to_rate_map})
    label_to_label_map = {0: 0, 1: 1, 2: 1, 3: 1, 4: 0, 5: 1}
    description.label_column = CategoricalColumnWithIdentity.from_series(
        feature_name=LABEL,
        series=interaction_df[LABEL],
        other_info={INT_MAP: label_to_label_map})
    description.time_column = CategoricalColumnWithIdentity.from_series(
        feature_name=TIME,
        series=interaction_df[TIME])
    # 统计交互信息
    description.get_user_interaction_statistic(interaction_df)

    merge_interaction_df = pd.merge(
        left=pd.merge(left=interaction_df, right=user_df, on=UID, how="left"),
        right=item_df, on=IID, how="left")
    logging.debug(merge_interaction_df)
    logging.debug(merge_interaction_df.info())

    dataset_dir = os.path.join(DATASET_DIR, dataset_name)
    check_dir_and_mkdir(dataset_dir)

    logging.info('保存数据...')
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
    init_console_logger(logging.DEBUG)

    prepare_user_data()
    prepare_item_data()
    prepare_interaction_data()

    dataset_name = RAW_DATA_NAME + "-PN"
    format_data(
        dataset_name=dataset_name,
        info="正负例化的MovieLens-100K数据集，评分为比赛计分规则，评分>0为正例"
    )
    generate_sequential_split(dataset_name=dataset_name, warm_n=5, vt_ratio=0.1)
    generate_leave_k_out_split(dataset_name=dataset_name, warm_n=5, k=1)
    generate_vt_negative_sample(seed=SEED, dataset_name=dataset_name, sample_n=99)
    generate_interaction_history_list(dataset_name=dataset_name, k=10)
