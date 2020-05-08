"""
ID数据加载器，加载数据集ID信息
可以通过Dataset类与DataLoader类以批量形式提供数据
"""
import gc
import logging
import pickle as pkl
from typing import List, Dict, Any, Optional, Set

import numpy as np
import pandas as pd
from numpy import ndarray
from numpy.random import default_rng
from pandas import DataFrame
from tqdm import tqdm

from torchrec.data.IDataReader import IDataReader
from torchrec.data.dataset import SplitMode, DatasetDescription
from torchrec.data.process import check_dataset_info
from torchrec.data.process import check_leave_k_out_split, generate_leave_k_out_split
from torchrec.data.process import check_sequential_split, generate_sequential_split
from torchrec.task import TrainMode
from torchrec.utils.argument import ArgumentDescription
from torchrec.utils.const import *
from torchrec.utils.enum import get_enum_values


class SimpleDataReader(IDataReader):
    """
    数据格式：
    评分/点击预测任务的训练集、验证集、测试集，TOPK推荐任务POINTWISE模式训练集：
    {
        INDEX:      int32
        UID:        int32
        IID:        int32
        RATE:       int32
        LABEL:      int32
        TIME:       int32
        c_XXX:      int32/float32  # 上下文特征，可选
        u_XXX:      int32/float32  # 用户特征，可选
        i_XXX:      int32/float32  # 物品特征，可选
    }

    TOPK推荐任务PAIRWISE模式训练集、TOPK推荐任务验证集、测试集：{
        INDEX:      int32
        UID:        int32
        IID:        ndarray(int32)  # 训练集长度为2，验证集测试集长度为vt_sample_n
        RATE:       int32
        LABEL:      int32
        TIME:       int32
        c_XXX:      int32/float32  # 上下文特征，可选
        u_XXX:      int32/float32  # 用户特征，可选
        i_XXX:      ndarray(int32/float32)  # 物品特征，可选
    }
    """

    @classmethod
    def get_argument_descriptions(cls) -> List[ArgumentDescription]:
        """获取参数描述信息"""
        argument_descriptions = super().get_argument_descriptions()
        argument_descriptions.extend([
            ArgumentDescription(name="dataset", type_=str, help_info="数据集名称",
                                legal_value_list=check_dataset_info()),
            ArgumentDescription(name="dataset_mode", type_=str, help_info="数据集划分模式",
                                default_value=SplitMode.LEAVE_K_OUT.value,
                                legal_value_list=get_enum_values(SplitMode)),
            ArgumentDescription(name="warm_n", type_=int, help_info="暖用户限制",
                                default_value=5,
                                lower_closed_bound=1),
            ArgumentDescription(name="vt_ratio", type_=float, help_info="顺序划分模式下验证集测试集比例",
                                default_value=0.1,
                                lower_open_bound=0.0,
                                upper_open_bound=0.5),
            ArgumentDescription(name="leave_k", type_=int, help_info="留出划分模式下验证集测试集正例个数",
                                default_value=1,
                                lower_closed_bound=1),
            ArgumentDescription(name="neg_sample_n", type_=int, help_info="验证集测试集负采样数量",
                                default_value=99,
                                lower_closed_bound=1),
            ArgumentDescription(name="load_feature", type_=bool, help_info="是否载入特征信息",
                                default_value=False),
            ArgumentDescription(name="append_id", type_=bool, help_info="特征信息中是否包含ID信息",
                                default_value=False)
        ])
        return argument_descriptions

    @classmethod
    def check_argument_values(cls, arguments: Dict[str, Any]) -> None:
        """根据预设信息检查参数值"""
        super().check_argument_values(arguments)

        # 数据集划分模式
        arguments["dataset_mode"] = SplitMode(arguments["dataset_mode"])

        # 添加数据集描述信息
        dataset_description = cls.get_dataset_description(arguments["dataset"], arguments["append_id"])
        arguments["dataset_description"] = dataset_description

    def __init__(self, dataset: str, dataset_mode: SplitMode, warm_n: int, vt_ratio: float, leave_k: int,
                 neg_sample_n: int, load_feature: bool, append_id: bool, description: DatasetDescription,
                 train_mode: TrainMode, random_seed: int):
        """
        :param dataset: 数据集名称
        """
        self.dataset = dataset
        self.dataset_mode = dataset_mode
        self.warm_n = warm_n
        self.vt_ratio = vt_ratio
        self.leave_k = leave_k
        self.neg_sample_n = neg_sample_n
        self.load_feature = load_feature
        self.append_id = append_id
        self.description = description
        self.train_mode = train_mode
        self.rng = default_rng(random_seed)

        self.interaction_df: Optional[DataFrame] = None
        self.user_df: Optional[DataFrame] = None
        self.item_df: Optional[DataFrame] = None

        self.train_index_array: Optional[ndarray] = None
        self.dev_index_array: Optional[ndarray] = None
        self.test_index_array: Optional[ndarray] = None

        self.train_df: Optional[DataFrame] = None
        self.dev_df: Optional[DataFrame] = None
        self.test_df: Optional[DataFrame] = None

        self.dev_iid_topk_array: Optional[ndarray] = None
        self.test_iid_topk_array: Optional[ndarray] = None

        self.min_iid_array_index: Optional[int] = None
        self.max_iid_array_index: Optional[int] = None
        self.user_pos_his_set_dict: Optional[Dict[int, Set[int]]] = None
        self.train_iid_pair_array: Optional[ndarray] = None

        self.context_feature_names: Optional[List[str]] = None
        self.user_feature_names: Optional[List[str]] = None
        self.item_feature_names: Optional[List[str]] = None

        logging.info(f'加载{dataset}数据集...')
        self._load_dataset()
        self._generate_feature_name_list()
        gc.collect()
        logging.info(f'加载{dataset}数据集结束')

    def _load_dataset(self) -> None:
        """数据集加载过程，构造函数调用，子类应该重载"""
        self._load_interactions()
        self._split_interactions()
        if self.dataset_mode == SplitMode.LEAVE_K_OUT:
            self._load_neg_sample()
        if self.load_feature:
            self._load_features()
        if self.train_mode == TrainMode.PAIR_WISE:
            self._prepare_train_neg_sample()

    def _load_interactions(self) -> None:
        """加载交互数据"""
        logging.info("加载用户交互数据...")
        if self.load_feature:
            interaction_pkl = os.path.join(DATASET_DIR, self.dataset, INTERACTION_PKL)
        else:
            interaction_pkl = os.path.join(DATASET_DIR, self.dataset, BASE_INTERACTION_PKL)
        self.interaction_df: DataFrame = pd.read_pickle(interaction_pkl)
        logging.debug(self.interaction_df)
        logging.info('交互数据大小：%d' % len(self.interaction_df))
        logging.info('交互数据正负例统计：' + str(dict(self.interaction_df[LABEL].value_counts())))

    def _split_interactions(self) -> None:
        """分隔交互数据"""
        logging.info('划分交互数据...')
        split_index_dir = os.path.join(DATASET_DIR, self.dataset, SPLIT_INDEX_DIR)

        if self.dataset_mode == SplitMode.SEQUENTIAL_SPLIT:
            if (self.warm_n, self.vt_ratio) not in check_sequential_split(self.dataset):
                logging.info(f"顺序划分_{self.warm_n}_{self.vt_ratio}不存在，划分数据...")
                generate_sequential_split(self.dataset, self.warm_n, self.vt_ratio)
            split_name = SEQUENTIAL_SPLIT_NAME_TEMPLATE % (self.warm_n, self.vt_ratio)
        else:  # self.dataset_mode == DatasetMode.LEAVE_K_OUT
            if (self.warm_n, self.leave_k) not in check_leave_k_out_split(self.dataset):
                logging.info(f"留出划分_{self.warm_n}_{self.leave_k}不存在，划分数据...")
                generate_leave_k_out_split(self.dataset, self.warm_n, self.leave_k)
            split_name = LEAVE_K_OUT_SPLIT_NAME_TEMPLATE % (self.warm_n, self.leave_k)

        self.train_index_array: ndarray = np.load(os.path.join(split_index_dir, TRAIN_INDEX_NPY_TEMPLATE % split_name))
        self.dev_index_array: ndarray = np.load(os.path.join(split_index_dir, DEV_INDEX_NPY_TEMPLATE % split_name))
        self.test_index_array: ndarray = np.load(os.path.join(split_index_dir, TEST_INDEX_NPY_TEMPLATE % split_name))

        self.train_df: DataFrame = self.interaction_df.loc[
            self.interaction_df.index.intersection(self.train_index_array)]
        self.dev_df: DataFrame = self.interaction_df.loc[
            self.interaction_df.index.intersection(self.dev_index_array)]
        self.test_df: DataFrame = self.interaction_df.loc[
            self.interaction_df.index.intersection(self.test_index_array)]

        logging.debug(self.train_df)
        logging.info('训练集大小：%d' % len(self.train_df))
        logging.info('训练集正负例统计：' + str(dict(self.train_df[LABEL].value_counts())))
        logging.debug(self.dev_df)
        logging.info('验证集大小：%d' % len(self.dev_df))
        logging.info('验证集正负例统计：' + str(dict(self.dev_df[LABEL].value_counts())))
        logging.debug(self.test_df)
        logging.info('测试集大小：%d' % len(self.test_df))
        logging.info('测试集正负例统计：' + str(dict(self.test_df[LABEL].value_counts())))

    def _load_neg_sample(self) -> None:
        """
        加载负采样信息，合并验证集测试集的 IID 信息
        """
        logging.info('加载负采样信息...')
        neg_sample_dir = os.path.join(DATASET_DIR, self.dataset, NEGATIVE_SAMPLE_DIR)

        # 生成实际用户列表（因为暖用户可能会删去一些），注意下标-1（PAD）
        user_index_array: ndarray = self.dev_df[UID].values - 1

        logging.debug('加载验证集负采样信息...')
        dev_neg_npy = os.path.join(neg_sample_dir, DEV_NEG_NPY_TEMPLATE % self.neg_sample_n)
        dev_neg_array: ndarray = np.load(dev_neg_npy)[user_index_array]
        dev_pos_array: ndarray = self.dev_df[IID].values.reshape(-1, 1)
        self.dev_iid_topk_array: ndarray = np.hstack((dev_pos_array, dev_neg_array))
        logging.debug(self.dev_iid_topk_array)
        logging.debug(self.dev_iid_topk_array.shape)

        logging.debug('加载测试集负采样信息...')
        test_neg_npy = os.path.join(neg_sample_dir, TEST_NEG_NPY_TEMPLATE % self.neg_sample_n)
        test_neg_array: ndarray = np.load(test_neg_npy)[user_index_array]
        test_pos_array: ndarray = self.test_df[IID].values.reshape(-1, 1)
        self.test_iid_topk_array: ndarray = np.hstack((test_pos_array, test_neg_array))
        logging.debug(self.test_iid_topk_array)
        logging.debug(self.test_iid_topk_array.shape)

        assert self.dev_iid_topk_array.shape[1] == self.test_iid_topk_array.shape[1]

    def _load_features(self) -> None:
        """加载特征数据"""
        logging.info("加载特征数据...")
        logging.debug("加载用户数据...")
        self.user_df: DataFrame = pd.read_pickle(os.path.join(DATASET_DIR, self.dataset, USER_PKL))
        logging.debug(self.user_df)
        logging.info('用户集大小：%d' % len(self.user_df))
        logging.debug("加载物品数据...")
        self.item_df: DataFrame = pd.read_pickle(os.path.join(DATASET_DIR, self.dataset, ITEM_PKL))
        logging.debug(self.item_df)
        logging.info('物品集大小：%d' % len(self.item_df))

    def _prepare_train_neg_sample(self) -> None:
        """加载必要历史信息，为训练集负采样做准备工作"""
        logging.info('训练集负采样准备工作...')

        self.min_iid_array_index = 1  # pad: 0
        self.max_iid_array_index = self.description.iid_column.category_num

        logging.debug("删除训练集中的负例...")
        self.train_df: DataFrame = self.train_df[self.train_df[LABEL] == 1]
        logging.debug(self.train_df)
        logging.info('删去负例后训练集大小：%d' % len(self.train_df))
        logging.info('删去负例后训练集正负例统计：' + str(dict(self.train_df[LABEL].value_counts())))

        logging.debug('读入交互历史统计信息...')
        statistic_dir = os.path.join(DATASET_DIR, self.dataset, STATISTIC_DIR)
        with open(os.path.join(statistic_dir, USER_POS_HIS_SET_DICT_PKL), 'rb') as user_pos_his_set_dict_pkl:
            self.user_pos_his_set_dict: Dict[int, Set[int]] = pkl.load(user_pos_his_set_dict_pkl)

        train_neg_array: ndarray = np.empty_like(self.train_df[IID].values).reshape(-1, 1)
        train_pos_array: ndarray = self.train_df[IID].values.reshape(-1, 1)
        self.train_iid_pair_array: ndarray = np.hstack((train_pos_array, train_neg_array))
        logging.debug(self.train_iid_pair_array.shape)

    def _generate_feature_name_list(self) -> None:
        """生成特征名称列表"""
        logging.info("生成特征名称列表...")

        # 上下文
        self.context_feature_names = [context_categorical_column.feature_name for context_categorical_column in
                                      self.description.context_categorical_columns] + \
                                     [context_numeric_column.feature_name for context_numeric_column in
                                      self.description.context_numeric_columns]

        self.user_feature_names = [user_categorical_column.feature_name for user_categorical_column in
                                   self.description.user_categorical_columns if
                                   user_categorical_column.feature_name != UID] + \
                                  [user_numeric_column.feature_name for user_numeric_column in
                                   self.description.user_numeric_columns]

        self.item_feature_names = [item_categorical_column.feature_name for item_categorical_column in
                                   self.description.item_categorical_columns if
                                   item_categorical_column.feature_name != IID] + \
                                  [item_numeric_column.feature_name for item_numeric_column in
                                   self.description.item_numeric_columns]

    def train_neg_sample(self) -> None:
        """训练集负采样"""
        assert self.train_mode == TrainMode.PAIR_WISE
        logging.info('训练集负采样...')
        neg_iid_array: ndarray = self.rng.integers(
            low=self.min_iid_array_index,
            high=self.max_iid_array_index,
            size=len(self.train_df.index),
            dtype=np.int32
        )
        for index, uid in tqdm(enumerate(self.train_df[UID].values), total=len(self.train_df.index), leave=False):
            # 该用户训练集中正向交互过的物品 ID 集合
            inter_iid_set = self.user_pos_his_set_dict[uid]
            while neg_iid_array[index] in inter_iid_set:
                neg_iid_array[index] = self.rng.integers(
                    low=self.min_iid_array_index,
                    high=self.max_iid_array_index,
                    dtype=np.int32
                )
        self.train_iid_pair_array[:, 1] = neg_iid_array
        logging.info('训练集负采样结束')

    def get_train_dataset_size(self) -> int:
        """训练集大小"""
        return len(self.train_df.index)

    def get_dev_dataset_size(self) -> int:
        """验证集大小"""
        return len(self.dev_df.index)

    def get_test_dataset_size(self) -> int:
        """测试集大小"""
        return len(self.test_df.index)

    def get_train_dataset_item(self, index: int) -> Dict[str, Any]:
        """第 index 个训练集信息"""
        train_dataset_item = {
            INDEX: index,
            UID: self.train_df[UID].values[index],
            IID: self.train_df[IID].values[index] if self.train_mode == TrainMode.POINT_WISE else
            self.train_iid_pair_array[index],
            RATE: self.train_df[RATE].values[index],
            LABEL: self.train_df[LABEL].values[index],
            TIME: self.train_df[TIME].values[index]
        }
        if self.load_feature:
            for context_feature_name in self.context_feature_names:
                train_dataset_item[context_feature_name] = self.train_df[context_feature_name].values[index]
            user_index = train_dataset_item[UID] - 1  # 0: PAD
            for user_feature_name in self.user_feature_names:
                train_dataset_item[user_feature_name] = self.user_df[user_feature_name].values[user_index]
            item_index = train_dataset_item[IID] - 1  # 0: PAD
            for item_feature_name in self.item_feature_names:
                train_dataset_item[item_feature_name] = self.item_df[item_feature_name].values[item_index]
        return train_dataset_item

    def get_dev_dataset_item(self, index: int) -> Dict[str, Any]:
        """第 index 个验证集信息"""
        dev_dataset_item = {
            INDEX: index,
            UID: self.dev_df[UID].values[index],
            IID: self.dev_df[IID].values[index] if self.dataset_mode == SplitMode.SEQUENTIAL_SPLIT else
            self.dev_iid_topk_array[index],
            RATE: self.dev_df[RATE].values[index],
            LABEL: self.dev_df[LABEL].values[index],
            TIME: self.dev_df[TIME].values[index]
        }
        if self.load_feature:
            for context_feature_name in self.context_feature_names:
                dev_dataset_item[context_feature_name] = self.dev_df[context_feature_name].values[index]
            user_index = dev_dataset_item[UID] - 1  # 0: PAD
            for user_feature_name in self.user_feature_names:
                dev_dataset_item[user_feature_name] = self.user_df[user_feature_name].values[user_index]
            item_index = dev_dataset_item[IID] - 1  # 0: PAD
            for item_feature_name in self.item_feature_names:
                dev_dataset_item[item_feature_name] = self.item_df[item_feature_name].values[item_index]
        return dev_dataset_item

    def get_test_dataset_item(self, index: int) -> Dict[str, Any]:
        """第 index 个测试集信息"""
        test_dataset_item = {
            INDEX: index,
            UID: self.test_df[UID].values[index],
            IID: self.test_df[IID].values[index] if self.dataset_mode == SplitMode.SEQUENTIAL_SPLIT else
            self.test_iid_topk_array[index],
            RATE: self.test_df[RATE].values[index],
            LABEL: self.test_df[LABEL].values[index],
            TIME: self.test_df[TIME].values[index]
        }
        if self.load_feature:
            for context_feature_name in self.context_feature_names:
                test_dataset_item[context_feature_name] = self.test_df[context_feature_name].values[index]
            user_index = test_dataset_item[UID] - 1  # 0: PAD
            for user_feature_name in self.user_feature_names:
                test_dataset_item[user_feature_name] = self.user_df[user_feature_name].values[user_index]
            item_index = test_dataset_item[IID] - 1  # 0: PAD
            for item_feature_name in self.item_feature_names:
                test_dataset_item[item_feature_name] = self.item_df[item_feature_name].values[item_index]
        return test_dataset_item


if __name__ == '__main__':
    from torchrec.utils.system import init_console_logger
    from torchrec.utils.timer import Timer

    init_console_logger()
    with Timer():
        reader = SimpleDataReader(
            dataset='MovieLens-100K-P',
            dataset_mode=SplitMode.SEQUENTIAL_SPLIT,
            warm_n=5,
            vt_ratio=0.1,
            leave_k=1,
            neg_sample_n=99,
            load_feature=True,
            append_id=True,
            description=IDataReader.get_dataset_description('MovieLens-100K-P', append_id=True),
            train_mode=TrainMode.PAIR_WISE,
            random_seed=2020
        )

    with Timer(divided_by=10):
        for i in range(10):
            reader.train_neg_sample()

    print(reader.get_train_dataset_size())
    print(reader.get_dev_dataset_size())
    print(reader.get_test_dataset_size())
    print(reader.get_train_dataset_item(0))
    print(reader.get_dev_dataset_item(0))
    print(reader.get_test_dataset_item(0))
