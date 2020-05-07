"""
数据加载器接口
可以通过Dataset类与DataLoader类以批量形式提供数据
"""
import pickle as pkl
from abc import abstractmethod, ABC
from typing import Dict, Any

from torchrec.utils.argument import IWithArguments
from torchrec.utils.const import *
from .dataset import DatasetDescription


class IDataReader(IWithArguments, ABC):
    """数据读入接口类"""

    @staticmethod
    def get_dataset_description(dataset_name: str, append_id: bool) -> DatasetDescription:
        """获取数据集信息供其他模块使用"""
        with open(os.path.join(DATASET_DIR, dataset_name, DESCRIPTION_PKL), "rb") as dataset_description_pkl:
            dataset_description: DatasetDescription = pkl.load(dataset_description_pkl)
        if append_id:
            assert dataset_description.uid_column is not None and dataset_description.iid_column is not None
            dataset_description.user_categorical_columns.insert(0, dataset_description.uid_column)
            dataset_description.item_categorical_columns.insert(0, dataset_description.iid_column)
        return dataset_description

    @abstractmethod
    def get_train_dataset_size(self) -> int:
        """获取训练集大小"""

    @abstractmethod
    def get_train_dataset_item(self, index: int) -> Dict[str, Any]:
        """获取训练集数据"""

    @abstractmethod
    def get_dev_dataset_size(self) -> int:
        """获取验证集大小"""

    @abstractmethod
    def get_dev_dataset_item(self, index: int) -> Dict[str, Any]:
        """获取验证集数据"""

    @abstractmethod
    def get_test_dataset_size(self) -> int:
        """获取测试集大小"""

    @abstractmethod
    def get_test_dataset_item(self, index: int) -> Dict[str, Any]:
        """获取测试集数据"""
