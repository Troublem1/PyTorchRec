"""
数据加载器接口
可以通过Dataset类与DataLoader类以批量形式提供数据
"""
import pickle as pkl
from abc import abstractmethod
from typing import Dict, Any

from torchrec.utils.argument import IWithArguments
from torchrec.utils.const import *
from .dataset import DatasetDescription


class IDataReader(IWithArguments):
    """数据读入接口类"""

    @staticmethod
    def get_dataset_description(dataset_name) -> DatasetDescription:
        """获取数据集信息供其他模块使用"""
        with open(os.path.join(DATASET_DIR, dataset_name, DESCRIPTION_PKL), "rb") as dataset_description_pkl:
            dataset_description = pkl.load(dataset_description_pkl)
        print(dataset_description)
        return dataset_description

    @abstractmethod
    def get_train_dataset_size(self) -> int:
        """获取训练集大小"""

    @abstractmethod
    def get_train_dataset_item(self, index: int) -> Dict[str, Any]:
        """获取训练集数据"""

    @abstractmethod
    def get_validation_dataset_size(self) -> int:
        """获取训练集大小"""

    @staticmethod
    def get_validation_dataset_item(self, index: int) -> Dict[str, Any]:
        """获取训练集数据"""

    @staticmethod
    def get_test_dataset_size(self) -> int:
        """获取训练集大小"""

    @staticmethod
    def get_test_dataset_item(self, index: int) -> Dict[str, Any]:
        """获取训练集数据"""
