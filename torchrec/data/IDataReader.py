"""
数据加载器接口
可以通过Dataset类与DataLoader类以批量形式提供数据
"""
import json
from abc import abstractmethod
from typing import Dict, Any

from torchrec.utils.argument import IWithArguments
from torchrec.utils.const import *


class IDataReader(IWithArguments):
    """数据读入接口类"""

    @staticmethod
    def get_dataset_info(dataset_name) -> Dict[str, Any]:
        """获取数据集信息供其他模块使用"""
        dataset_description_json = os.path.join(DATASET_DIR, dataset_name, DESCRIPTION_JSON)
        with open(dataset_description_json) as dataset_description_json_file:
            dataset_description = json.load(dataset_description_json_file)
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
