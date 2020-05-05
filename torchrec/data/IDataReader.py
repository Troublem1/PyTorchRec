"""
数据加载器接口
可以通过Dataset类与DataLoader类以批量形式提供数据
"""
import json
from abc import abstractmethod
from typing import Dict, Any

from torchrec.feature_columns import *
from torchrec.utils.argument import IWithArguments
from torchrec.utils.const import *


class IDataReader(IWithArguments):
    """数据读入接口类"""

    @staticmethod
    def get_columns(dataset_name) -> Dict[str, Any]:
        """获取数据集信息供其他模块使用"""
        dataset_description_json = os.path.join(DATASET_DIR, dataset_name, DESCRIPTION_JSON)
        with open(dataset_description_json) as dataset_description_json_file:
            dataset_description = json.load(dataset_description_json_file)

        columns = dict()

        # 处理基本信息
        base_column_map = {
            UID: "uid_column",
            IID: "iid_column",
            RATE: "rate_column",
            LABEL: "label_column",
            TIME: "time_column",
        }
        for column_description in dataset_description[BASE_FEATURES]:
            column_name = base_column_map[column_description[FEATURE_NAME]]
            columns[column_name] = CategoricalColumn.from_description_dict(column_description)

        # 处理用户/物品/上下文特征
        feature_columns_map = {
            USER_FEATURES: {
                NUMERIC_COLUMN: "user_numeric_columns",
                CATEGORICAL_COLUMN: "user_categorical_columns",
            },
            ITEM_FEATURES: {
                NUMERIC_COLUMN: "item_numeric_columns",
                CATEGORICAL_COLUMN: "item_categorical_columns",
            },
            CONTEXT_FEATURES: {
                NUMERIC_COLUMN: "context_numeric_columns",
                CATEGORICAL_COLUMN: "context_categorical_columns",
            }
        }
        for i in feature_columns_map:
            for j in feature_columns_map[i]:
                columns[feature_columns_map[i][j]] = list()
        for i in feature_columns_map:
            for column_description in dataset_description[i]:
                if column_description[FEATURE_TYPE] == NUMERIC_COLUMN:
                    columns[feature_columns_map[i][NUMERIC_COLUMN]].append(
                        NumericColumn.from_description_dict(column_description)
                    )
                elif column_description[FEATURE_TYPE] == CATEGORICAL_COLUMN:
                    columns[feature_columns_map[i][CATEGORICAL_COLUMN]].append(
                        CategoricalColumn.from_description_dict(column_description)
                    )
        return columns

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
