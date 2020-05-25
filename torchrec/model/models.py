from typing import Dict, Type

from torchrec.data.IDataReader import IDataReader
from torchrec.data.SimpleDataReader import SimpleDataReader
from torchrec.model.FunkSVD import FunkSVD
from torchrec.model.IModel import IModel

_model_classes: Dict[str, Type[IModel]] = {
    "funksvd": FunkSVD,
}

_model_to_data_reader: Dict[Type[IModel], Type[IDataReader]] = {
    FunkSVD: SimpleDataReader,
}

model_name_list = _model_classes.keys()


def get_model_type(model_name: str) -> Type[IModel]:
    """根据模型名称返回模型类"""
    if (not isinstance(model_name, str)) or (model_name not in _model_classes):
        raise ValueError(f"model_name参数不合法: {model_name}")
    return _model_classes[model_name]


def get_data_reader_type(model_type: Type[IModel]) -> Type[IDataReader]:
    """根据模型类型返回数据加载器类型"""
    if model_type not in _model_to_data_reader:
        raise ValueError(f"model_type参数不合法: {model_type}")
    return _model_to_data_reader[model_type]
