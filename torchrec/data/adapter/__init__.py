"""
数据集适配器
将IDataReader的实现子类适配到torch.utils.data.Dataset
"""
from .DevDataset import DevDataset
from .TestDataset import TestDataset
from .TrainDataset import TrainDataset
