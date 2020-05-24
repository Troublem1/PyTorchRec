"""
任务类接口
"""
from abc import ABC, abstractmethod

from torchrec.utils.argument.IWithArguments import IWithArguments


class ITask(IWithArguments, ABC):
    """任务类接口"""

    @abstractmethod
    def run(self):
        """执行任务"""
        pass
