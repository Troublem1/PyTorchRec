"""
使用参数的接口类
"""
from typing import List, Dict, Any

from .description import ArgumentDescription


class IWithArguments:
    """使用参数的接口类"""

    @classmethod
    def get_argument_descriptions(cls) -> List[ArgumentDescription]:
        """获取参数描述信息"""
        raise NotImplementedError

    @classmethod
    def check_argument_values(cls, arguments: Dict[str, Any]) -> None:
        """检查参数值"""
        # 检查参数描述设置部分
        argument_descriptions = cls.get_argument_descriptions()
        for description in argument_descriptions:
            description.check_value(arguments[description.name])
        # 自定义检查部分，由实现类实现
        pass
