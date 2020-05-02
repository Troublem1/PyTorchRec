"""
参数描述类，以兼容多种参数输入形式
"""
from typing import Type


class ArgumentDescription:
    """参数描述信息"""
    __type_set = {str, int, float, bool}
    __number_type_set = {int, float}

    @classmethod
    def __is_number_type(cls, value) -> bool:
        """判断是否为数字类型"""
        for type_ in cls.__number_type_set:
            if isinstance(value, type_):
                return True
        return False

    def __init__(self, name: str, type_: Type, help_info: str, is_logged: bool = True, default_value=None,
                 legal_value_list=None, lower_open_bound=None, lower_closed_bound=None, upper_open_bound=None,
                 upper_closed_bound=None):
        """
        参数的基本信息
        :param name: 参数名
        :param type_: 参数类型，允许str、int、float、bool四种类型
        :param help_info: 帮助信息
        :param is_logged: 是否记录（一些无关参数比如GPU序号不需要被保存）
        :param default_value: 默认值（可选）
        :param legal_value_list: 取值列表（可选）
        :param lower_open_bound: 下界（开区间），只能用于数值类型，会被legal_value_list设置覆盖（可选）
        :param lower_closed_bound: 下界（闭区间），只能用于数值类型，会被legal_value_list设置覆盖（可选）
        :param upper_open_bound: 上界（开区间），只能用于数值类型，会被legal_value_list设置覆盖（可选）
        :param upper_closed_bound: 上界（闭区间），只能用于数值类型，会被legal_value_list设置覆盖（可选）
        """
        assert type_ in self.__type_set
        if default_value:
            assert isinstance(default_value, type_)
        if legal_value_list:
            for legal_value in legal_value_list:
                assert isinstance(legal_value, type_)
            lower_open_bound = None
            lower_closed_bound = None
            upper_open_bound = None
            upper_closed_bound = None
        if lower_open_bound or lower_closed_bound or upper_open_bound or upper_closed_bound:
            assert type_ in self.__number_type_set
        if lower_open_bound:
            assert self.__is_number_type(lower_open_bound)
        if lower_closed_bound:
            assert self.__is_number_type(lower_closed_bound)
        if upper_open_bound:
            assert self.__is_number_type(upper_open_bound)
        if upper_closed_bound:
            assert self.__is_number_type(upper_closed_bound)
        self.name = name
        self.type = type_
        self.help_info = help_info
        self.is_logged = is_logged
        self.default_value = default_value
        self.legal_value_list = legal_value_list
        self.lower_open_bound = lower_open_bound
        self.lower_closed_bound = lower_closed_bound
        self.upper_open_bound = upper_open_bound
        self.upper_closed_bound = upper_closed_bound
