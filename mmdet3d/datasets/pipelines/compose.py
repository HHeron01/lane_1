# Copyright (c) OpenMMLab. All rights reserved.
import collections

from mmcv.utils import build_from_cfg

from mmdet.datasets.builder import PIPELINES as MMDET_PIPELINES
from ..builder import PIPELINES


# 注册Compose类，用于按顺序组合多个数据变换操作的管道
@PIPELINES.register_module()
class Compose:
    """按顺序组合多个变换。mmdet3d的管道注册与mmdet分开，然而，有时我们可能需要使用mmdet的管道。
    因此，重写了这个类，以便能够使用mmdet3d和mmdet中的管道。

    Args:
        transforms (Sequence[dict | callable]): 要组合的变换对象或配置字典的序列。
    """

    def __init__(self, transforms):
        # 断言transforms是collections.abc.Sequence的实例，即一个序列
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                # 从配置字典中解析出变换的类型
                _, key = PIPELINES.split_scope_key(transform['type'])
                if key in PIPELINES._module_dict.keys():
                    # 如果类型在PIPELINES中，则从PIPELINES中构建变换
                    transform = build_from_cfg(transform, PIPELINES)
                else:
                    # 否则从MMDET_PIPELINES中构建变换
                    transform = build_from_cfg(transform, MMDET_PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                # 如果transform是可调用对象，直接添加到transforms列表中
                self.transforms.append(transform)
            else:
                # 如果transform既不是字典也不是可调用对象，则抛出TypeError异常
                raise TypeError('transform must be callable or a dict')

    def __call__(self, data):
        """调用函数按顺序应用变换。
        Args:
            data (dict): 包含要进行变换的数据的结果字典。
        Returns:
           dict: 变换后的数据。
        """
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string
