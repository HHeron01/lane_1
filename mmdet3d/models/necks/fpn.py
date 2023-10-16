# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from ..builder import NECKS


@NECKS.register_module()
class CustomFPN(BaseModule):
    r"""特征金字塔网络 (Feature Pyramid Network).
    这是论文 `Feature Pyramid Networks for Object Detection <https://arxiv.org/abs/1612.03144>`_ 的实现。

    参数 Args:
        in_channels (List[int]): 每个尺度的输入通道数。例如: in_channels:256/512
        out_channels (int): 输出通道数 (每个尺度上都使用)。例如: out_channels:64
        num_outs (int): 输出尺度的数量。例如: nums_out=1
        start_level (int): 用于构建特征金字塔的起始输入骨干层级的索引。默认: 0。例如: start_levels=0
        end_level (int): 用于构建特征金字塔的结束输入骨干层级的索引(不包括此层)。默认: -1，代表最后一层。
        add_extra_convs (bool | str): 决定是否在原始特征图上添加卷积层。默认为False。
            如果为True，等价于 `add_extra_convs='on_input'`。
            如果为str，指定额外卷积的源特征图。
            只允许以下选项：
            - 'on_input': 颈部输入的最后特征图 (即骨干特征)。
            - 'on_lateral': 侧边卷积后的最后特征图。
            - 'on_output': FPN卷积后的最后输出特征图。
        relu_before_extra_convs (bool): 在额外的卷积之前是否应用relu。默认: False。
        no_norm_on_lateral (bool): 是否在侧边上应用规范化。默认: False。
        conv_cfg (dict): 卷积层的配置字典。默认: None。
        norm_cfg (dict): 规范化层的配置字典。默认: None。
        act_cfg (str): ConvModule中激活层的配置字典。默认: None。
        upsample_cfg (dict): 插值层的配置字典。默认: `dict(mode='nearest')`
        init_cfg (dict or list[dict], optional): 初始化配置字典。

    示例 Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """
    def __init__(self,
                 in_channels,             # 输入通道数，应为列表形式，代表不同层级的输入特征图的通道数量
                 out_channels,            # 输出通道数
                 num_outs,                # 输出的数量
                 start_level=0,           # 开始的层级，默认从0开始
                 end_level=-1,            # 结束的层级，默认为-1，代表使用所有的输入层级
                 out_ids=[],              # 输出的ID列表，标记哪些层级需要输出
                 add_extra_convs=False,   # 是否在FPN的顶部添加额外的卷积层
                 relu_before_extra_convs=False,  # 在额外的卷积层前是否添加ReLU激活函数
                 no_norm_on_lateral=False,       # 在侧向连接上是否不使用归一化
                 conv_cfg=None,           # 卷积层的配置
                 norm_cfg=None,           # 归一化的配置
                 act_cfg=None,            # 激活函数的配置
                 upsample_cfg=dict(mode='nearest'),  # 上采样的配置，默认使用最近邻上采样
                 init_cfg=dict(           # 初始化配置，默认使用Xavier初始化
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        
        # 调用父类构造函数进行初始化
        super(CustomFPN, self).__init__(init_cfg)
        
        # 断言确保`in_channels`是列表形式
        assert isinstance(in_channels, list)

        # 根据参数初始化类的属性
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False  # 指示是否启用FP16精度。默认为False。
        self.upsample_cfg = upsample_cfg.copy()  # 复制上采样配置以避免意外修改
        self.out_ids = out_ids
        
        # 判断结束层级，并根据`end_level`设置背景的结束层级
        if end_level == -1:
            self.backbone_end_level = self.num_ins  # 如果`end_level`为-1，则使用所有输入层级
        else:
            # 如果`end_level`小于输入的数量，那么不允许有额外的层级
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level

        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        
        # 断言确保`add_extra_convs`为字符串或布尔值，并根据它的值进行相应的设置
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # 如果`add_extra_convs`是字符串，则它的值应为'on_input'、'on_lateral'或'on_output'之一
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # 如果为True
            self.add_extra_convs = 'on_input'

        # 定义两个模块列表：一个用于侧向连接的卷积，另一个用于FPN的卷积
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        # 对于指定的层级范围内的每一个层级
        for i in range(self.start_level, self.backbone_end_level):
            # 创建一个用于侧向连接的卷积模块
            l_conv = ConvModule(
                in_channels[i],              # 当前层级的输入通道数
                out_channels,                # 输出通道数
                1,                           # 卷积核大小
                conv_cfg=conv_cfg,           # 卷积配置
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,  # 若no_norm_on_lateral为True，则不使用归一化
                act_cfg=act_cfg,             # 激活函数配置
                inplace=False)               # 是否原地操作

            # 将侧向连接的卷积模块添加到lateral_convs列表中
            self.lateral_convs.append(l_conv)

            # 如果当前层级在输出ID列表中
            if i in self.out_ids:
                # 创建一个用于FPN的卷积模块
                fpn_conv = ConvModule(
                    out_channels,             # 输入通道数
                    out_channels,             # 输出通道数
                    3,                        # 卷积核大小
                    padding=1,                # 填充
                    conv_cfg=conv_cfg,        # 卷积配置
                    norm_cfg=norm_cfg,        # 归一化配置
                    act_cfg=act_cfg,          # 激活函数配置
                    inplace=False)            # 是否原地操作

                # 将FPN的卷积模块添加到fpn_convs列表中
                self.fpn_convs.append(fpn_conv)

        # 添加额外的卷积层，例如在RetinaNet中使用
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        # 如果设置了添加额外的卷积层并且额外的层级数大于等于1
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                # 如果是第一个额外的层级且添加位置是'on_input'
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]  # 输入通道数为上一个层级的输出通道数
                else:
                    in_channels = out_channels  # 否则，输入通道数为指定的输出通道数

                # 创建额外的FPN卷积模块
                extra_fpn_conv = ConvModule(
                    in_channels,               # 输入通道数
                    out_channels,              # 输出通道数
                    3,                         # 卷积核大小
                    stride=2,                  # 步长
                    padding=1,                 # 填充
                    conv_cfg=conv_cfg,         # 卷积配置
                    norm_cfg=norm_cfg,         # 归一化配置
                    act_cfg=act_cfg,           # 激活函数配置
                    inplace=False)             # 是否原地操作

                # 将额外的FPN卷积模块添加到fpn_convs列表中
                self.fpn_convs.append(extra_fpn_conv)

    @auto_fp16()
    def forward(self, inputs):
        """前向传播函数"""
        # 确保输入的数量与in_channels中定义的数量相等
        assert len(inputs) == len(self.in_channels)

        # 构建侧向连接
        # 对于每个输入，使用相应的侧向连接卷积
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # 构建自顶向下的路径
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # 在某些情况下，固定的`scale factor`（例如2）是首选
            # 但它不能与`F.interpolate`中的`size`同时存在。
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # 构建输出
        # 第一部分：来自原始层级的输出
        outs = [self.fpn_convs[i](laterals[i]) for i in self.out_ids]
        # 第二部分：添加额外的层级
        if self.num_outs > len(outs):
            # 使用最大池化在输出的顶部获得更多的层级（例如，Faster R-CNN，Mask R-CNN）
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # 在原始特征图的顶部添加卷积层（例如RetinaNet）
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return outs[0]
