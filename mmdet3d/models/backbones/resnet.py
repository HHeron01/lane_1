# Copyright (c) Phigent Robotics. All rights reserved.

import torch.utils.checkpoint as checkpoint
from torch import nn

from mmdet.models import BACKBONES
from mmdet.models.backbones.resnet import BasicBlock, Bottleneck


@BACKBONES.register_module()
class CustomResNet(nn.Module):
    """自定义的ResNet模型类。"""
    def __init__(
            self,
            numC_input,
            num_layer=[2, 2, 2],
            num_channels=None,
            stride=[2, 2, 2],
            backbone_output_ids=None,
            norm_cfg=dict(type='BN'),
            with_cp=False,
            block_type='Basic',
            oft=False,
            height=6,
    ):
        super(CustomResNet, self).__init__()

        # 构建ResNet模型的各种配置参数
        assert len(num_layer) == len(stride)
        # 如果num_channels为None，则根据num_layer自动生成
        num_channels = [numC_input*2**(i+1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        # 如果backbone_output_ids为None，则默认为[0, 1, 2]
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids
        layers = []

        # 根据block_type构建ResNet的每个阶段
        # 根据 block_type 参数构建 ResNet 每个阶段
        if block_type == 'BottleNeck':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                # 创建一个包含一个 Bottleneck 残差块的列表
                # 每个 Bottleneck 残差块由下采样、卷积层和标准化层组成
                layer = [
                    Bottleneck(
                        curr_numC,
                        num_channels[i] // 4,
                        stride=stride[i],
                        downsample=nn.Conv2d(curr_numC, num_channels[i], 3,
                                            stride[i], 1),
                        norm_cfg=norm_cfg)
                ]
                # 更新当前通道数
                curr_numC = num_channels[i]
                # 添加额外的 Bottleneck 残差块到当前阶段
                layer.extend([
                    Bottleneck(curr_numC, curr_numC // 4, norm_cfg=norm_cfg)
                    for _ in range(num_layer[i] - 1)
                ])
                # 将当前阶段添加到层列表中
                layers.append(nn.Sequential(*layer))
        elif block_type == 'Basic':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                # 创建一个包含一个 BasicBlock 残差块的列表
                # 每个 BasicBlock 残差块由下采样、卷积层和标准化层组成
                layer = [
                    BasicBlock(
                        curr_numC,
                        num_channels[i],
                        stride=stride[i],
                        downsample=nn.Conv2d(curr_numC, num_channels[i], 3,
                                            stride[i], 1),
                        norm_cfg=norm_cfg)
                ]
                # 更新当前通道数
                curr_numC = num_channels[i]
                # 添加额外的 BasicBlock 残差块到当前阶段
                layer.extend([
                    BasicBlock(curr_numC, curr_numC, norm_cfg=norm_cfg)
                    for _ in range(num_layer[i] - 1)
                ])
                # 将当前阶段添加到层列表中
                layers.append(nn.Sequential(*layer))
        else:
            # 如果 block_type 不是 Basic 或 BottleNeck，触发断言错误
            assert False

        # 存储ResNet的各个阶段
        self.layers = nn.Sequential(*layers)

        self.with_cp = with_cp

        self.oft = oft

        # 如果oft为True，添加一层卷积和标准化层
        if oft:
            self.conv_1 = nn.Conv2d(in_channels=numC_input * height, out_channels=numC_input, kernel_size=1, stride=stride[0], bias=False)
            self.bn_1 = nn.BatchNorm2d(numC_input)

    def forward(self, x):
        # 前向传播函数
        if self.oft:
            x = self.conv_1(x)
            x = self.bn_1(x)
        feats = []
        x_tmp = x

        # 遍历ResNet的各个阶段
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                # 使用checkpoint来减少内存占用#
                # 将前向传播过程中的中间状态（activations）保存到磁盘或显存中，而不是一次性保存在内存中
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats
