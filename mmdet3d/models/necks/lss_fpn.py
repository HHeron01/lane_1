# Copyright (c) Phigent Robotics. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer

from mmdet.models import NECKS

@NECKS.register_module()
class FPN_LSS(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor=4,
                 input_feature_index=(0, 2),
                 norm_cfg=dict(type='BN'),
                 extra_upsample=2,
                 lateral=None,
                 use_input_conv=False):
        super().__init__()

        # 保存输入特征图的索引
        self.input_feature_index = input_feature_index
        # 确定是否需要额外的上采样
        self.extra_upsample = extra_upsample is not None
        # 创建上采样层，用于将特征图的大小放大
        self.up = nn.Upsample(
            scale_factor=scale_factor, mode='bilinear', align_corners=True)
        # assert norm_cfg['type'] in ['BN', 'SyncBN']

        # 如果需要使用输入卷积，则创建一个输入卷积层
        channels_factor = 2 if self.extra_upsample else 1
        if use_input_conv:
            in_channels = out_channels * channels_factor
            self.input_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * channels_factor,
                    kernel_size=1,
                    padding=0,
                    bias=False),
                build_norm_layer(
                    norm_cfg, out_channels * channels_factor, postfix=0)[1],
                nn.ReLU(inplace=True),
            )
        else:
            self.input_conv = None

        # 创建一个卷积层序列，用于特征处理
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels * channels_factor,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(
                norm_cfg, out_channels * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels * channels_factor,
                out_channels * channels_factor,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(
                norm_cfg, out_channels * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
        )

        # 如果需要额外的上采样，则创建一个上采样层序列
        if self.extra_upsample:
            self.up2 = nn.Sequential(
                nn.Upsample(
                    scale_factor=extra_upsample,
                    mode='bilinear',
                    align_corners=True),
                nn.Conv2d(
                    out_channels * channels_factor,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False),
                build_norm_layer(norm_cfg, out_channels, postfix=0)[1],
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    out_channels, out_channels, kernel_size=1, padding=0),
            )

        # 是否需要进行侧边连接
        self.lateral = lateral is not None
        if self.lateral:
            self.lateral_conv = nn.Sequential(
                nn.Conv2d(
                    lateral, lateral, kernel_size=1, padding=0, bias=False),
                build_norm_layer(norm_cfg, lateral, postfix=0)[1],
                nn.ReLU(inplace=True),
            )

    def forward(self, feats):
        # 获取输入特征图
        x2, x1 = feats[self.input_feature_index[0]], feats[self.input_feature_index[1]]

        # 如果需要进行侧边连接，则进行侧边连接操作
        if self.lateral:
            x2 = self.lateral_conv(x2)
        
        # 上采样操作
        x1 = self.up(x1)
        # x1 = F.interpolate(x1, size=x2.shape[-2:], mode='bilinear', align_corners=True)
        
        # 在通道维度上连接特征图
        x = torch.cat([x2, x1], dim=1)
        # 如果使用了输入卷积，则应用输入卷积
        if self.input_conv is not None:
            x = self.input_conv(x)

        # 应用卷积层进行特征处理
        x = self.conv(x)
        # 如果需要额外的上采样，则应用上采样层
        if self.extra_upsample:
            x = self.up2(x)

        return x
