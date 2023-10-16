      
# Copyright (c) Phigent Robotics. All rights reserved.
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32

from mmdet.models import DETECTORS
from .. import builder
from mmdet3d.models.detectors.base import Base3DDetector

@DETECTORS.register_module()  # 使用装饰器注册为目标检测器
class LaneAF2DDetector(Base3DDetector):  # 继承自Base3DDetector类
    def __init__(self, img_backbone, img_neck, lane_head, train_cfg, test_cfg, pretrained=None, **kwargs):
        super(LaneAF2DDetector, self).__init__(**kwargs)  # 调用父类的初始化方法

        # 如果传入了img_backbone参数，则使用builder.build_backbone方法构建图像骨干网络
        if img_backbone:
            self.img_backbone = builder.build_backbone(img_backbone)
        
        # 如果传入了img_neck参数，则使用builder.build_neck方法构建图像颈部网络
        if img_neck is not None:
            self.img_neck = builder.build_neck(img_neck)

        # 使用builder.build_head方法构建车道线检测头
        self.lane_head = builder.build_head(lane_head)

        self.train_cfg = train_cfg  # 训练配置
        self.test_cfg = test_cfg    # 测试配置
        self.pretrained = pretrained  # 预训练模型的路径

    def image_encoder(self, img):
        # 获取输入图像
        imgs = img  #(640, 960)
        
        # 获取图像的形状，其中：
        # B 为 batch size（批次大小）
        # N 为图像数量
        # C 为通道数量（例如RGB图像C=3）
        # imH 为图像高度
        # imW 为图像宽度
        B, N, C, imH, imW = imgs.shape
        
        # 将图像的形状从 [B, N, C, imH, imW] 调整为 [B*N, C, imH, imW]，这样可以同时处理所有图像
        imgs = imgs.view(B * N, C, imH, imW)
        
        # 使用backbone网络对图像进行特征提取，返回的可能是多尺度的特征图
        x = self.img_backbone(imgs)  # 例如：[[8, 64, 160, 240], [8, 128, 80, 120], [8, 256, 40, 60] [8, 512, 20, 30]]
        
        # 如果有定义img_neck（例如，用于特征融合的结构），则进行处理
        if self.with_img_neck:       
            x = self.img_neck(x)  # (8, 256, 160, 240)
            
            # 如果返回的是多个特征图的列表或元组，只取第一个
            if type(x) in [list, tuple]:
                x = x[0]
        
        # 获取处理后的特征图的形状
        _, output_dim, ouput_H, output_W = x.shape
        
        # 如果原始输入中每个batch包含多于1个图像，调整形状回到原始的batch形式
        if N!=1:
            x = x.view(B, N, output_dim, ouput_H, output_W)
        
        return x

    def extract_img_feat(self, img, img_metas, **kwargs):
        """
        提取图像的特征。
        Args:
            img (tuple): 包含多种图像和相关参数的元组。
            img_metas (type not specified): 图像的元信息。
            **kwargs: 其他关键字参数。
        Returns:
            torch.Tensor: 提取的图像特征。
        """
        # features, ks, imu2cs, post_rots, post_trans, undists
        # 从输入img中解包各种参数和信息
        # ks: 内参矩阵
        # imu2cs: IMU到摄像机的转换矩阵
        # post_rots: 位姿旋转
        # post_trans: 位姿平移
        # undists: 非失真参数（可能用于校正镜头畸变）
        
        ks, imu2cs, post_rots, post_trans, undists = img[1], img[2], img[3], img[4], img[5]
        
        # grid: 网格信息
        # drop_idx: 被舍弃的索引信息
        grid, drop_idx = img[9], img[10]
        
        # 使用image_encoder方法提取图像特征
        x = self.image_encoder(img[0])
        
        return x
    
    #   output['img_inputs'] = (imgs, rots, trans, intrins, post_rots,
    #                                  post_trans, bda_rot, extrinsics, undists, gt_lanes)
    # output['maps'] = (gt_mask, mask_haf, mask_vaf, mask_offset)
    # 被注释掉的部分提供了输入的结构信息，以帮助理解img元组中的内容：
    # img_inputs: 输入的图像和相关参数。包括：
    # imgs: 图像数据
    # rots: 旋转参数
    # trans: 平移参数
    # intrins: 内部参数（或内参）
    # post_rots: 位姿旋转
    # post_trans: 位姿平移
    # bda_rot: BDA的旋转参数
    # extrinsics: 外部参数
    # undists: 非失真参数
    # gt_lanes: 真实车道线信息

    # maps: 相关的映射信息，包括：
    # gt_mask: 真实的掩码
    # mask_haf: HAF的掩码
    # mask_vaf: VAF的掩码
    # mask_offset: 偏移掩码

    def extract_feat(self, img, img_metas, **kwargs):
        """
        从图像和点云数据中提取特征。
        Args:
            img (type not specified): 输入图像数据。
            img_metas (type not specified): 图像的元信息。
            **kwargs: 其他关键字参数。
        Returns:
            torch.Tensor: 提取的图像特征。
        """
        
        # 使用extract_img_feat方法提取图像特征
        img_feats = self.extract_img_feat(img, img_metas, **kwargs)
        
        return img_feats

    def forward_lane_train(self, img_feats, gt_labels, **kwargs):
        """
        训练过程中的前向传播函数，专门用于处理车道线任务。
        Args:
            img_feats (torch.Tensor): 从输入图像中提取的特征。
            gt_labels (list of torch.Tensor): 真实标签，包括车道线的分割掩码、水平方向的掩码、垂直方向的掩码和偏移掩码。
            **kwargs: 其他关键字参数。
        Returns:
            dict: 包含损失信息的字典。
            torch.Tensor: 网络输出的预测值。
        """
        #NOTE: input = [seg_mask, haf_mask, vaf_mask, mask_offset]
        # array1 = maps[0].detach().cpu().numpy()
        # cv2.imwrite("./map.png", np.argmax(array1, axis=0) * 100)
        # NOTE: 输入数据的结构为：[分割掩码, 水平方向的掩码, 垂直方向的掩码, 偏移掩码]
        # 使用lane_head网络部分处理图像特征
        outs = self.lane_head(img_feats)
        # 准备输入到损失函数的数据
        loss_inputs = [outs, gt_labels]
        # 将输出和真实标签传递给损失函数以计算损失
        # 这里注释掉了一个可能用于计算损失的函数，但最终选择使用lane_head的损失函数计算损失
        # losses = self.bev_lane_loss(*loss_inputs)
        losses = self.lane_head.loss(*loss_inputs)
        
        # 返回损失和网络的输出
        return losses, outs


    def forward_train(self,
                  img_metas=None,
                  img_inputs=None,
                  maps_bev=None,
                  maps_2d=None,
                  label_bev=None,
                  label_2d=None,
                  **kwargs):
        """
        训练过程中的前向传播函数。
        Args:
            img_metas (list): 各图像的元数据信息，例如尺寸、缩放因子等。
            img_inputs (torch.Tensor): 输入的图像数据。
            maps_bev (torch.Tensor): BEV (Bird's Eye View) 的映射数据，可能用于其他任务，这里没有用到。
            maps_2d (torch.Tensor): 2D图像的映射数据，用于车道线检测。
            label_bev (list): BEV的标签数据，可能用于其他任务，这里没有用到。
            label_2d (list): 2D图像的标签数据，用于车道线检测。
            **kwargs: 其他关键字参数。
        Returns:
            dict: 包含损失信息的字典。
            torch.Tensor: 网络输出的预测值。
        """
        
        # 从输入图像中提取特征
        img_feats = self.extract_feat(img=img_inputs, img_metas=img_metas, **kwargs)
        # 初始化损失字典
        losses = dict()
        # 使用专门的车道线训练前向传播函数处理图像特征，并获取车道线的损失和输出
        losses_lane, out = self.forward_lane_train(img_feats, maps_2d)
        # 将车道线的损失添加到总损失字典中
        losses.update(losses_lane)
        # 返回总损失和网络的输出
        return losses, out

    def forward_test(self,
                      img_metas=None,
                      img_inputs=None,
                      maps_bev=None,
                      maps_2d=None,
                      **kwargs):
        return self.simple_test(img_metas, img_inputs, maps_2d)

    def aug_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        assert False

    def simple_test(self,
                img_metas=None,
                img_inputs=None,
                gt_labels=None,
                **kwargs):
        """
        简化的测试过程函数。
        Args:
            img_metas (list): 各图像的元数据信息，例如尺寸、缩放因子等。
            img_inputs (torch.Tensor): 输入的图像数据。
            gt_labels (list): 真实标签数据，用于评估或其他目的，但在此函数内部没有使用。
            **kwargs: 其他关键字参数。
        Returns:
            torch.Tensor: 车道线检测的网络输出结果。
        """
        
        # 从输入图像中提取特征
        img_feats = self.extract_feat(img=img_inputs, img_metas=img_metas, **kwargs)
        # 通过车道线头部进行前向传播，获取输出结果
        outs = self.lane_head(img_feats)
        # 返回网络的输出结果
        return outs

    def forward_dummy(self,
                  img_metas=None,
                  img_inputs=None,
                  maps_bev=None,
                  maps_2d=None,
                  **kwargs):
        """
        虚拟的前向传播函数。
        该函数通常用于模型的性能基准测试，不实际考虑模型的输出内容，只关心模型的运行速度。
        Args:
            img_metas (list, optional): 各图像的元数据信息，例如尺寸、缩放因子等。
            img_inputs (torch.Tensor, optional): 输入的图像数据。
            maps_bev (torch.Tensor, optional): BEV（Bird's Eye View，俯视图）的地图数据，本函数中未使用。
            maps_2d (torch.Tensor, optional): 2D地图数据，本函数中未使用。
            **kwargs: 其他关键字参数。
        Returns:
            torch.Tensor: 车道线检测的网络输出结果。
        """
        
        # 从输入图像中提取特征，但忽略其他返回值（例如点云特征）
        img_feats, _ = self.extract_feat(img=img_inputs, img_metas=img_metas, **kwargs)
        # 通过车道线头部进行前向传播，获取输出结果
        outs = self.lane_head(img_feats)
        # 返回网络的输出结果
        return outs

    @property
    def with_img_neck(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, 'img_neck') and self.img_neck is not None


# 使用DETECTORS装饰器进行注册，使得LaneATT2DDetector可以被自动识别和使用
@DETECTORS.register_module()
class LaneATT2DDetector(LaneAF2DDetector):
    def __init__(self, img_backbone, img_neck, lane_head, train_cfg, test_cfg, pretrained=None, **kwargs):
        """
        车道线ATT2D检测器初始化函数。
        Args:
            img_backbone (dict): 图像的主干网络配置。
            img_neck (dict): 图像的颈部网络配置。
            lane_head (dict): 车道线头部网络配置。
            train_cfg (dict): 训练相关的配置参数。
            test_cfg (dict): 测试相关的配置参数。
            pretrained (str, optional): 预训练模型的路径。
            **kwargs: 其他关键字参数。
        """
        # 通过父类进行初始化
        super(LaneATT2DDetector, self).__init__(img_backbone, img_neck, lane_head, train_cfg, test_cfg, pretrained=None, **kwargs)

    def forward_train(self,
                      img_metas=None,
                      img_inputs=None,
                      maps_bev=None,
                      maps_2d=None,
                      label_bev=None,
                      label_2d=None,
                      **kwargs):
        """
        训练模式下的前向传播函数。
        Args:
            img_metas (list, optional): 各图像的元数据信息，例如尺寸、缩放因子等。
            img_inputs (torch.Tensor, optional): 输入的图像数据。
            maps_bev (torch.Tensor, optional): BEV（Bird's Eye View，俯视图）的地图数据。
            maps_2d (torch.Tensor, optional): 2D地图数据。
            label_bev (torch.Tensor, optional): BEV的标签数据。
            label_2d (torch.Tensor, optional): 2D的标签数据。
            **kwargs: 其他关键字参数。
        Returns:
            dict: 各部分的损失。
            torch.Tensor: 车道线检测的网络输出结果。
        """
        # 从输入图像中提取特征
        img_feats = self.extract_feat(img=img_inputs, img_metas=img_metas, **kwargs)
        losses = dict()
        # 使用特征进行车道线的前向训练，并获取损失及输出
        losses_lane, out = self.forward_lane_train(img_feats, label_2d)
        # 更新损失字典
        losses.update(losses_lane)
        
        return losses, out

    def forward_test(self,
                      img_metas=None,
                      img_inputs=None,
                      maps_bev=None,
                      maps_2d=None,
                      label_bev=None,
                      label_2d=None,
                      **kwargs):
        """
        测试模式下的前向传播函数。
        Args:
            img_metas (list, optional): 各图像的元数据信息，例如尺寸、缩放因子等。
            img_inputs (torch.Tensor, optional): 输入的图像数据。
            maps_bev (torch.Tensor, optional): BEV（Bird's Eye View，俯视图）的地图数据。
            maps_2d (torch.Tensor, optional): 2D地图数据。
            label_bev (torch.Tensor, optional): BEV的标签数据。
            label_2d (torch.Tensor, optional): 2D的标签数据。
            **kwargs: 其他关键字参数。
        Returns:
            torch.Tensor: 车道线检测的网络输出结果。
        """
        return self.simple_test(img_metas, img_inputs, label_2d)
