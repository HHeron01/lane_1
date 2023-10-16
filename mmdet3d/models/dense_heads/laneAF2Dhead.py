      
# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@Author  :   wangjing
@Version :   0.1
@License :   (C)Copyright 2019-2035
@Desc    :   None
"""
import torch
import torch.nn as nn
from mmdet3d.models.builder import HEADS, build_loss
import numpy as np
import cv2

@HEADS.register_module()
class LaneAF2DHead(nn.Module):
    """
    一个用于车道检测的2D头部网络。它预测了车道的二值分割、水平透视场 (HAF) 和垂直透视场 (VAF)。
    
    属性:
    - num_classes: 分割头的类别数。
    - inner_channel: 中间卷积层的通道数。
    - seg_loss: 用于分割的损失函数。
    - haf_loss: 用于HAF的损失函数。
    - vaf_loss: 用于VAF的损失函数。
    - binary_seg: 用于预测车道分割的网络层。
    - haf_head: 用于预测HAF的网络层。
    - vaf_head: 用于预测VAF的网络层。
    - debug: 如果为True，则会在debug模式下运行，可能包括一些额外的日志或检查。
    """

    def __init__(
        self,
        num_classes=1,
        in_channel=64,
        debug=False,
        seg_loss=dict(type='Lane_FocalLoss'),
        haf_loss=dict(type='RegL1Loss'),
        vaf_loss=dict(type='RegL1Loss'),
    ):
        """
        初始化函数。
        
        参数:
        - num_classes: 分割任务的类别数，默认为1。
        - in_channel: 输入的通道数，默认为64。
        - debug: 是否开启调试模式，默认为False。
        - seg_loss: 用于分割任务的损失函数的配置。
        - haf_loss: 用于HAF任务的损失函数的配置。
        - vaf_loss: 用于VAF任务的损失函数的配置。
        """

        # 初始化父类的构造函数
        super().__init__()
        
        # 定义类属性
        self.num_classes = num_classes
        self.inner_channel = in_channel

        # 构建损失函数
        self.seg_loss = build_loss(seg_loss)
        self.haf_loss = build_loss(haf_loss)
        self.vaf_loss = build_loss(vaf_loss)

        # 定义二值分割头
        self.binary_seg = nn.Sequential(
            nn.Conv2d(in_channel, self.inner_channel, 1),
            nn.BatchNorm2d(self.inner_channel),
            nn.ReLU(),
            nn.Conv2d(self.inner_channel, self.num_classes, 1),
            # nn.Sigmoid(),
        )

        # 定义HAF头
        self.haf_head = nn.Sequential(
            nn.Conv2d(in_channel, self.inner_channel, 1),
            nn.BatchNorm2d(self.inner_channel),
            nn.ReLU(),
            nn.Conv2d(self.inner_channel, 1, 1),
        )
        
        # 定义VAF头
        self.vaf_head = nn.Sequential(
            nn.Conv2d(in_channel, self.inner_channel, 1),
            nn.BatchNorm2d(self.inner_channel),
            nn.ReLU(),
            nn.Conv2d(self.inner_channel, 2, 1),
        )

        # 如果开启调试模式，定义一个交叉熵损失函数
        self.debug = debug
        if self.debug:
            self.debug_loss = nn.CrossEntropyLoss()

    def loss(self, preds_dicts, gt_labels, **kwargs):
        """
        计算LaneAF2DHead模块的损失。
        参数:
        - preds_dicts (tuple): 一个包含模块预测的输出的元组，包括二值分割、HAF预测、VAF预测和topdown。
        - gt_labels (tuple): 一个包含真实标签的元组，包括真实的分割掩码、HAF掩码和VAF掩码。
        - **kwargs: 其他可能需要的关键字参数。
        返回:
        - loss_dict (dict): 包含各种损失的字典。键为损失的名称，值为相应的损失值。
        """
        
        # 初始化损失字典
        loss_dict = dict()
        # 解包预测和真实标签的元组
        binary_seg, haf_pred, vaf_pred, topdown = preds_dicts
        gt_mask, mask_haf, mask_vaf = gt_labels
        # 获取设备信息（例如，CPU或GPU）
        device = binary_seg.device
        # 将真实的分割掩码转移到相应的设备
        maps = gt_mask.to(device)
        # 增加HAF掩码的维度
        mask_haf = torch.unsqueeze(mask_haf, 1)
        
        # 计算HAF的损失
        # haf_loss = self.haf_loss(haf_pred, mask_haf, binary_seg)
        haf_loss = self.haf_loss(haf_pred, mask_haf, gt_mask)
        # 计算VAF的损失
        # vaf_loss = self.vaf_loss(vaf_pred, mask_vaf, binary_seg)
        vaf_loss = self.vaf_loss(vaf_pred, mask_vaf, gt_mask)
        # 计算二值分割的损失
        seg_loss = self.seg_loss(binary_seg, maps)

        # 将各个损失乘以10并存储到损失字典中
        loss_dict['haf_loss'] = haf_loss * 10.0
        loss_dict['vaf_loss'] = vaf_loss * 10.0
        loss_dict['seg_loss'] = seg_loss * 10.0

        # 计算总损失并存储到损失字典中
        loss_dict['loss'] = (2 * haf_loss + 2 * vaf_loss + 8 * seg_loss) * 10.0

        return loss_dict

    def forward(self, topdown):
        """
        前向传播函数。当我们向模型提供一个输入时，它将通过这个函数进行处理。
        参数:
        - topdown (Tensor): 上下文特征映射。
        返回:
        - lane_head_output (tuple): 包含三个部分的输出元组: 二值分割, HAF和VAF，以及输入的上下文特征映射。
        """
        
        # 通过二值分割头部进行处理，获得二值分割的结果
        binary_seg = self.binary_seg(topdown)
        # 通过HAF头部进行处理，获得HAF的结果
        haf = self.haf_head(topdown)
        # 通过VAF头部进行处理，获得VAF的结果
        vaf = self.vaf_head(topdown)
        # 将所有结果整合成一个输出元组
        lane_head_output = binary_seg, haf, vaf, topdown

        return lane_head_output

    def tensor2image(self, tensor, mean, std):
        """
        将给定的Tensor转换为图片格式。
        参数:
        - tensor (Tensor): 需要转换的张量。
        - mean (array): 均值，用于逆归一化。
        - std (array): 标准差，用于逆归一化。
        返回:
        - image (array): 从张量转换而来的图片。
        """

        # 为均值和标准差增加维度，使它们能与张量匹配
        mean = mean[..., np.newaxis, np.newaxis] 
        mean = np.tile(mean, (1, tensor.size()[2], tensor.size()[3])) 
        std = std[..., np.newaxis, np.newaxis]
        std = np.tile(std, (1, tensor.size()[2], tensor.size()[3])) 

        # 逆归一化
        image = 255.0*(std*tensor[0].cpu().float().numpy() + mean)

        # 如果图像是单通道的，复制通道以形成三通道图像
        if image.shape[0] == 1:
            image = np.tile(image, (3, 1, 1))

        # 调整通道的顺序，从(CHW)转换为(HWC)
        image = np.transpose(image, (1, 2, 0))
        # 将RGB格式转换为BGR格式
        image = image[:, :, ::-1]

        # 返回uint8格式的图像
        return image.astype(np.uint8)

    def decodeAFs(BW, VAF, HAF, fg_thresh=128, err_thresh=5, viz=True):
        """
        从给定的二值分割图(BW)和透视场(VAF and HAF)解码车道线。
        参数:
        - BW: 二值分割图，其中车道线像素为前景。
        - VAF: 垂直透视场
        - HAF: 水平透视场
        - fg_thresh: 用于确定BW中的前景像素的阈值。
        - err_thresh: 确定车道线聚类的阈值。
        - viz: 是否可视化解码过程。
        返回:
        - 输出图，其中每个像素都标有一个车道ID。
        """
        # 初始化输出数组为0，其大小与 BW 相同  # BW分割
        output = np.zeros_like(BW, dtype=np.uint8)
        # 用于存储每条车道的末端点的列表
        lane_end_pts = [] 
        # 定义下一个可用的车道ID
        next_lane_id = 1
        
        # 可视化给定的二值图像
        if viz:
            im_color = cv2.applyColorMap(BW, cv2.COLORMAP_JET)
            cv2.imshow('BW', im_color)
            ret = cv2.waitKey(0)
    
        # 从最后一行开始解码到第一行
        # 求解每一行的中心点
        for row in range(BW.shape[0]-1, -1, -1):
            # 获取当前行中的前景像素列，即车道线像素
            cols = np.where(BW[row, :] > fg_thresh)[0]
            # 初始化簇/集群
            clusters = [[]]     
            
            # 如果存在前景像素，则初始化 prev_col 为第一个前景像素列的位置
            if cols.size > 0:
                prev_col = cols[0]
                
            # 水平地解析像素
            for col in cols:
                # 如果当前列与上一个列之间的差值大于给定的阈值，则视为新的集群开始
                if col - prev_col > err_thresh:
                    clusters.append([])  # 新开一个聚类
                    clusters[-1].append(col)
                    prev_col = col
                    continue
                
                # 根据水平透视场（HAF）的值，确定像素点是如何与其它像素相关联的
                if HAF[row, prev_col] >= 0 and HAF[row, col] >= 0: 
                    # 继续向右移动
                    clusters[-1].append(col)
                    prev_col = col
                    continue
                elif HAF[row, prev_col] >= 0 and HAF[row, col] < 0: 
                    # 找到了车道的中心，处理垂直透视场（VAF）
                    clusters[-1].append(col)
                    prev_col = col
                elif HAF[row, prev_col] < 0 and HAF[row, col] >= 0: 
                    # 找到车道的末端，生成新的车道
                    clusters.append([])
                    clusters[-1].append(col)
                    prev_col = col
                    continue
                elif HAF[row, prev_col] < 0 and HAF[row, col] < 0: 
                    # 继续向右移动
                    clusters[-1].append(col)
                    prev_col = col
                    continue
                
            # vaf与haf中心点差距
            # 上一行指向的有一个值和本行估计的值进行就差距，在一定范围内则连成一条线
            # 行列嵌套循环  
            # 分配线的lane id
            # 建立每条线与头坐标与当前行聚类点之间的cost矩阵（线头来源于上一行的end_point）
            # cost前向infer做，模型里面不做
            assigned = [False for _ in clusters]
            C = np.Inf*np.ones((len(lane_end_pts), len(clusters)), dtype=np.float64)
            #计算每一个线头坐标点与当前行聚类点之间的dist_error
            for r, pts in enumerate(lane_end_pts): # for each end point in an active lane
                for c, cluster in enumerate(clusters):
                    if len(cluster) == 0:
                        continue
                    # 计算每一个聚类簇的中心点
                    cluster_mean = np.array([[np.mean(cluster), row]], dtype=np.float32)
                    # 获取每一个线头的坐标在vaf map上的单位向量
                    vafs = np.array([VAF[int(round(x[1])), int(round(x[0])), :] for x in pts], dtype=np.float32)
                    vafs = vafs / np.linalg.norm(vafs, axis=1, keepdims=True) 
                    # 用计算出来的线头坐标结合vaf计算的单位向量来推算下一行的聚类中心
                    pred_points = pts + vafs*np.linalg.norm(pts - cluster_mean, axis=1, keepdims=True)
                    # 计算真正的聚类中心与vaf预测的聚类中心之间的error 
                    error = np.mean(np.linalg.norm(pred_points - cluster_mean, axis=1))
                    # 赋值给线头与当前行的error给cost矩阵
                    C[r, c] = error
            # 获取线头点与当前行聚类点在C.shape下的坐标
            row_ind, col_ind = np.unravel_index(np.argsort(C, axis=None), C.shape)
            for r, c in zip(row_ind, col_ind):
                if C[r, c] >= err_thresh:
                    break
                if assigned[c]:
                    continue
                assigned[c] = True
                # 给当前像素点更新最匹配的lane_id
                output[row, clusters[c]] = r+1
                # 根据当前行匹配好的像素点更新线头坐标列表
                lane_end_pts[r] = np.stack((np.array(clusters[c], dtype=np.float32), row*np.ones_like(clusters[c])), axis=1)
            # 没被分配的线分配新的lane_id
            for c, cluster in enumerate(clusters):
                if len(cluster) == 0:
                    continue
                if not assigned[c]:
                    output[row, cluster] = next_lane_id
                    lane_end_pts.append(np.stack((np.array(cluster, dtype=np.float32), row*np.ones_like(cluster)), axis=1))
                    next_lane_id += 1
        
        # 可视化最终解码的车道线
        if viz:
            im_color = cv2.applyColorMap(40*output, cv2.COLORMAP_JET)
            cv2.imshow('Output', im_color)
            ret = cv2.waitKey(0)

        return output

    # 定义get_lane函数，从预测的字典中获取车道信息
    def get_lane(self, preds_dicts):
        
        # 从预测字典中提取二值化分割、水平注意力场、垂直注意力场和topdown（可能是俯视图或其他相关数据）
        binary_seg, haf_pred, vaf_pred, topdown = preds_dicts

        # 转换为数组
        # 对二值化分割应用sigmoid激活函数，重复三次以匹配三个通道，然后将其从tensor转换为numpy数组
        mask_out = self.tensor2image(torch.sigmoid(binary_seg).repeat(1, 3, 1, 1).detach(), 
            np.array([0.0 for _ in range(3)], dtype='float32'), np.array([1.0 for _ in range(3)], dtype='float32'))
        
        # 转换vaf_pred（垂直注意力场）从PyTorch tensor到numpy数组，并进行适当的维度重排
        vaf_out = np.transpose(vaf_pred[0, :, :, :].detach().cpu().float().numpy(), (1, 2, 0))
        # 转换haf_pred（水平注意力场）从PyTorch tensor到numpy数组，并进行适当的维度重排
        haf_out = np.transpose(haf_pred[0, :, :, :].detach().cpu().float().numpy(), (1, 2, 0))
        # 使用先前定义的decodeAFs方法解码注意力场，以获取车道实例
        seg_out = self.decodeAFs(mask_out[:, :, 0], vaf_out, haf_out, fg_thresh=128, err_thresh=5)

        return seg_out  # 返回解码后的车道实例

    # 定义create_viz函数，用于将车道线的预测可视化到输入图像上
    def create_viz(img, seg, mask, vaf, haf):
        # 设置缩放因子
        scale = 8

        # 对输入图像进行2倍放大
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        # 将图像转换为连续数组并设置数据类型为uint8
        img = np.ascontiguousarray(img, dtype=np.uint8)
        # 对车道分割应用颜色映射以生成彩色版本
        seg_color = cv2.applyColorMap(40*seg, cv2.COLORMAP_JET)
        # 获取非零的行和列的索引，即车道的位置
        rows, cols = np.nonzero(seg)

        # 对于每一个非零的位置，即车道的位置
        for r, c in zip(rows, cols):
            # 在图像上绘制表示车道线方向的箭头
            img = cv2.arrowedLine(
                img,  # 输入图像
                (c*scale, r*scale),  # 箭头的起始位置
                (int(c*scale+vaf[r, c, 0]*scale*0.75), int(r*scale+vaf[r, c, 1]*scale*0.5)),  # 箭头的结束位置
                seg_color[r, c, :].tolist(),  # 箭头的颜色
                1,  # 箭头的厚度
                tipLength=0.4  # 箭头尖的长度相对于箭头长度的比率
            )

        # 返回可视化后的图像
        return img
