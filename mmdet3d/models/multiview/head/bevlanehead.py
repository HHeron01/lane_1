# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@Author  :   linjun.shi
@Version :   0.1
@License :   (C)Copyright 2019-2035
@Desc    :   None
"""
import torch
import torch.nn as nn
from mmdet3d.models.builder import HEADS, build_loss


def CoordFeat2d(feat, mode="normal"):
    """
    get coord feature, (batch,2,h,w)
    params:
        mode: normal, [-1,1]; nature, [0,length-1]
    """
    shape = feat.shape
    # TODO: if use_temporal_module=False
    # e.g. shape:  torch.Size([batch_size, T=1, C=256, H=160, W=64])
    # TODO: if use_temporal_module=True
    # e.g. shape:  torch.Size([batch_size, C=256, H=160, W=64])
    b = shape[0]
    w = shape[-1]
    h = shape[-2]
    if mode == "normal":
        x_low = -1
        x_high = 1
        y_low = -1
        y_high = 1
    elif mode == "nature":
        x_low = 0
        x_high = w - 1
        y_low = 0
        y_high = h - 1
    x_range = torch.linspace(x_low, x_high, w, device=feat.device)
    y_range = torch.linspace(y_low, y_high, h, device=feat.device)
    y, x = torch.meshgrid(y_range, x_range)
    y = y.expand([b, 1, -1, -1])
    x = x.expand([b, 1, -1, -1])
    coord_feat = torch.cat([x, y], 1)
    return coord_feat


@HEADS.register_module()
class BevLaneHead(nn.Module):
    def __init__(
        self,
        num_classes=2,
        in_channel=64,
        embed_dim=4,
        grid_cfg=None,
        use_lanenet=True,
        use_hafvaf=True,
        use_offset=True,
        use_z=True,
        debug=False,
        off_loss=dict(type='OffsetLoss'),
        seg_loss=dict(type='Lane_FocalLoss'),
        haf_loss=dict(type='RegL1Loss'),
        vaf_loss=dict(type='RegL1Loss'),
        z_loss=dict(type='RegL1Loss'),
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.use_lanenet = use_lanenet
        self.use_hafvaf = use_hafvaf
        self.use_offset = use_offset
        self.use_z = use_z
        self.inner_channel = 64

        self.off_loss = build_loss(off_loss)
        self.seg_loss = build_loss(seg_loss)
        self.haf_loss = build_loss(haf_loss)
        self.vaf_loss = build_loss(vaf_loss)
        self.z_loss = build_loss(z_loss)

        if self.use_lanenet:
            self.binary_seg = nn.Sequential(
                nn.Conv2d(in_channel, self.inner_channel, 1),
                nn.BatchNorm2d(self.inner_channel),
                nn.ReLU(),
                nn.Conv2d(self.inner_channel, self.num_classes, 1),
                nn.Sigmoid(),
            )
            self.binary_seg1 = nn.Sequential(
                nn.Conv2d(in_channel, self.inner_channel, 1),
                nn.BatchNorm2d(self.inner_channel),
                nn.ReLU(),
                nn.Conv2d(self.inner_channel, self.num_classes, 1),
                nn.Sigmoid(),
            )


            self.embedding = nn.Sequential(
                nn.Conv2d(in_channel + 2, self.inner_channel, 1),
                nn.BatchNorm2d(self.inner_channel),
                nn.ReLU(),
                nn.Conv2d(self.inner_channel, self.embed_dim, 1),
            )

        if self.use_hafvaf:
            self.haf_head = nn.Sequential(
                nn.Conv2d(in_channel, self.inner_channel, 1),
                nn.BatchNorm2d(self.inner_channel),
                nn.ReLU(),
                nn.Conv2d(self.inner_channel, 1, 1),
            )
            self.vaf_head = nn.Sequential(
                nn.Conv2d(in_channel, self.inner_channel, 1),
                nn.BatchNorm2d(self.inner_channel),
                nn.ReLU(),
                nn.Conv2d(self.inner_channel, 2, 1),
            )

        if self.use_offset:
            self.offset_head = nn.Sequential(
                nn.Conv2d(in_channel, self.inner_channel, 1),
                nn.BatchNorm2d(self.inner_channel),
                nn.ReLU(),
                # nn.Conv2d(self.inner_channel, 2, 1),
                nn.Conv2d(self.inner_channel, 2, 1),
                nn.Sigmoid(),
            )

        if self.use_z:
            self.z_head = nn.Sequential(
                nn.Conv2d(in_channel, self.inner_channel, 1),
                nn.BatchNorm2d(self.inner_channel),
                nn.ReLU(),
                nn.Conv2d(self.inner_channel, 1, 1),
            )

        self.debug = debug
        if self.debug:
            self.debug_loss = nn.CrossEntropyLoss()

    def loss(self, preds_dicts, gt_labels, **kwargs):
        loss_dict = dict()
        binary_seg, binary_seg1,embedding, haf_pred, vaf_pred, off_pred, z_pred, topdown = preds_dicts
        gt_mask, mask_haf, mask_vaf, mask_offset, mask_z,haf_masked = gt_labels
        device = binary_seg.device
        maps = gt_mask.to(device)
        maps1 = haf_masked.to(device)
        haf_loss = self.haf_loss(haf_pred, mask_haf,binary_seg1)  #这里增加标签
        vaf_loss = self.vaf_loss(vaf_pred, mask_vaf, binary_seg1)
        seg_loss = 0.6*self.seg_loss(binary_seg, maps) + 0.4*self.seg_loss(binary_seg, maps1)
        offset_loss = self.off_loss(off_pred, mask_offset, binary_seg)
        z_loss = self.z_loss(z_pred, mask_z, binary_seg)
        loss_dict['haf_loss'] = haf_loss * 12.0
        loss_dict['vaf_loss'] = vaf_loss * 15.0
        loss_dict['seg_loss'] = seg_loss * 10.0
        loss_dict['offset_loss'] = offset_loss * 10.0
        loss_dict['z_loss'] = z_loss * 10.0
        loss_dict['loss'] = (2 * haf_loss + 2 * vaf_loss + 8 * seg_loss + offset_loss + z_loss) * 10.0
        # loss_dict['loss'] = seg_loss
        return loss_dict

    def forward(self, topdown):
        binary_seg, embedding = None, None
        # topdown = topdown.squeeze(1)
        # if self.use_lanenet:
        #     coordfeat = CoordFeat2d(topdown)
        #     # print("coordfeat_em:", coordfeat.shape, topdown.shape)
        #     embedfeat = torch.cat([topdown, coordfeat], 1)
        #     embedding = self.embedding(embedfeat)
        # print("topdown:", topdown.shape)
        binary_seg = self.binary_seg(topdown)
        binary_seg1 = self.binary_seg1(topdown)
        haf, vaf = None, None
        if self.use_hafvaf:
            haf = self.haf_head(topdown)
            vaf = self.vaf_head(topdown)

        offset_feature = None
        if self.use_offset:
            # modify
            offset_feature = self.offset_head(topdown)

        z_feature = None
        if self.z_loss:
            z_feature = self.z_head(topdown)

        lane_head_output = binary_seg, binary_seg1,embedding, haf, vaf, offset_feature, z_feature, topdown

        return lane_head_output













