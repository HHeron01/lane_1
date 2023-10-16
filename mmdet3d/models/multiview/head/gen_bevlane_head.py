# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@Author  :
@Version :   0.1
@License :   (C)Copyright 2019-2035
@Desc    :   None
"""
import torch
import torch.nn as nn
from mmdet3d.models.builder import HEADS, build_loss
import numpy as np
from mmdet3d.models.multiview.head.focal_loss import FocalLoss


def make_one_layer(in_channels, out_channels, kernel_size=3, padding=1, stride=1, batch_norm=False):
    conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
    if batch_norm:
        layers = [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
    else:
        layers = [conv2d, nn.ReLU(inplace=True)]
    return layers


@HEADS.register_module()
class Gen_BevLaneHead(nn.Module):
    def __init__(
        self,
        num_classes=1,
        in_channel=256,
        grid_config=None,
        debug=False,
        batch_norm=False,
        prob_th=0.45,
        cls_loss=dict(type='FocalLoss',
                      alpha=0.25,
                      gamma=2.),
        reg_loss=dict(type='RegL1Loss'),
    ):
        super().__init__()
        self.head_name = 'gen_anchor_head'
        self.x_min = grid_config['x'][0]
        self.x_max = grid_config['x'][1]
        self.y_min = grid_config['y'][0]
        self.y_max = grid_config['y'][1]

        self.width_res = grid_config['x'][2]
        self.depth_res = grid_config['y'][2]

        self.bev_w = self.x_max - self.x_min
        self.bev_h = self.y_max - self.y_min

        # self.anchor_x_steps = np.linspace(self.x_min, self.x_max,
        #                                   num=int((self.x_max - self.x_min) // self.width_res + 1), endpoint=False)
        self.anchor_x_steps = np.linspace(self.x_min, self.x_max,
                                          num=int(self.bev_w / 0.4) + 1, endpoint=False)

        self.num_x_steps = len(self.anchor_x_steps)

        self.anchor_y_steps = np.linspace(self.y_min, self.y_max, num=int(self.bev_h / 2.), endpoint=False)
        # self.anchor_y_steps = np.linspace(self.y_min, self.y_max, num=int((self.y_max - self.y_min) // self.depth_res),
        #                                   endpoint=False)
        self.num_y_steps = len(self.anchor_y_steps)
        self.feat_bev_h = int((self.y_max - self.y_min) // self.depth_res)
        self.feat_bev_w = int((self.x_max - self.x_min) // self.width_res) + 1

        self.num_classes = num_classes
        self.anchor_dim = 3 * self.num_y_steps + self.num_classes

        self.prob_th = prob_th

        layers = []
        layers += make_one_layer(256, 64, kernel_size=3, padding=(1, 1), batch_norm=batch_norm)
        layers += make_one_layer(64, 64, kernel_size=3, padding=(1, 1), batch_norm=batch_norm)
        layers += make_one_layer(64, 64, kernel_size=3, padding=(0, 1), stride=2, batch_norm=batch_norm)

        # layers += make_one_layer(64, 64, kernel_size=5, padding=(0, 2), batch_norm=batch_norm)
        # layers += make_one_layer(64, 64, kernel_size=5, padding=(0, 2), batch_norm=batch_norm)
        # layers += make_one_layer(64, 64, kernel_size=5, padding=(0, 2), batch_norm=batch_norm)
        # layers += make_one_layer(64, 64, kernel_size=5, padding=(0, 2), batch_norm=batch_norm)
        self.features = nn.Sequential(*layers)

        self.linear_layer = nn.Linear(64 * self.feat_bev_h, self.anchor_dim)

        dim_rt_layers = []
        dim_rt_layers += make_one_layer(10176, 128, kernel_size=(5, 1), padding=(2, 0), batch_norm=batch_norm)
        # dim_rt_layers += make_one_layer(10176, 128, kernel_size=3, padding=(1, 1), batch_norm=batch_norm)
        dim_rt_layers += [nn.Conv2d(128, self.num_classes * self.anchor_dim, kernel_size=(5, 1), padding=(2, 0))]
        self.dim_rt = nn.Sequential(*dim_rt_layers)

        self.cls_loss = build_loss(cls_loss)
        self.reg_loss = build_loss(reg_loss)

        self.debug = debug

    def forward_att(self, topdown):
        feature = self.features(topdown)

        sizes = feature.shape
        # x = feature.reshape(sizes[0], sizes[3], sizes[1] * sizes[2])
        x = feature.reshape(-1, sizes[1] * sizes[2])

        x = self.linear_layer(x)

        x = x.reshape(sizes[0], sizes[3], self.num_classes, -1)

        # apply sigmoid to the probability terms to make it in (0, 1)
        # for i in range(self.num_classes):
        #     x[:, :, i * self.anchor_dim + 2 * self.num_y_steps:(i + 1) * self.anchor_dim] = \
        #         torch.sigmoid(x[:, :, i * self.anchor_dim + 2 * self.num_y_steps:(i + 1) * self.anchor_dim])

        return x, feature

    def forward(self, topdown):
        feature = self.features(topdown)
        # x suppose to be N X 64 X 4 X ipm_w/8, reshape to N X 256 X ipm_w/8 X 1
        sizes = feature.shape
        x = feature.reshape(sizes[0], sizes[1]*sizes[2], sizes[3], 1)
        x = self.dim_rt(x)
        x = x.squeeze(-1).transpose(1, 2)
        # apply sigmoid to the probability terms to make it in (0, 1)
        # for i in range(self.num_lane_type):
        #     x[:, :, (i+1)*self.anchor_dim-1] = torch.sigmoid(x[:, :, (i+1)*self.anchor_dim-1])
        return x, feature



    def loss(self, preds_dicts, gt_labels, **kwargs):
        loss_dict = dict()

        cls_loss = torch.tensor(0).float().to(gt_labels.device)
        reg_loss_x = torch.tensor(0).float().to(gt_labels.device)
        reg_loss_z = torch.tensor(0).float().to(gt_labels.device)
        vis_loss = torch.tensor(0).float().to(gt_labels.device)

        pred_3D_lanes, topdown = preds_dicts
        # gt_anchors = gt_labels
        # print(torch.unique(gt_anchors))
        sizes = pred_3D_lanes.shape

        pred_3D_lanes = pred_3D_lanes.reshape(sizes[0], sizes[1], self.num_classes, self.anchor_dim)
        # gt_3D_lanes = gt_anchors.reshape(sizes[0], sizes[1], self.num_classes, self.anchor_dim)

        pred_3D_lanes = pred_3D_lanes
        gt_3D_lanes = gt_labels

        for i in range(sizes[0]):
            positives_label = gt_3D_lanes[i][:, :, -1].squeeze(-1)

            new_valid_vis_idxes = []
            valid_vis_idxes = torch.nonzero(positives_label > 0).squeeze(-1)
            for valid_vis_idx in valid_vis_idxes:
                # for j in range(valid_vis_idx - 2, valid_vis_idx + 3):
                for j in range(valid_vis_idx - 4, valid_vis_idx + 5):
                    if j >= self.num_x_steps or j < 0:
                        continue
                    new_valid_vis_idxes.append(j)

            gt_3D_lane = gt_3D_lanes[i]
            pred_3D_lane = pred_3D_lanes[i]

            # gt_3D_lane = gt_3D_lane[new_valid_vis_idxes, :, :]
            # pred_3D_lane = pred_3D_lane[new_valid_vis_idxes, :, :]

            pred_class = pred_3D_lane[:, :, -1].unsqueeze(-1)
            # pred_class = nn.Sigmoid()(pred_class)
            pred_anchors_x = pred_3D_lane[:, :, :self.num_y_steps]
            pred_anchors_z = pred_3D_lane[:, :, self.num_y_steps:2 * self.num_y_steps]
            pred_visibility = pred_3D_lane[:, :, 2*self.num_y_steps:3*self.num_y_steps]

            gt_class = gt_3D_lane[:, :, -1].unsqueeze(-1)
            gt_anchors_x = gt_3D_lane[:, :, :self.num_y_steps]
            gt_anchors_z = gt_3D_lane[:, :, self.num_y_steps:2 * self.num_y_steps]
            gt_visibility = gt_3D_lane[:, :, 2*self.num_y_steps:3*self.num_y_steps]

            cls_loss += self.cls_loss(pred_class, gt_class)
            reg_loss_x += self.reg_loss(pred_anchors_x, gt_anchors_x, gt_visibility)
            reg_loss_z += self.reg_loss(pred_anchors_z, gt_anchors_z, gt_visibility)
            vis_loss += self.cls_loss(pred_visibility, gt_visibility)

        loss = 10 * cls_loss + reg_loss_x + reg_loss_z + 2 * vis_loss

        loss_dict['cls_loss'] = 10 * cls_loss
        loss_dict['reg_loss_x'] = reg_loss_x
        loss_dict['reg_loss_z'] = reg_loss_z
        loss_dict['vis_loss'] = 2 * vis_loss
        loss_dict['loss'] = loss

        return loss_dict

    def decode(self, lane_anchor):
        sizes = lane_anchor.shape
        lane_anchor = lane_anchor.reshape(sizes[0], self.num_classes, self.anchor_dim)
        sigmoid = nn.Sigmoid()
        for i in range(self.num_classes):
            lane_anchor[:, :, (i+1)*2 * self.num_y_steps:] = torch.sigmoid(lane_anchor[:, :, (i+1)*2 * self.num_y_steps:])
        lane_anchor = lane_anchor.detach().cpu().numpy()
        all_lines = []
        for j in range(lane_anchor.shape[0]):
            if lane_anchor[j, :, -1] > self.prob_th:
                x_offsets = lane_anchor[j, :, :self.num_y_steps].squeeze(0)
                # try:
                x_3d = x_offsets + self.anchor_x_steps[j]
                # except:
                #     x_offsets = x_offsets
                z_3d = lane_anchor[j, :, self.num_y_steps:2 * self.num_y_steps].squeeze(0)
                visibility = lane_anchor[j, :, 2 * self.num_y_steps:3 * self.num_y_steps].squeeze(0)
                # visibility = torch.tensor(visibility)
                # visibility = sigmoid(visibility)
                # visibility.detach().cpu().numpy()
                x_3d = x_3d[np.where(visibility > self.prob_th)]
                z_3d = z_3d[np.where(visibility > self.prob_th)]
                y_3d = self.anchor_y_steps[np.where(visibility > self.prob_th)]

                points = np.stack((x_3d, y_3d, z_3d), axis=1)
                # points = points.data.cpu().numpy()

                all_lines.append(points)

        return all_lines







