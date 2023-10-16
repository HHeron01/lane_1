# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@Author  :
@Version :   0.1
@License :   (C)Copyright 2019-2035
@Desc    :   None
"""

import math

import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.models.multiview.head.focal_loss import FocalLoss

from nms import nms


from .anchor_utils import match_proposals_with_targets
from tools.evaluation.anchor_bevlane.lane import Lane
from mmdet3d.models.builder import HEADS, build_loss

@HEADS.register_module()
class LaneATTHead(nn.Module):
    def __init__(self,
                 S=96,
                 fmap_w=192,
                 fmap_h=160,
                 in_channel=64,
                 topk_anchors=1000,
                 anchors_freq_path=None,
                 num_classes=2,
                 anchor_feat_channels=16,
                 cls_loss=dict(type='FocalLoss',
                               alpha=0.25,
                               gamma=2.),
                 reg_loss=dict(type='RegL1Loss'),
    ):
        super(LaneATTHead, self).__init__()
        # Some definitions
        self.head_name = 'att_anchor_head'
        self.img_w = 38.4
        self.img_h = 96
        self.n_strips = S # - 1
        self.n_offsets = S
        self.num_category = num_classes
        self.fmap_w = fmap_w
        self.fmap_h = fmap_h

        self.anchor_ys = torch.linspace(1, 0, steps=self.n_offsets, dtype=torch.float32)
        self.anchor_cut_ys = torch.linspace(1, 0, steps=self.fmap_h, dtype=torch.float32)

        # self.anchor_ys = torch.linspace(self.fmap_h, 0, steps=self.n_offsets, dtype=torch.float32)
        # self.anchor_cut_ys = torch.linspace(self.fmap_h, 0, steps=self.n_offsets, dtype=torch.float32)
        self.anchor_feat_channels = anchor_feat_channels

        # Anchor angles, same ones used in Line-CNN
        self.left_angles = [72., 60., 49., 39., 30., 22.]
        self.right_angles = [108., 120., 131., 141., 150., 158.]
        # self.bottom_angles = [165., 150., 141., 131., 120., 108., 100., 95., 93, 91, 89., 87., 85., 80., 72., 60., 49., 39., 30., 15.]
        self.bottom_angles = [120., 115., 110., 105., 100., 95., 92., 90., 88., 85., 80., 75., 70., 65., 60.]

        # Generate anchors
        self.anchors, self.anchors_cut = self.generate_anchors(bottom_n=self.fmap_w // 2, only_bottom=True)

        # Filter masks if `anchors_freq_path` is provided
        if anchors_freq_path is not None:
            anchors_mask = torch.load(anchors_freq_path).cpu()
            assert topk_anchors is not None
            ind = torch.argsort(anchors_mask, descending=True)[:topk_anchors]
            self.anchors = self.anchors[ind]
            self.anchors_cut = self.anchors_cut[ind]

        self.cut_zs, self.cut_ys, self.cut_xs, self.invalid_mask = self.compute_anchor_cut_indices(
            self.anchor_feat_channels, self.fmap_w, self.fmap_h)

        self.conv1 = nn.Conv2d(in_channel, self.anchor_feat_channels, kernel_size=1)
        self.cls_layer = nn.Linear(2 * self.anchor_feat_channels * self.fmap_h, self.num_category)
        self.reg_layer = nn.Linear(2 * self.anchor_feat_channels * self.fmap_h, 3 * self.n_offsets)
        self.attention_layer = nn.Linear(self.anchor_feat_channels * self.fmap_h, len(self.anchors) - 1)
        self.initialize_layer(self.attention_layer)
        self.initialize_layer(self.conv1)
        self.initialize_layer(self.cls_layer)
        self.initialize_layer(self.reg_layer)

        self.cls_loss = build_loss(cls_loss)
        self.reg_loss = build_loss(reg_loss)
        # self.decode = self.decode()

    def forward(self, batch_features, conf_threshold=0.5, nms_thres=15., nms_topk=50, vis_threshold=0.4):
        batch_features = self.conv1(batch_features)
        batch_anchor_features = self.cut_anchor_features(batch_features)

        batch_anchor_features = batch_anchor_features.view(-1, self.anchor_feat_channels * self.fmap_h)

        # Move relevant tensors to device
        self.anchors = self.anchors.to(device=batch_features.device)

        # Add attention features
        softmax = nn.Softmax(dim=1)
        scores = self.attention_layer(batch_anchor_features)
        attention = softmax(scores).reshape(batch_features.shape[0], len(self.anchors), -1)
        attention_matrix = torch.eye(attention.shape[1], device=batch_features.device).repeat(batch_features.shape[0],
                                                                                              1, 1)
        non_diag_inds = torch.nonzero(attention_matrix == 0., as_tuple=False)
        attention_matrix[:] = 0
        attention_matrix[non_diag_inds[:, 0], non_diag_inds[:, 1], non_diag_inds[:, 2]] = attention.flatten()
        batch_anchor_features = batch_anchor_features.reshape(batch_features.shape[0], len(self.anchors), -1)
        attention_features = torch.bmm(torch.transpose(batch_anchor_features, 1, 2),
                                       torch.transpose(attention_matrix, 1, 2)).transpose(1, 2)
        attention_features = attention_features.reshape(-1, self.anchor_feat_channels * self.fmap_h)
        batch_anchor_features = batch_anchor_features.reshape(-1, self.anchor_feat_channels * self.fmap_h)
        batch_anchor_features = torch.cat((attention_features, batch_anchor_features), dim=1)

        # Predict
        cls_logits = self.cls_layer(batch_anchor_features)
        reg = self.reg_layer(batch_anchor_features)

        # Undo joining
        cls_logits = cls_logits.reshape(batch_features.shape[0], -1, cls_logits.shape[1])
        reg = reg.reshape(batch_features.shape[0], -1, reg.shape[1])
        sigmoid = nn.Sigmoid()
        reg[:, :, 2 * self.n_offsets:3 * self.n_offsets] = sigmoid(reg[:, :, 2 * self.n_offsets:3 * self.n_offsets])

        # Add offsets to anchors
        reg_proposals = torch.zeros((*cls_logits.shape[:2], self.num_category + 3 + 3 * self.n_offsets),
                                    device=batch_features.device)
        reg_proposals += self.anchors
        reg_proposals[:, :, :self.num_category] = cls_logits
        reg_proposals[:, :, self.num_category + 3:] += reg

        # Apply nms
        # proposals_list = self.nms(reg_proposals, attention_matrix, nms_thres, nms_topk, conf_threshold, vis_threshold)

        proposals_list = []
        for proposals, att_matrix in zip(reg_proposals, attention_matrix):
            anchor_inds = torch.arange(reg_proposals.shape[1], device=proposals.device)
            proposals_list.append((proposals, self.anchors, att_matrix, anchor_inds))
        lane_head_output = proposals_list, batch_features
        return lane_head_output

    def nms(self, batch_proposals, batch_attention_matrix, nms_thres, nms_topk, conf_threshold, vis_threshold):
        softmax = nn.Softmax(dim=1)
        proposals_list = []
        # for proposals, _, attention_matrix, _ in batch_proposals_list:
        for proposals, attention_matrix in zip(batch_proposals, batch_attention_matrix):
            anchor_inds = torch.arange(proposals.shape[0], device=proposals.device)
            scores = softmax(proposals[:, :self.num_category])
            # only preserve the max prob category for one anchor
            scores_one_category = torch.max(scores[:, 1:], dim=1)[0]
            # scores_one_category = softmax(proposals[:, :2])[:, 1]
            with torch.no_grad():
                if conf_threshold is not None:
                    # apply confidence threshold
                    above_threshold = scores_one_category > conf_threshold
                    proposals = proposals[above_threshold]
                    scores_one_category = scores_one_category[above_threshold]
                    anchor_inds = anchor_inds[above_threshold]
                if proposals.shape[0] == 0:
                    proposals_list.append((proposals[[]], self.anchors[[]], attention_matrix[[]], None))
                    continue

                proposal_vis = proposals[:, self.num_category + 3 + 2 * self.n_offsets:]
                x_s, y_s = torch.nonzero(proposal_vis >= vis_threshold, as_tuple=True)
                new_last_vis_idx = np.zeros(proposals.shape[0], dtype=np.int)
                new_last_vis_idx[x_s.cpu().numpy()] = y_s.cpu().numpy().astype(np.int)
                ends = torch.from_numpy(new_last_vis_idx).to(proposals.device)

                # change anchor dim back to fit in the nms pkg
                proposals_nms = proposals.new_zeros(proposals.shape[0], 2 + 3 + self.n_offsets)
                proposals_nms[:, 2:2 + 2] = proposals[:, self.num_category:self.num_category + 2]

                # TODO: no inplace modification
                _a = ends - proposals_nms[:, 2] + 1
                proposals_nms[:, 2 + 2] = _a
                proposals_nms[:, 2 + 3:2 + 3 + self.n_offsets] = \
                    proposals[:, self.num_category + 2:self.num_category + 2 + self.n_offsets]
                keep, num_to_keep, _ = nms(proposals_nms, scores_one_category, overlap=nms_thres, top_k=nms_topk)
                keep = keep[:num_to_keep]

            # TODO: no inplace modification
            _b = proposals[keep]
            proposals = _b
            anchor_inds = anchor_inds[keep]
            attention_matrix = attention_matrix[keep]
            proposals_list.append((proposals, self.anchors[keep], attention_matrix, anchor_inds))

        return proposals_list


    def loss(self, preds, targets, cls_loss_weight=10, reg_vis_loss_weight=10):
        proposals_list, topdowns = preds
        loss_dict = {}
        focal_loss = FocalLoss(alpha=0.25, gamma=2.)
        binary_cross_entropy_loss = nn.BCELoss()
        # sigmoid = nn.Sigmoid()
        cls_loss = torch.tensor(0).float().to(targets.device)
        reg_loss_x = torch.tensor(0).float().to(targets.device)
        reg_loss_z = torch.tensor(0).float().to(targets.device)
        reg_vis_loss = torch.tensor(0).float().to(targets.device)
        valid_imgs = len(targets)
        total_positives = 0
        batch_anchors_positives = []

        for (proposals, anchors, _, _), target in zip(proposals_list, targets):
            # Filter lanes that do not exist (confidence == 0)
            target = target[target[:, 0] == 0]
            # in case no proposals when large nms suppression for test
            if len(proposals) == 0:
                continue
            if len(target) == 0:
                # If there are no targets, all proposals have to be negatives (i.e., 0 confidence)
                cls_target = proposals.new_zeros(len(proposals)).long()
                cls_pred = proposals[:, :self.num_category]
                # cls_loss += self.cls_loss(cls_pred, cls_target)
                cls_loss += focal_loss(cls_pred, cls_target).sum()
                batch_anchors_positives.append([])
                continue
            # Gradients are also not necessary for the positive & negative matching
            with torch.no_grad():
                anchors = anchors.to(device=target.device)
                positives_mask, invalid_offsets_mask, negatives_mask, target_positives_indices = \
                    match_proposals_with_targets(self, anchors, target)
                anchors_positives = anchors[positives_mask]
                batch_anchors_positives.append(anchors_positives)

            positives = proposals[positives_mask]
            num_positives = len(positives)
            total_positives += num_positives
            negatives = proposals[negatives_mask]
            num_negatives = len(negatives)
            # print("num_positives, num_negatives: ", num_positives, num_negatives)

            # Handle edge case of no positives found
            if num_positives == 0:
                cls_target = proposals.new_zeros(len(proposals)).long()
                # cls_pred = proposals[:, :2]
                cls_pred = proposals[:, :self.num_category]
                # cls_loss += self.cls_loss(cls_pred, cls_target)
                cls_loss += focal_loss(cls_pred, cls_target).sum()
                continue

            # Get classification targets
            print("positives:", len(positives), len(negatives))
            all_proposals = torch.cat([positives, negatives], 0)
            cls_pred = all_proposals[:, :self.num_category]
            with torch.no_grad():
                target_positives = target[target_positives_indices]
                cls_target = proposals.new_zeros(num_positives + num_negatives, self.num_category).long()
                cls_target[:num_positives, :] = target_positives[:, :self.num_category]
                cls_target[num_positives:, 0] = 1

            log_prob = F.log_softmax(cls_pred, dim=-1)
            cls_loss = -torch.sum(log_prob * cls_target) / (num_positives + num_negatives)

            # cls_loss += self.cls_loss(cls_pred, cls_target)
            # Regression targets
            reg_pred_x = positives[:, self.num_category + 3: self.num_category + 3 + self.n_offsets]
            reg_pred_z = positives[:, self.num_category + 3 + self.n_offsets: self.num_category + 3 + 2 * self.n_offsets]
            reg_vis_pred = positives[:, self.num_category + 3 + 2 * self.n_offsets:]
            # reg_vis_pred = sigmoid(reg_vis_pred)
            with torch.no_grad():
                target = target[target_positives_indices]

                reg_target_x = target[:, self.num_category + 3 : self.num_category + 3 + self.n_offsets]
                reg_target_z = target[:, self.num_category + 3 + self.n_offsets : self.num_category + 3 + 2 * self.n_offsets]
                reg_vis_target = target[:, self.num_category + 3 + 2 * self.n_offsets:]
                reg_target_length = torch.sum(reg_vis_target, dim=-1) + 1e-9

            # Loss calc
            reg_loss_x += self.reg_loss(reg_pred_x, reg_target_x, reg_vis_target)
            reg_loss_z += self.reg_loss(reg_pred_z, reg_target_z, reg_vis_target)
            # reg_vis_loss += self.cls_loss(reg_vis_pred, reg_vis_target)
            reg_vis_loss += binary_cross_entropy_loss(reg_vis_pred, reg_vis_target)

        # Batch mean
        cls_loss /= valid_imgs
        reg_loss_x /= valid_imgs
        reg_loss_z /= valid_imgs
        reg_vis_loss /= valid_imgs

        loss = cls_loss_weight * cls_loss + reg_loss_x + reg_loss_z + reg_vis_loss_weight * reg_vis_loss
        loss.requires_grad_(True)

        loss_dict['cls_loss'] = cls_loss
        loss_dict['reg_loss_x'] = reg_loss_x
        loss_dict['reg_loss_z'] = reg_loss_z
        loss_dict['vis_loss'] = reg_vis_loss
        loss_dict['loss'] = loss

        return loss_dict

    def compute_anchor_cut_indices(self, n_fmaps, fmaps_w, fmaps_h):
        # definitions
        n_proposals = len(self.anchors_cut)

        # indexing
        unclamped_xs = torch.flip((self.anchors_cut[:, self.num_category + 3:self.num_category + 3 + fmaps_h]).round().long(), dims=(1,))
        unclamped_xs = unclamped_xs.unsqueeze(2)
        unclamped_xs = torch.repeat_interleave(unclamped_xs, n_fmaps, dim=0).reshape(-1, 1)
        cut_xs = torch.clamp(unclamped_xs, 0, fmaps_w - 1)
        unclamped_xs = unclamped_xs.reshape(n_proposals, n_fmaps, fmaps_h, 1)
        invalid_mask = (unclamped_xs < 0) | (unclamped_xs > fmaps_w)
        cut_ys = torch.arange(0, fmaps_h)
        cut_ys = cut_ys.repeat(n_fmaps * n_proposals)[:, None].reshape(n_proposals, n_fmaps, fmaps_h)
        cut_ys = cut_ys.reshape(-1, 1)
        cut_zs = torch.arange(n_fmaps).repeat_interleave(fmaps_h).repeat(n_proposals)[:, None]

        return cut_zs, cut_ys, cut_xs, invalid_mask


    def cut_anchor_features(self, features):
        # definitions
        batch_size = features.shape[0]
        n_proposals = len(self.anchors)
        n_fmaps = features.shape[1]
        batch_anchor_features = torch.zeros((batch_size, n_proposals, n_fmaps, self.fmap_h, 1), device=features.device)

        # actual cutting
        for batch_idx, img_features in enumerate(features):
            rois = img_features[self.cut_zs, self.cut_ys, self.cut_xs].view(n_proposals, n_fmaps, self.fmap_h, 1)
            rois[self.invalid_mask] = 0
            batch_anchor_features[batch_idx] = rois

        return batch_anchor_features

    def generate_anchors(self, lateral_n=10, bottom_n=72, only_bottom=True):
        if only_bottom:
            bottom_anchors, bottom_cut = self.generate_side_anchors(self.bottom_angles, y=1., nb_origins=bottom_n)
            return bottom_anchors, bottom_cut
        else:
            left_anchors, left_cut = self.generate_side_anchors(self.left_angles, x=0., nb_origins=lateral_n)
            right_anchors, right_cut = self.generate_side_anchors(self.right_angles, x=1., nb_origins=lateral_n)
            bottom_anchors, bottom_cut = self.generate_side_anchors(self.bottom_angles, y=1., nb_origins=bottom_n)
            return torch.cat([left_anchors, bottom_anchors, right_anchors]), torch.cat([left_cut, bottom_cut, right_cut])

    def generate_side_anchors(self, angles, nb_origins, x=None, y=None):
        if x is None and y is not None:
            starts = [(x, y) for x in np.linspace(1., 0., num=nb_origins)]
        elif x is not None and y is None:
            starts = [(x, y) for y in np.linspace(1., 0., num=nb_origins)]
        else:
            raise Exception('Please define exactly one of `x` or `y` (not neither nor both)')

        n_anchors = nb_origins * len(angles)

        # each row, first for x and second for y:
        # 2 scores, 1 start_y, start_x, 1 lenght, S coordinates[x, y, z], score[0] = negative prob, score[1] = positive prob
        anchors = torch.zeros((n_anchors, self.num_category + 3 + 3 * self.n_offsets))
        anchors_cut = torch.zeros((n_anchors, self.num_category + 3 + 3 * self.fmap_h))
        for i, start in enumerate(starts):
            for j, angle in enumerate(angles):
                k = i * len(angles) + j
                anchors[k] = self.generate_anchor(start, angle)
                anchors_cut[k] = self.generate_anchor(start, angle, cut=True)

        return anchors, anchors_cut

    def generate_anchor(self, start, angle, cut=False):
        angle = angle * math.pi / 180.  # degrees to radians
        start_x, start_y = start

        if cut:
            anchor_ys = self.anchor_cut_ys
            anchor = torch.zeros(self.num_category + 3 + 3 * self.fmap_h)

            anchor[self.num_category] = 1 - start_y
            anchor[self.num_category + 1] = start_x
            anchor[self.num_category + 2] = 0. #start_z
            # anchor[4] = start_z
            # anchor[5] = lane_lenght
            anchor[6:6 + self.fmap_h] = (start_x + (1 - anchor_ys - 1 + start_y) / math.tan(angle)) * self.fmap_w
        else:
            anchor_ys = self.anchor_ys
            anchor = torch.zeros(self.num_category + 3 + 3 * self.n_offsets)

            anchor[self.num_category] = 1 - start_y
            anchor[self.num_category + 1] = start_x
            anchor[self.num_category + 2] = 0. #start_z

            anchor[self.num_category + 3:self.num_category + 3 + self.n_offsets] = (start_x + (1 - anchor_ys - 1 + start_y) / math.tan(angle)) * self.img_w

        return anchor

    def draw_anchors(self, img_w, img_h, k=None):
        base_ys = self.anchor_ys.cpu().numpy()
        img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        i = -1
        for i, anchor in enumerate(self.anchors):
        # for i, anchor in enumerate(self.anchors_cut):
            # if i < len(self.anchors)//2 or i > len(self.anchors)//2 + 10:
            #     continue
            i += 1
            if k is not None and i != k:
                continue
            anchor = anchor.cpu().numpy()
            xs = anchor[6:6 + self.n_offsets]
            ys = base_ys * img_h
            # ys = base_ys * self.fmap_h

            points = np.vstack((xs, ys)).T.round().astype(int)
            for p_curr, p_next in zip(points[:-1], points[1:]):
                img = cv2.line(img, tuple(p_curr), tuple(p_next), color=(0, 255, 0), thickness=1)

        return img

    @staticmethod
    def initialize_layer(layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            torch.nn.init.normal_(layer.weight, mean=0., std=0.001)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0)

    def anchors_to_lanes(self, batch_anchors):
        decoded = []
        for anchors in batch_anchors:
            if isinstance(anchors, list):
                decoded.append([])
                continue
            self.anchor_ys = self.anchor_ys.to(anchors.device)
            self.anchor_ys = self.anchor_ys.double()
            lanes = []
            for anchor in anchors:
                lane_xs = anchor[self.num_category + 2:self.num_category + 2 + self.n_offsets] / self.img_w
                lane_ys = self.anchor_ys[(lane_xs >= 0.) & (lane_xs <= 1.)]
                lane_xs = lane_xs[(lane_xs >= 0.) & (lane_xs <= 1.)]
                lane_xs = lane_xs.flip(0).double()
                lane_ys = lane_ys.flip(0)
                if len(lane_xs) <= 1:
                    continue
                points = torch.stack((lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)), dim=1).squeeze(2)
                points = points.data.cpu().numpy()
                if np.shape(points)[0] < 2:
                    continue
                # diff from proposals_to_lanes, directly ouput points here rather than Lane
                lanes.append(points)
            decoded.append(lanes)
        return decoded

    def proposals_to_pred(self, proposals, vis_thres):
        self.anchor_ys = self.anchor_ys.to(proposals.device)
        self.anchor_ys = self.anchor_ys.double()
        lanes = []
        for id, lane in enumerate(proposals):
            lane_xs = lane[self.num_category + 3:self.num_category + 3 + self.n_offsets]
            lane_zs = lane[self.num_category + 3 + self.n_offsets:self.num_category + 3 + 2 * self.n_offsets]
            start = int(round(lane[self.num_category].item()))
            # length = int(round(lane[4].item()))
            lane_vis = lane[self.num_category + 3 + 2 * self.n_offsets:]
            valid_vis_idxes = torch.nonzero(lane_vis >= vis_thres)
            start_vis_idx = valid_vis_idxes[0, 0].item() if len(valid_vis_idxes) else 0
            end_vis_idx = valid_vis_idxes[-1, 0].item() if len(valid_vis_idxes) else 0
            length = (end_vis_idx - start + 1)

            end = start + length - 1
            end = min(end, len(self.anchor_ys) - 1)
            start = max(start, start_vis_idx)
            # end = label_end
            # if the proposal does not start at the bottom of the image,
            # extend its proposal until the x is outside the image
            mask = ~((((lane_xs[:start] >= 0.) &
                       (lane_xs[:start] <= 1.)).cpu().numpy()[::-1].cumprod()[::-1]).astype(np.bool))
            lane_xs[end + 1:] = -2
            lane_xs[:start][mask] = -2
            lane_ys = self.anchor_ys[lane_xs >= 0]
            lane_zs = lane_zs[lane_xs >= 0]
            lane_xs = lane_xs[lane_xs >= 0]
            lane_xs = lane_xs.flip(0).double()
            lane_ys = lane_ys.flip(0) * self.n_offsets
            lane_zs = lane_zs.flip(0).double()
            if len(lane_xs) <= 1:
                continue
            # np.around(x_values[visibility_indexs][0], decimals=4)
            points = torch.stack((lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1), lane_zs.reshape(-1, 1)), dim=1).squeeze(2)
            points = points.data.cpu().numpy()
            if np.shape(points)[0] < 2:
                continue
            pred_cat = torch.argmax(lane[1:self.num_category])
            lane = Lane(points=points,
                        metadata={
                            'start_x': lane[self.num_category + 1].item(),
                            'start_y': lane[self.num_category].item(),
                            'start_z': lane[self.num_category + 2].item(),
                            'pred_cat': pred_cat.item() + 1,
                            'conf': lane[pred_cat.item()]
                        })
            lanes.append(lane)
        return lanes

    def decode(self, proposals_list, vis_thres=0.5, as_lanes=True):
        softmax = nn.Softmax(dim=1)
        decoded = []
        for proposals, _, _, _ in proposals_list:
            a = softmax(proposals[:, :self.num_category])
            proposals = torch.cat((a, proposals[:, self.num_category:]), dim=1)

            if proposals.shape[0] == 0:
                decoded.append([])
                continue
            if as_lanes:
                pred = self.proposals_to_pred(proposals, vis_thres)
            else:
                pred = proposals
            decoded.append(pred)

        return decoded