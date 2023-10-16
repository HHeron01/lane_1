      
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
from scipy.interpolate import InterpolatedUnivariateSpline
from nms import nms
import math
INFINITY = 987654.

class Lane:
    def __init__(self, points=None, invalid_value=-2., metadata=None):
        super(Lane, self).__init__()
        self.curr_iter = 0
        self.points = points
        self.invalid_value = invalid_value
        self.function = InterpolatedUnivariateSpline(points[:, 1], points[:, 0], k=min(3, len(points) - 1))
        self.min_y = points[:, 1].min() - 0.01
        self.max_y = points[:, 1].max() + 0.01

        self.metadata = metadata or {}

    def __repr__(self):
        return '[Lane]\n' + str(self.points) + '\n[/Lane]'

    def __call__(self, lane_ys):
        lane_xs = self.function(lane_ys)

        lane_xs[(lane_ys < self.min_y) | (lane_ys > self.max_y)] = self.invalid_value
        return lane_xs

    def __iter__(self):
        return self

    def __next__(self):
        if self.curr_iter < len(self.points):
            self.curr_iter += 1
            return self.points[self.curr_iter - 1]
        self.curr_iter = 0
        raise StopIteration


@HEADS.register_module()
class LaneATT2DHead(nn.Module):
    def __init__(self,
                 S=72,
                 img_w=640,
                 img_h=360,
                 in_channel=256,
                 anchors_freq_path=None,
                 topk_anchors=None,
                 anchor_feat_channels=64,
                 cls_loss=dict(type='LaneATTFocalLoss'),
                 reg_loss=dict(type='RegL1Loss'),
                 start_y_loss=dict(type='RegL1Loss')):
        super(LaneATT2DHead, self).__init__()
        # Some definitions
        self.stride = 16
        self.img_w = img_w
        self.img_h = img_h
        self.n_strips = S - 1
        self.n_offsets = S
        self.fmap_h = img_h // self.stride    # 32 downsample
        fmap_w = img_w // self.stride
        self.fmap_w = fmap_w
        self.anchor_ys = torch.linspace(1, 0, steps=self.n_offsets, dtype=torch.float32)
        self.anchor_cut_ys = torch.linspace(1, 0, steps=self.fmap_h, dtype=torch.float32)
        self.anchor_feat_channels = anchor_feat_channels
        

        self.cls_loss = build_loss(cls_loss)
        self.reg_loss = build_loss(reg_loss)
        self.start_y_loss = build_loss(start_y_loss)

        # Anchor angles, same ones used in Line-CNN
        self.left_angles = [72., 60., 49., 39., 30., 22.]
        self.right_angles = [108., 120., 131., 141., 150., 158.]
        self.bottom_angles = [165., 150., 141., 131., 120., 108., 100., 90., 80., 72., 60., 49., 39., 30., 15.]

        # Generate anchors
        self.anchors, self.anchors_cut = self.generate_anchors(lateral_n=72, bottom_n=128)

        # Filter masks if `anchors_freq_path` is provided
        if anchors_freq_path is not None:
            anchors_mask = torch.load(anchors_freq_path).cpu()
            assert topk_anchors is not None
            ind = torch.argsort(anchors_mask, descending=True)[:topk_anchors]
            self.anchors = self.anchors[ind]
            self.anchors_cut = self.anchors_cut[ind]

        # Pre compute indices for the anchor pooling
        self.cut_zs, self.cut_ys, self.cut_xs, self.invalid_mask = self.compute_anchor_cut_indices(
            self.anchor_feat_channels, fmap_w, self.fmap_h)

        # Setup and initialize layers
        self.conv1 = nn.Conv2d(in_channel, self.anchor_feat_channels, kernel_size=1)
        self.cls_layer = nn.Linear(2 * self.anchor_feat_channels * self.fmap_h, 2)
        self.start_offset_layer = nn.Linear(2 * self.anchor_feat_channels * self.fmap_h, 1)
        self.reg_layer = nn.Linear(2 * self.anchor_feat_channels * self.fmap_h, self.n_offsets + 1)
        self.attention_layer = nn.Linear(self.anchor_feat_channels * self.fmap_h, len(self.anchors) - 1)
        self.initialize_layer(self.attention_layer)
        self.initialize_layer(self.conv1)
        self.initialize_layer(self.cls_layer)
        self.initialize_layer(self.reg_layer)
        self.initialize_layer(self.start_offset_layer)

    def forward(self, x):
        self.cuda(x.device)
        batch_features = self.conv1(x) # 1, 64, 12, 20
        batch_anchor_features = self.cut_anchor_features(batch_features)
        # b, num_anchors, 64, h, 1    # 1, 2784, 64, 11, 1

        # Join proposals from all images into a single proposals features batch
        # (b*num_anchors, 64*h) ([2784, 704])
        batch_anchor_features = batch_anchor_features.view(-1, self.anchor_feat_channels * self.fmap_h)

        # Add attention features
        softmax = nn.Softmax(dim=1)
        scores = self.attention_layer(batch_anchor_features) # (b*n, n-1) #([2784, 2783])
        attention = softmax(scores).reshape(x.shape[0], len(self.anchors), -1) # (b, n, n-1) ([1, 2784, 2783])
        attention_matrix = torch.eye(attention.shape[1], device=x.device).repeat(x.shape[0], 1, 1)# ([1, 2784, 2784])
        # b, n, n 对角是1
        # 索引出不等于0的位置, (n*(n-1), 2)
        non_diag_inds = torch.nonzero(attention_matrix == 0., as_tuple=False)
        attention_matrix[:] = 0
        attention_matrix[non_diag_inds[:, 0], non_diag_inds[:, 1], non_diag_inds[:, 2]] = attention.flatten()
        batch_anchor_features = batch_anchor_features.reshape(x.shape[0], len(self.anchors), -1)
        attention_features = torch.bmm(torch.transpose(batch_anchor_features, 1, 2), # [1, 2784, 704]
                                       torch.transpose(attention_matrix, 1, 2)).transpose(1, 2)
        attention_features = attention_features.reshape(-1, self.anchor_feat_channels * self.fmap_h) # ([2784, 704])
        batch_anchor_features = batch_anchor_features.reshape(-1, self.anchor_feat_channels * self.fmap_h)
        batch_anchor_features = torch.cat((attention_features, batch_anchor_features), dim=1) # [2784, 1408]

        # Predict
        cls_logits = self.cls_layer(batch_anchor_features) # [2784, 2]
        start_offset = self.start_offset_layer(batch_anchor_features) # [2784, 1]
        start_offset = start_offset.sigmoid() # start 偏移范围 -0.5～0.5  0~0.5
        reg = self.reg_layer(batch_anchor_features)        # [2784, 73]
        

        # Undo joining
        cls_logits = cls_logits.reshape(x.shape[0], -1, cls_logits.shape[1]) # ([1, 2784, 2])
        start_offset = start_offset.reshape(x.shape[0], -1, start_offset.shape[1]) # ([1, 2784, 1])
        reg = reg.reshape(x.shape[0], -1, reg.shape[1]) # ([1, 2784, 73])
        

        # Add offsets to anchors
        reg_proposals = torch.zeros((*cls_logits.shape[:2], 5 + self.n_offsets), device=x.device) # [1, 2784, 77]
        reg_proposals += self.anchors
        reg_proposals[:, :, :2] = cls_logits
        reg_proposals[:,:, 2:3] = (reg_proposals[:,:, 2:3]+start_offset)*0.5 # (0~2)->(0~1)
        reg_proposals[:, :, 4:] += reg

        return (reg_proposals, attention_matrix, batch_features)
        
    def postprocess(self, net_out, nms_thres, nms_topk, conf_threshold):
        reg_proposals, attention_matrix, batch_features = net_out
        proposals_list = self.nms(reg_proposals, attention_matrix, nms_thres, nms_topk, conf_threshold)
        return proposals_list
    
    def get_decodes(self, net_out, nms_thres, nms_topk, conf_threshold):
        proposals_list = self.postprocess(net_out, nms_thres, nms_topk, conf_threshold)
        decodes = self.decode(proposals_list, as_lanes=True)
        return decodes
    
    def draw_anchor_match(self, targets, proposels):
        image = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        cv2.putText(image, "anchor_match", (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        for line_id, gt_lane in enumerate(targets):
            if len(gt_lane) < 2:
                continue
            gt_lane = np.array(gt_lane)
            x_2d, y_2d = (gt_lane[:, 0]*self.img_w).astype(np.int32), (gt_lane[:, 1]*self.img_h).astype(np.int32)
            for k in range(1, gt_lane.shape[0]):
                image = cv2.line(image, (x_2d[k - 1], y_2d[k - 1]),
                                (x_2d[k], y_2d[k]), (0,0,255), 2)
                
        for line_id, anchor_lane in enumerate(proposels):
            if len(anchor_lane) < 2:
                continue
            anchor_lane = np.array(anchor_lane)
            x_2d, y_2d = (anchor_lane[:, 0]*self.img_w).astype(np.int32), (anchor_lane[:, 1]*self.img_h).astype(np.int32)
            
            for k in range(1, anchor_lane.shape[0]):
                image = cv2.line(image, (x_2d[k - 1], y_2d[k - 1]),
                                (x_2d[k], y_2d[k]), (255,0,0), 2)
        return image


    def get_targets_decodes(self, targets, as_lanes=True):
        softmax = nn.Softmax(dim=1)
        decoded = []
        for proposals in targets: # [13, 77]
            proposals[:, :2] = softmax(proposals[:, :2]) # [13, 77] 分类进行softmax
            proposals[:, 4] = torch.round(proposals[:, 4]) # 长度预测值取整，此处为直接回归长度值，长度为[0,72]离散值，可以考虑换成72分类的形式。
            if proposals.shape[0] == 0:
                decoded.append([])
                continue
            if as_lanes:
                pred = self.proposals_to_pred(proposals)
            else:
                pred = proposals
            decoded.append(pred)

        return decoded

    def get_lane2d_points(self, decodeds):
        lanes_2d = []
        for i, decoded in enumerate(decodeds[0]):
            points = decoded.points.tolist()
            # line = {
            #     'line_id': 0,
            #     'coordinates': [],
            # }
            # line['line_id'] = i
            line = []
            for point in points:
                    x = round(point[0], 3)
                    y = round(point[1], 3)
                    # line['coordinates'].append([x, y])
                    line.append([x,y])
            lanes_2d.append(line)
        return lanes_2d

    def nms(self, batch_proposals, batch_attention_matrix, nms_thres, nms_topk, conf_threshold):
        softmax = nn.Softmax(dim=1)
        proposals_list = []
        for proposals, attention_matrix in zip(batch_proposals, batch_attention_matrix):
            anchor_inds = torch.arange(batch_proposals.shape[1], device=proposals.device)
            # The gradients do not have to (and can't) be calculated for the NMS procedure
            with torch.no_grad():
                scores = softmax(proposals[:, :2])[:, 1] # 取出每个anchor预测为车道线的置信度
                if conf_threshold is not None:
                    # apply confidence threshold
                    above_threshold = scores > conf_threshold # 先用置信度过滤
                    proposals = proposals[above_threshold]    # 164, 77
                    scores = scores[above_threshold]          # 164,
                    anchor_inds = anchor_inds[above_threshold]# 164,
                if proposals.shape[0] == 0:
                    proposals_list.append((proposals[[]], self.anchors[[]], attention_matrix[[]], None))
                    continue
                keep, num_to_keep, _ = nms(proposals, scores, overlap=nms_thres, top_k=nms_topk) # 164, 
                keep = keep[:num_to_keep] # keep 经过nms之后留下的proposal索引 [1,334,,34,5,0,0,0,0,0,0,0]
            proposals = proposals[keep] # [13, 77]
            anchor_inds = anchor_inds[keep] # [13]
            attention_matrix = attention_matrix[anchor_inds] # [13, 2784]
            proposals_list.append((proposals, self.anchors[keep], attention_matrix, anchor_inds))

        return proposals_list

    def loss(self, preds, targets, cls_loss_weight=10):
        
        cls_loss = torch.zeros(1, device=targets.device)
        reg_loss = torch.zeros(1, device=targets.device)
        start_y_loss = torch.zeros(1, device=targets.device)
        valid_imgs = len(targets)
        total_positives = 0

        batch_proposals, batch_attention_matrix, batch_features = preds
        for batch_idx in range(len(batch_proposals)):
            target = targets[batch_idx]
            proposals = batch_proposals[batch_idx]
            
            target = target[target[:, 1] == 1]
            if len(target) == 0:
                # If there are no targets, all proposals have to be negatives (i.e., 0 confidence)
                cls_target = proposals.new_zeros(len(proposals)).long()
                cls_pred = proposals[:, :2]
                cls_loss += self.cls_loss(cls_pred, cls_target).sum()
                continue
            # Gradients are also not necessary for the positive & negative matching
            with torch.no_grad():
                positives_mask, invalid_offsets_mask, negatives_mask, target_positives_indices = self.match_proposals_with_targets(
                    self, self.anchors, target)

            positives = proposals[positives_mask]
            num_positives = len(positives)
            total_positives += num_positives
            negatives = proposals[negatives_mask]
            num_negatives = len(negatives)

            # Handle edge case of no positives found
            if num_positives == 0:
                cls_target = proposals.new_zeros(len(proposals)).long()
                cls_pred = proposals[:, :2]
                cls_loss += self.cls_loss(cls_pred, cls_target).sum()
                continue

            # Get classification targets
            all_proposals = torch.cat([positives, negatives], 0)
            cls_target = proposals.new_zeros(num_positives + num_negatives).long()
            cls_target[:num_positives] = 1.
            cls_pred = all_proposals[:, :2]

            # Regression targets
            reg_pred = positives[:, 4:]
            start_y_pred = positives[:, 2]

            '''
            target = target[target_positives_indices]
            positives[:,4] = 72
            target_decodes = self.get_targets_decodes([target])
            proposal_decodes = self.get_targets_decodes([positives])
            target_point = self.get_lane2d_points(target_decodes)
            proposal_point = self.get_lane2d_points(proposal_decodes)

            match_image = self.draw_anchor_match(target_point, proposal_point)
            cv2.imwrite('/root/autodl-tmp/model_log/work_dirs/sample_debug/' + str(batch_idx)+ 'math_debug_tpos25.jpg', match_image)
            match_image    
            '''

            
            with torch.no_grad(): 
                target = target[target_positives_indices]   
                start_y_target = target[:, 2]
                # positive_starts = (positives[:, 2] * self.n_strips).round().long() # 计算起始点y   (0~71)
                target_starts = (start_y_target * self.n_strips).round().long()  # 计算target起始点y [0~71]

                # target[:, 4] -= positive_starts - target_starts
                all_indices = torch.arange(num_positives, dtype=torch.long)
                # ends = (positive_starts + target[:, 4] - 1).round().long()
                ends = (target_starts + target[:, 4] - 1).round().long()
                invalid_offsets_mask = torch.zeros((num_positives, 1 + self.n_offsets + 1),
                                                   dtype=torch.int)  # length + S + pad
                # invalid_offsets_mask[all_indices, 1 + positive_starts] = 1
                # invalid_offsets_mask[all_indices, 1 + ends + 1] -= 1

                invalid_offsets_mask[all_indices, 1 + target_starts] = 1
                invalid_offsets_mask[all_indices, 1 + ends + 1] -= 1
                invalid_offsets_mask = invalid_offsets_mask.cumsum(dim=1) == 0
                invalid_offsets_mask = invalid_offsets_mask[:, :-1]
                invalid_offsets_mask[:, 0] = False
                reg_target = target[:, 4:]
                reg_target[invalid_offsets_mask] = reg_pred[invalid_offsets_mask]
                
            
            # Loss calc
            reg_loss += self.reg_loss(reg_pred, reg_target)
            cls_loss += self.cls_loss(cls_pred, cls_target).sum() / num_positives
            start_y_loss += self.start_y_loss(start_y_pred, start_y_target)

        # Batch mean
        cls_loss /= valid_imgs
        reg_loss /= valid_imgs
        start_y_loss /= valid_imgs

        loss = cls_loss_weight * cls_loss + reg_loss + 10*start_y_loss
        return {'loss': loss, 'cls_loss': cls_loss, 'reg_loss': reg_loss, 'start_y_loss': start_y_loss}

    def match_proposals_with_targets(self, model, proposals, targets, t_pos=25., t_neg=30.):
        # repeat proposals and targets to generate all combinations
        num_proposals = proposals.shape[0]
        num_targets = targets.shape[0]
        # pad proposals and target for the valid_offset_mask's trick
        proposals_pad = proposals.new_zeros(proposals.shape[0], proposals.shape[1] + 1)
        proposals_pad[:, :-1] = proposals
        proposals = proposals_pad
        targets_pad = targets.new_zeros(targets.shape[0], targets.shape[1] + 1)
        targets_pad[:, :-1] = targets
        targets = targets_pad

        proposals = torch.repeat_interleave(proposals, num_targets,
                                            dim=0)  # repeat_interleave'ing [a, b] 2 times gives [a, a, b, b]

        targets = torch.cat(num_proposals * [targets])  # applying this 2 times on [c, d] gives [c, d, c, d]

        # get start and the intersection of offsets
        targets_starts = targets[:, 2] * model.n_strips
        proposals_starts = proposals[:, 2] * model.n_strips
        starts = torch.max(targets_starts, proposals_starts).round().long()
        ends = (targets_starts + targets[:, 4] - 1).round().long()
        lengths = ends - starts + 1
        # ends[lengths < 0] = starts[lengths < 0] - 1
        # lengths[lengths < 0] = 0  # a negative number here means no intersection, thus zero lenght
        ends[(lengths/targets[:, 4]) < 0.3] = starts[(lengths/targets[:, 4]) < 0.3] - 1
        lengths[(lengths/targets[:, 4]) < 0.3] = 0  # a negative number here means no intersection, thus zero lenght

        # generate valid offsets mask, which works like this:
        #   start with mask [0, 0, 0, 0, 0]
        #   suppose start = 1
        #   lenght = 2
        valid_offsets_mask = targets.new_zeros(targets.shape)
        all_indices = torch.arange(valid_offsets_mask.shape[0], dtype=torch.long, device=targets.device)
        #   put a one on index `start`, giving [0, 1, 0, 0, 0]
        valid_offsets_mask[all_indices, 5 + starts] = 1.
        valid_offsets_mask[all_indices, 5 + ends + 1] -= 1.
        #   put a -1 on the `end` index, giving [0, 1, 0, -1, 0]
        #   if lenght is zero, the previous line would put a one where it shouldnt be.
        #   this -=1 (instead of =-1) fixes this
        #   the cumsum gives [0, 1, 1, 0, 0], the correct mask for the offsets
        valid_offsets_mask = valid_offsets_mask.cumsum(dim=1) != 0.
        invalid_offsets_mask = ~valid_offsets_mask

        # compute distances
        # this compares [ac, ad, bc, bd], i.e., all combinations
        distances = torch.abs((targets - proposals) * valid_offsets_mask.float()).sum(dim=1) / (lengths.float() + 1e-9
                                                                                                )  # avoid division by zero
        distances[lengths == 0] = INFINITY
        invalid_offsets_mask = invalid_offsets_mask.view(num_proposals, num_targets, invalid_offsets_mask.shape[1])
        distances = distances.view(num_proposals, num_targets)  # d[i,j] = distance from proposal i to target j

        positives = distances.min(dim=1)[0] < t_pos
        negatives = distances.min(dim=1)[0] > t_neg

        if positives.sum() == 0:
            target_positives_indices = torch.tensor([], device=positives.device, dtype=torch.long)
        else:
            target_positives_indices = distances[positives].argmin(dim=1)
        invalid_offsets_mask = invalid_offsets_mask[positives, target_positives_indices]

        return positives, invalid_offsets_mask[:, :-1], negatives, target_positives_indices

    def compute_anchor_cut_indices(self, n_fmaps, fmaps_w, fmaps_h):
        # definitions
        n_proposals = len(self.anchors_cut) # 2784

        # indexing
        unclamped_xs = torch.flip((self.anchors_cut[:, 5:] / self.stride).round().long(), dims=(1,)) # [2784, 11]
        unclamped_xs = unclamped_xs.unsqueeze(2) # [2784, 11, 1]
        unclamped_xs = torch.repeat_interleave(unclamped_xs, n_fmaps, dim=0).reshape(-1, 1) # [1959936, 1]
        cut_xs = torch.clamp(unclamped_xs, 0, fmaps_w - 1)
        unclamped_xs = unclamped_xs.reshape(n_proposals, n_fmaps, fmaps_h, 1) # [2784, 64, 11, 1]
        invalid_mask = (unclamped_xs < 0) | (unclamped_xs > fmaps_w)
        cut_ys = torch.arange(0, fmaps_h)
        cut_ys = cut_ys.repeat(n_fmaps * n_proposals)[:, None].reshape(n_proposals, n_fmaps, fmaps_h) # [2784, 64, 11]
        cut_ys = cut_ys.reshape(-1, 1)
        cut_zs = torch.arange(n_fmaps).repeat_interleave(fmaps_h).repeat(n_proposals)[:, None]

        return cut_zs, cut_ys, cut_xs, invalid_mask

    def cut_anchor_features(self, features): # 1, 64, 12, 20
        # definitions
        batch_size = features.shape[0]  # 1
        n_proposals = len(self.anchors) # 2784
        n_fmaps = features.shape[1]  # 64    
        # 1, 2784, 64, 11, 1
        batch_anchor_features = torch.zeros((batch_size, n_proposals, n_fmaps, self.fmap_h, 1), device=features.device)

        # actual cutting b,c,fh,fw
        for batch_idx, img_features in enumerate(features): # [64, 12, 20]
            rois = img_features[self.cut_zs, self.cut_ys, self.cut_xs].view(n_proposals, n_fmaps, self.fmap_h, 1)
            rois[self.invalid_mask] = 0
            batch_anchor_features[batch_idx] = rois

        return batch_anchor_features

    def generate_anchors(self, lateral_n, bottom_n):
        left_anchors, left_cut = self.generate_side_anchors(self.left_angles, x=0., nb_origins=lateral_n)
        right_anchors, right_cut = self.generate_side_anchors(self.right_angles, x=1., nb_origins=lateral_n)
        bottom_anchors, bottom_cut = self.generate_side_anchors(self.bottom_angles, y=1., nb_origins=bottom_n)

        return torch.cat([left_anchors, bottom_anchors, right_anchors]), torch.cat([left_cut, bottom_cut, right_cut])

    def generate_side_anchors(self, angles, nb_origins, x=None, y=None):
        if x is None and y is not None: # 低边anchor初始点
            starts = [(x, y) for x in np.linspace(1., 0., num=nb_origins)]
        elif x is not None and y is None: # 侧边anchor初始点
            starts = [(x, y) for y in np.linspace(1., 0., num=nb_origins)]
        else:
            raise Exception('Please define exactly one of `x` or `y` (not neither nor both)')

        n_anchors = nb_origins * len(angles)

        # each row, first for x and second for y:
        # 2 scores, 1 start_y, start_x, 1 lenght, S coordinates, score[0] = negative prob, score[1] = positive prob
        anchors = torch.zeros((n_anchors, 2 + 2 + 1 + self.n_offsets))
        anchors_cut = torch.zeros((n_anchors, 2 + 2 + 1 + self.fmap_h))
        for i, start in enumerate(starts):
            for j, angle in enumerate(angles):
                k = i * len(angles) + j
                anchors[k] = self.generate_anchor(start, angle)
                anchors_cut[k] = self.generate_anchor(start, angle, cut=True)

        return anchors, anchors_cut

    def generate_anchor(self, start, angle, cut=False):
        if cut:
            anchor_ys = self.anchor_cut_ys
            anchor = torch.zeros(2 + 2 + 1 + self.fmap_h)
        else:
            anchor_ys = self.anchor_ys
            anchor = torch.zeros(2 + 2 + 1 + self.n_offsets)
        angle = angle * math.pi / 180.  # degrees to radians
        start_x, start_y = start
        anchor[2] = 1 - start_y
        anchor[3] = start_x
        anchor[5:] = (start_x + (1 - anchor_ys - 1 + start_y) / math.tan(angle)) * self.img_w

        return anchor

    def draw_anchors(self, img_w, img_h, k=100):
        base_ys = self.anchor_ys.detach().cpu().numpy()
        img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        i = -1
        for anchor in self.anchors:
            i += 1
            # if k is not None and i != k:
            #     continue
            if i%k!=0:
                continue
            anchor = anchor.detach().cpu().numpy()
            xs = anchor[5:]
            ys = base_ys * img_h
            points = np.vstack((xs, ys)).T.round().astype(int)
            for p_curr, p_next in zip(points[:-1], points[1:]):
                img = cv2.line(img, tuple(p_curr), tuple(p_next), color=(0, 255, 0), thickness=5)

        return img

    @staticmethod
    def initialize_layer(layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            torch.nn.init.normal_(layer.weight, mean=0., std=0.001)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0)

    def proposals_to_pred(self, proposals):
        self.anchor_ys = self.anchor_ys.to(proposals.device)
        self.anchor_ys = self.anchor_ys.double()  # 图像高度方向72个离散坐标，1～0范围
        lanes = []
        for lane in proposals:  # 处理每条车道线
            lane_xs = lane[5:] / self.img_w # x方向偏移量归一化
            start = int(round(lane[2].item() * self.n_strips)) # y_start (0, 1)->(0, 77)
            length = int(round(lane[4].item())) # y方向 长度取整
            end = start + length - 1  # 根据y方向起点和长度计算终点
            end = min(end, len(self.anchor_ys) - 1) # 过滤图像外的点
            # end = label_end
            # if the proposal does not start at the bottom of the image,
            # extend its proposal until the x is outside the image
            # mask = ~((((lane_xs[:start] >= 0.) &
            #            (lane_xs[:start] <= 1.)).cpu().numpy()[::-1].cumprod()[::-1]).astype(bool))
            lane_xs[end + 1:] = -2  # 将终点之后和起点之前的偏移量置为-2
            # lane_xs[:start][mask] = -2
            lane_xs[:start] = -2
            lane_ys = self.anchor_ys[lane_xs >= 0] # 将有效区间的x偏移量对应的y方向采样点拿出来
            lane_xs = lane_xs[lane_xs >= 0] # 取出有效偏移量
            lane_xs = lane_xs.flip(0).double()
            lane_ys = lane_ys.flip(0)
            if len(lane_xs) <= 1:
                continue
            points = torch.stack((lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)), dim=1).squeeze(2)
            lane = Lane(points=points.detach().cpu().numpy(),
                        metadata={
                            'start_x': lane[3],
                            'start_y': lane[2],
                            'conf': lane[1]
                        })
            lanes.append(lane)
        return lanes

    def decode(self, proposals_list, as_lanes=True):
        softmax = nn.Softmax(dim=1)
        decoded = []
        for proposals, _, _, _ in proposals_list: # [13, 77]
            proposals[:, :2] = softmax(proposals[:, :2]) # [13, 77] 分类进行softmax
            proposals[:, 4] = torch.round(proposals[:, 4]) # 长度预测值取整，此处为直接回归长度值，长度为[0,72]离散值，可以考虑换成72分类的形式。
            if proposals.shape[0] == 0:
                decoded.append([])
                continue
            if as_lanes:
                pred = self.proposals_to_pred(proposals)
            else:
                pred = proposals
            decoded.append(pred)

        return decoded

    def cuda(self, device=None):
        cuda_self = super().cuda(device)
        cuda_self.anchors = cuda_self.anchors.cuda(device)
        cuda_self.anchor_ys = cuda_self.anchor_ys.cuda(device)
        cuda_self.cut_zs = cuda_self.cut_zs.cuda(device)
        cuda_self.cut_ys = cuda_self.cut_ys.cuda(device)
        cuda_self.cut_xs = cuda_self.cut_xs.cuda(device)
        cuda_self.invalid_mask = cuda_self.invalid_mask.cuda(device)
        return cuda_self

    def to(self, *args, **kwargs):
        device_self = super().to(*args, **kwargs)
        device_self.anchors = device_self.anchors.to(*args, **kwargs)
        device_self.anchor_ys = device_self.anchor_ys.to(*args, **kwargs)
        device_self.cut_zs = device_self.cut_zs.to(*args, **kwargs)
        device_self.cut_ys = device_self.cut_ys.to(*args, **kwargs)
        device_self.cut_xs = device_self.cut_xs.to(*args, **kwargs)
        device_self.invalid_mask = device_self.invalid_mask.to(*args, **kwargs)
        return device_self
