# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet.core.bbox.builder import BBOX_SAMPLERS
from . import RandomSampler, SamplingResult


@BBOX_SAMPLERS.register_module()
class IoUNegPiecewiseSampler(RandomSampler):
    """
    IoU分段采样。
    根据IoU阈值的列表对负样本进行采样。负样本根据 `neg_iou_piece_thrs` 被分成几个部分。每个部分的比例由 `neg_piece_fractions` 指示。
    参数：
        num (int): 提议的数量。
        pos_fraction (float): 正样本的比例。
        neg_piece_fractions (list): 包含每个部分负样本比例的列表。
        neg_iou_piece_thrs (list): 包含每个部分的IoU阈值的列表，表示该部分的上限。
        neg_pos_ub (float): 限制负样本总比例的上限。
        add_gt_as_proposals (bool): 是否将gt添加为提议。
    """


    def __init__(self,
                 num,
                 pos_fraction=None,
                 neg_piece_fractions=None,
                 neg_iou_piece_thrs=None,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=False,
                 return_iou=False):
        super(IoUNegPiecewiseSampler,
              self).__init__(num, pos_fraction, neg_pos_ub,
                             add_gt_as_proposals)
        assert isinstance(neg_piece_fractions, list)
        assert len(neg_piece_fractions) == len(neg_iou_piece_thrs)
        self.neg_piece_fractions = neg_piece_fractions
        self.neg_iou_thr = neg_iou_piece_thrs
        self.return_iou = return_iou
        self.neg_piece_num = len(self.neg_piece_fractions)

    def _sample_pos(self, assign_result, num_expected, **kwargs):
        """随机采样一些正样本。"""
        pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
        # 获取正样本的索引，这里正样本的索引为assign_result.gt_inds大于0的位置
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            return self.random_choice(pos_inds, num_expected)

    """
    该方法实现了随机采样一些负样本的功能。首先，找到所有负样本的索引neg_inds（即assign_result.gt_inds为0的位置）。然后根据负样本的IoU值将负样本划分成若干部分，每个部分根据预期的负样本数量进行采样。在每个部分，如果实际采样到的负样本数量小于预期数量，则将已采样到的负样本加入到neg_inds_choice中，并在后续部分中扩展同样数量的负样本。在最后一个部分，如果负样本数量仍然不足预期数量，则在当前部分的负样本中随机选择，使得最终选择的负样本总数为预期数量num_expected。最后，将选择的负样本索引neg_inds_choice返回。
    """
    def _sample_neg(self, assign_result, num_expected, **kwargs):
        """随机采样一些负样本。"""
        neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
        # 获取负样本的索引，这里负样本的索引为assign_result.gt_inds为0的位置
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= 0:
            return neg_inds.squeeze(1)
        else:
            neg_inds_choice = neg_inds.new_zeros([0])
            extend_num = 0
            max_overlaps = assign_result.max_overlaps[neg_inds]
            
            for piece_inds in range(self.neg_piece_num):
                if piece_inds == self.neg_piece_num - 1:  # 如果是最后一个部分
                    piece_expected_num = num_expected - len(neg_inds_choice)
                    min_iou_thr = 0
                else:
                    # 如果之前的部分负样本数小于预期数目，在当前部分扩展同样的数量
                    piece_expected_num = int(num_expected *
                                                self.neg_piece_fractions[piece_inds]) + extend_num
                    min_iou_thr = self.neg_iou_thr[piece_inds + 1]
                max_iou_thr = self.neg_iou_thr[piece_inds]
                # 筛选在IoU范围内的负样本索引
                piece_neg_inds = torch.nonzero(
                    (max_overlaps >= min_iou_thr)
                    & (max_overlaps < max_iou_thr),
                    as_tuple=False).view(-1)

                if len(piece_neg_inds) < piece_expected_num:
                    # 当部分负样本数目小于预期数目时，在neg_inds_choice中加入当前部分的负样本
                    neg_inds_choice = torch.cat(
                        [neg_inds_choice, neg_inds[piece_neg_inds]], dim=0)
                    extend_num += piece_expected_num - len(piece_neg_inds)
                    
                    if piece_inds == self.neg_piece_num - 1:
                        extend_neg_num = num_expected - len(neg_inds_choice)
                        if piece_neg_inds.numel() > 0:
                            # 对最后一个部分的负样本进行随机选择，使得选择的负样本总数为num_expected
                            rand_idx = torch.randint(
                                low=0,
                                high=piece_neg_inds.numel(),
                                size=(extend_neg_num,)).long()
                            neg_inds_choice = torch.cat(
                                [neg_inds_choice, piece_neg_inds[rand_idx]], dim=0)
                        else:
                            # 如果最后一个部分负样本数目为0，则在全部先前部分的负样本中随机选择，使得选择的负样本总数为num_expected
                            rand_idx = torch.randint(
                                low=0,
                                high=neg_inds_choice.numel(),
                                size=(extend_neg_num,)).long()
                            neg_inds_choice = torch.cat(
                                [neg_inds_choice, neg_inds_choice[rand_idx]], dim=0)
                else:
                    # 在当前部分的负样本中随机选择预期数量的负样本
                    piece_choice = self.random_choice(piece_neg_inds, piece_expected_num)
                    neg_inds_choice = torch.cat(
                        [neg_inds_choice, neg_inds[piece_choice]], dim=0)
                    extend_num = 0
            assert len(neg_inds_choice) == num_expected
            return neg_inds_choice


    def sample(self,
               assign_result,
               bboxes,
               gt_bboxes,
               gt_labels=None,
               **kwargs):
        """
        对正负框进行采样。
        这是一个简单的实现，根据候选框、分配结果和真实框对框进行采样。
        参数：
            assign_result (:obj:`AssignResult`): 框分配结果。
            bboxes (torch.Tensor): 待采样的框。
            gt_bboxes (torch.Tensor): 真实框。
            gt_labels (torch.Tensor, optional): 真实框的类别标签。
        返回：
            :obj:`SamplingResult`: 采样结果。
        """

        if len(bboxes.shape) < 2:
            bboxes = bboxes[None, :]
        # 将bboxes转换为二维张量，以确保形状为(1, N)

        gt_flags = bboxes.new_zeros((bboxes.shape[0], ), dtype=torch.bool)
        # 创建与bboxes形状相同的布尔类型标记张量gt_flags，用于标记真实框的位置，初始值为0

        if self.add_gt_as_proposals and len(gt_bboxes) > 0:
            if gt_labels is None:
                raise ValueError(
                    'gt_labels must be given when add_gt_as_proposals is True')
            # 如果add_gt_as_proposals为True且存在真实框，则将真实框添加到bboxes中
            # 同时将相应的真实框类别标签添加到assign_result中
            bboxes = torch.cat([gt_bboxes, bboxes], dim=0)
            assign_result.add_gt_(gt_labels)
            # 创建布尔类型张量gt_ones，其长度与真实框数量相同，并全部初始化为1
            gt_ones = bboxes.new_ones(gt_bboxes.shape[0], dtype=torch.bool)
            gt_flags = torch.cat([gt_ones, gt_flags])
            # 将gt_flags中的1和先前的0拼接起来，用于标记真实框的位置

        num_expected_pos = int(self.num * self.pos_fraction)
        # 根据正样本比例计算期望的正样本数量
        pos_inds = self.pos_sampler._sample_pos(
            assign_result, num_expected_pos, bboxes=bboxes, **kwargs)
        # 从assign_result中采样出正样本的索引pos_inds，数量为num_expected_pos

        pos_inds = pos_inds.unique()
        # 去除pos_inds中的重复索引

        num_sampled_pos = pos_inds.numel()
        num_expected_neg = self.num - num_sampled_pos
        # 计算采样后的正样本数量和期望的负样本数量

        if self.neg_pos_ub >= 0:
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
            # 根据负样本和正样本的比例上限限制负样本数量

        neg_inds = self.neg_sampler._sample_neg(
            assign_result, num_expected_neg, bboxes=bboxes, **kwargs)
        # 从assign_result中采样出负样本的索引neg_inds，数量为num_expected_neg

        sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                                         assign_result, gt_flags)
        # 创建SamplingResult对象，其中包含了正样本索引、负样本索引、所有框的集合、真实框的集合、assign_result对象和真实框标志

        if self.return_iou:
            sampling_result.iou = assign_result.max_overlaps[torch.cat(
                [pos_inds, neg_inds])]
            sampling_result.iou.detach_()
            # 如果return_iou为True，则将采样结果中的iou属性赋值为正负样本的IoU值，并从assign_result中获取最大重叠度

        return sampling_result
        # 返回采样结果
