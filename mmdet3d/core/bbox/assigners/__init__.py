# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.core.bbox import AssignResult, BaseAssigner, MaxIoUAssigner

__all__ = ['BaseAssigner', 'MaxIoUAssigner', 'AssignResult']

"""
BaseAssigner:
存储预测框与真实框之间的匹配结果。
属性:
    num_gts (int): 当计算此匹配结果时考虑的真实框数量。
    gt_inds (LongTensor): 对于每个预测框，指示分配的真实框的索引（从1开始计数）。0表示未分配，-1表示忽略。
    max_overlaps (FloatTensor): 预测框与分配的真实框之间的IoU（交并比）。
    labels (None | LongTensor): 如果指定，对于每个预测框，指示分配的真实框的类别标签。

示例:
    >>> # 包含4个预测框和9个真实框的匹配结果，只有两个框被分配到真实框。
    >>> num_gts = 9
    >>> max_overlaps = torch.LongTensor([0, .5, .9, 0])
    >>> gt_inds = torch.LongTensor([-1, 1, 2, 0])
    >>> labels = torch.LongTensor([0, 3, 4, 0])
    >>> self = AssignResult(num_gts, gt_inds, max_overlaps, labels)
    >>> print(str(self))  # xdoctest: +IGNORE_WANT
    <AssignResult(num_gts=9, gt_inds.shape=(4,), max_overlaps.shape=(4,), labels.shape=(4,))>
    >>> # 强制添加gt标签（当将gt添加为proposals时）
    >>> new_labels = torch.LongTensor([3, 4, 5])
    >>> self.add_gt_(new_labels)
    >>> print(str(self))  # xdoctest: +IGNORE_WANT
    <AssignResult(num_gts=9, gt_inds.shape=(7,), max_overlaps.shape=(7,), labels.shape=(7,))>
    
BaseAssigner:
基础的分配器，用于将框分配给真实框。

MaxIoUAssigner:
每个proposals将被分配为`-1`或一个半正整数，表示分配的真实框索引。

- `-1`：负样本，未分配真实框
- 半正整数：正样本，分配的真实框索引（从0开始）

参数:
    pos_iou_thr (float): 正样本框的IoU阈值。
    neg_iou_thr (float or tuple): 负样本框的IoU阈值。
    min_pos_iou (float): 被视为正样本框的最小IoU值。由于第4步（将最大IoU样本分配给每个真实框），正样本的IoU可以小于pos_iou_thr。设置`min_pos_iou`是为了避免将与真实框的IoU极小的框分配为正样本。在1x调度中，可以提高约0.3的mAP，但不会影响3x调度的性能。更多比较可见于 `PR #7464 <https://github.com/open-mmlab/mmdetection/pull/7464>`_。
    gt_max_assign_all (bool): 是否将所有具有与某个真实框具有相同最高重叠的框分配给该真实框。
    ignore_iof_thr (float): 忽略框的IoF阈值（如果指定了`gt_bboxes_ignore`）。负值表示不忽略任何框。
    ignore_wrt_candidates (bool): 是否计算`bboxes`和`gt_bboxes_ignore`之间的IoF，或者反过来。
    match_low_quality (bool): 是否允许低质量匹配。这通常允许在RPN和单阶段检测器中，但在第二阶段不允许。具体细节请参见第4步。
    gpu_assign_thr (int): GPU分配的真实框数量的上限。当GT的数量超过此阈值时，将在CPU设备上进行分配。负值表示不在CPU上分配。
"""
