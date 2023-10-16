import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmdet.models.losses.utils import weighted_loss
from mmdet3d.models.builder import LOSSES

# 注册损失函数模块
@LOSSES.register_module()
class OffsetLoss(nn.Module):
    def __init__(self, reduction='mean'):
        """
        计算logits和labels之间的L1损失，同时还提供了遮罩（mask）功能。如果提供了遮罩，则只计算那些在遮罩下的值大于0的位置的损失。此外，如果logits是一个四维张量而mask或labels是三维张量，则会自动扩展mask或labels的维度，使其与logits在形状上匹配。
        """
        """
        初始化方法。
        参数:
            reduction (str): 指定损失如何减少。'mean'代表平均，'sum'代表求和。
        """
        super().__init__()
        self.reduction = reduction
        # self.loss_func = torch.nn.MSELoss(reduction="mean") # 这一行被注释掉了，暂时没有使用MSELoss。

    def forward(self, logits, labels, mask=None):
        """
        前向计算方法。
        参数:
            logits (Tensor): 模型的输出。
            labels (Tensor): 真实标签。
            mask (Tensor, optional): 遮罩张量，用于选择性地计算损失。
        返回:
            Tensor: L1损失的值。
        """
        # 如果没有提供遮罩，则直接使用logits和labels
        if mask is None:
            off_logits = logits
            off_labels = labels
        else:
            # 创建bool类型的遮罩，值大于0的位置为True，其他位置为False
            bool_mask = mask > 0

            # 如果logits是四维张量，而mask是三维张量，则扩展mask的维度
            if len(logits.shape) == 4 and len(mask.shape) == 3:
                channel = logits.shape[1]
                # 扩展mask的shape从[batch_size, h, w]到[batch_size, channel, h, w]
                bool_mask = bool_mask.unsqueeze(1)
                bool_mask = torch.repeat_interleave(bool_mask, repeats=channel, dim=1)

            # 如果logits是四维张量，而labels是三维张量，则扩展labels的维度
            if len(logits.shape) == 4 and len(labels.shape) == 3:
                channel = logits.shape[1]
                labels = labels.unsqueeze(1)
                labels = torch.repeat_interleave(labels, repeats=channel, dim=1)

            # 使用bool遮罩选取对应的logits和labels的值
            off_logits = torch.masked_select(logits, bool_mask)
            off_labels = torch.masked_select(labels, bool_mask)

        # 计算L1损失
        loss = F.l1_loss(off_logits, off_labels, reduction=self.reduction)
        return loss

@LOSSES.register_module()
class Lane_FocalLoss(nn.Module):
    # 处理类别不平衡问题的损失函数
    def __init__(self, gamma=2, alpha=[0.5, 0.5], size_average=True):
        """
        初始化方法。

        参数:
            gamma (float): FocalLoss的gamma系数。
            alpha (list or float or int): 类别的权重，用于平衡正负样本。默认为两个类别的权重均为0.5。
            size_average (bool): 如果为True，则返回loss的平均值；否则返回loss的总和。
        """
        super(Lane_FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): 
            self.alpha = torch.Tensor([1 - alpha, alpha])
        if isinstance(alpha, list): 
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        self.eps = 1e-10  # 避免数值不稳定，如log(0)

    def forward(self, outputs, targets):
        """
        前向计算方法。
        参数:
            outputs (Tensor): 模型的输出。
            targets (Tensor): 真实标签。
        返回:
            Tensor: FocalLoss的值。
        """
        # 将outputs和targets重塑为(N, 1)的形状
        outputs, targets = torch.sigmoid(outputs.reshape(-1, 1)), targets.reshape(-1, 1).long()  # (N, 1)
        # 沿dim=1的方向堆叠(1-outputs)和outputs，使其形状变为(N, 2)
        outputs = torch.cat((1 - outputs, outputs), dim=1)  # (N, 2)
        # 使用gather方法获取每个样本对应的概率值
        pt = outputs.gather(1, targets).view(-1)
        
        # 计算log概率
        logpt = torch.log(outputs + self.eps)
        logpt = logpt.gather(1, targets).view(-1)

        # 如果alpha不为None，则应用alpha权重
        if self.alpha is not None:
            if self.alpha.type() != outputs.data.type():
                self.alpha = self.alpha.type_as(outputs.data)
            at = self.alpha.gather(0, targets.data.view(-1))
            logpt = logpt * at

        # 计算FocalLoss
        loss = -1 * (1 - pt)**self.gamma * logpt
        
        # 返回平均值或总和，取决于self.size_average
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

@LOSSES.register_module()
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        # inputs = inputs.squeeze(1)
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        r"""此函数用于测量目标与输入对数间的二进制交叉熵。
        查看 :class:`~torch.nn.BCEWithLogitsLoss` 了解详情。
        Args:
            input: 任意形状的张量，作为非标准化的得分（常称为对数）。
            target: 形状与input相同的张量，其值在0和1之间。
            weight (Tensor, 可选): 手动的重新调节权重。若提供，则会重复以匹配input张量的形状。
            size_average (bool, 可选): 已废弃 (查看 :attr:`reduction`)。默认情况下，对每批中的每个损失元素进行平均。注意，对于某些损失，每个样本有多个元素。如果 :attr:`size_average` 设为 ``False``，则损失会对每个小批量进行累加。当reduce为 ``False`` 时，此参数会被忽略。默认: ``True``。
            reduce (bool, 可选): 已废弃 (查看 :attr:`reduction`)。默认情况下，根据 :attr:`size_average` 对每批次的损失进行平均或累加。当 :attr:`reduce` 为 ``False`` 时，返回每批次元素的损失，并忽略 :attr:`size_average`。默认: ``True``。
            reduction (string, 可选): 指定应用于输出的减少方式: ``'none'`` | ``'mean'`` | ``'sum'``。``'none'``: 不进行任何减少，``'mean'``: 输出的总和将除以输出中的元素数，``'sum'``: 输出将被求和。注意：:attr:`size_average` 和 :attr:`reduce` 正在废弃中，同时指定这两个参数将覆盖 :attr:`reduction`。默认: ``'mean'``。
            pos_weight (Tensor, 可选): 正样本的权重。必须是长度与类数相等的向量。
        示例::
            >>> input = torch.randn(3, requires_grad=True)
            >>> target = torch.empty(3).random_(2)
            >>> loss = F.binary_cross_entropy_with_logits(input, target)
            >>> loss.backward()
        """
        # 下面是对binary_cross_entropy_with_logits函数的详细解释
        r"""此函数用于测量目标与输入对数间的二进制交叉熵...
        """
        
        # 计算pt，它是(1 - p)的估计值，其中p是目标值
        pt = torch.exp(-BCE_loss)
        # 计算Focal Loss，它为困难分类的实例赋予了更大的权重
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        # 根据指定的减少方式返回损失
        if self.reduce == 'mean':
            return torch.mean(F_loss)  # 返回F_loss的平均值
        elif self.reduce == 'sum':
            return torch.sum(F_loss)  # 返回F_loss的总和
        else:
            raise NotImplementedError  # 如果指定了不支持的减少方式，则抛出异常

# 使用@LOSSES.register_module()装饰器注册了一个新的损失模块。
@LOSSES.register_module()
class RegL1Loss(nn.Module):
    def __init__(self, ignore_index=255):
        super(RegL1Loss, self).__init__()
        self.ignore_index = ignore_index  # 忽略的索引值，这个索引值的数据在损失计算中将被忽略
        self.fg_threshold = 0.2  # 一个阈值，决定哪些mask值应被视为前景（或有效）

    def forward(self, output, target, mask):
        # 如果提供了遮罩
        if mask is not None:
            _mask = mask.detach().clone()  # 创建mask的一个副本，不需要梯度
            # 将遮罩中值为ignore_index的位置设为0
            _mask[mask == self.ignore_index] = 0.
            # 将遮罩中值小于等于fg_threshold的位置设为0
            _mask[_mask <= self.fg_threshold] = 0.
            # 将遮罩中值大于fg_threshold的位置设为1.0
            _mask[_mask > self.fg_threshold] = 1.0
            # 计算L1损失，但只在_mask为1的位置上进行计算
            loss = F.l1_loss(output * _mask, target * _mask, reduction='mean')
        else:
            # 如果没有提供遮罩，则直接计算L1损失
            loss = F.l1_loss(output, target, reduction='mean')
        return loss  # 返回计算得到的损失值

@LOSSES.register_module()
class OhemLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(OhemLoss, self).__init__()
        self.ignore_index = ignore_index
        self.fg_threshold = 0.2

        self.smooth_l1_sigma = 1.0
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='none')  # reduce=False

    def forward(self, inputs, targets):
        # inputs = inputs.squeeze(1)
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce == 'mean':
            return torch.mean(F_loss)
        elif self.reduce == 'sum':
            return torch.sum(F_loss)
        else:
            raise NotImplementedError

    def ohem_loss(self, batch_size, cls_pred, cls_target, loc_pred, loc_target):
        """    Arguments:
         batch_size (int): number of sampled rois for bbox head training
         loc_pred (FloatTensor): [R, 4], location of positive rois
         loc_target (FloatTensor): [R, 4], location of positive rois
         pos_mask (FloatTensor): [R], binary mask for sampled positive rois
         cls_pred (FloatTensor): [R, C]
         cls_target (LongTensor): [R]
         Returns:
               cls_loss, loc_loss (FloatTensor)
        """

        ohem_cls_loss = F.cross_entropy(cls_pred, cls_target, reduction='none', ignore_index=-1)
        ohem_loc_loss = self.smooth_l1_loss(loc_pred, loc_target).sum(dim=1)

        print(ohem_cls_loss.shape, ohem_loc_loss.shape)
        loss = ohem_cls_loss + ohem_loc_loss

        sorted_ohem_loss, idx = torch.sort(loss, descending=True)
        # 再对loss进行降序排列

        keep_num = min(sorted_ohem_loss.size()[0], batch_size)
        # 得到需要保留的loss数量

        if keep_num < sorted_ohem_loss.size()[0]:
            # 这句的作用是如果保留数目小于现有loss总数，则进行筛选保留，否则全部保留

            keep_idx_cuda = idx[:keep_num]  # 保留到需要keep的数目
            ohem_cls_loss = ohem_cls_loss[keep_idx_cuda]
            ohem_loc_loss = ohem_loc_loss[keep_idx_cuda]  # 分类和回归保留相同的数目

        cls_loss = ohem_cls_loss.sum() / keep_num
        loc_loss = ohem_loc_loss.sum() / keep_num  # 然后分别对分类和回归loss求均值
        return cls_loss, loc_loss

@LOSSES.register_module()
class LanePushPullLoss(nn.Module):
    """
    An embedding loss to min var of per cluster and max distance between different clusters.

    So, for easier cluster, margin_var should be small, and margin_dist should be larger

    Inputs:
    featmap: prediction of network, [b,N,h,w], float tensor
    gt: gt, [b,N,h,w], long tensor, all val >= ignore_label will NOT be contributed to loss.

    loss = var_weight * var_loss + dist_weight * dist_loss

    Args:
        var_weight (float):
        dist_weight (float):
        margin_var (float): margin for var, any var < this margin will NOT be counted in loss
        margin_dist (float): margin for distance, any distance > this margin will NOT be counted in loss
        ignore_label: val in gt >= this arg, will be ignored.
    """

    def __init__(self, var_weight, dist_weight, margin_var, margin_dist, ignore_label):
        super(LanePushPullLoss, self).__init__()
        self.var_weight = var_weight
        self.dist_weight = dist_weight
        self.margin_var = margin_var
        self.margin_dist = margin_dist
        self.ignore_label = ignore_label

    def forward(self, featmap, gt):
        assert (featmap.shape[2:] == gt.shape[2:])
        pull_loss = []
        push_loss = []
        C = gt[gt < self.ignore_label].max().item()
        # [B, N, H, W] = fm, [B, 1, H, W]  = gt
        # TODO not an optimized implement here. Should not expand B dim.
        for b in range(featmap.shape[0]):
            bfeat = featmap[b]
            bgt = gt[b][0]
            instance_centers = {}
            for i in range(1, int(C) + 1):
                instance_mask = bgt == i
                if instance_mask.sum() == 0:
                    continue
                pos_featmap = bfeat[:, instance_mask].T.contiguous()  # mask_num x N
                instance_center = pos_featmap.mean(dim=0, keepdim=True)  # N x mask_num (mean)-> N x 1
                instance_centers[i] = instance_center
                # TODO xxx
                instance_loss = torch.clamp(torch.cdist(pos_featmap, instance_center) - self.margin_var, min=0.0)
                pull_loss.append(instance_loss.mean())
            for i in range(1, int(C) + 1):
                for j in range(1, int(C) + 1):
                    if i == j:
                        continue  # No need to push
                    if i not in instance_centers or j not in instance_centers:
                        continue
                    instance_loss = torch.clamp(
                        2 * self.margin_dist - torch.cdist(instance_centers[i], instance_centers[j]), min=0.0)
                    push_loss.append(instance_loss)
        if len(pull_loss) > 0:
            pull_loss = torch.cat([item.unsqueeze(0) for item in pull_loss]).mean() * self.var_weight
        else:
            pull_loss = 0.0 * featmap.mean()  # Fake loss

        if len(push_loss) > 0:
            push_loss = torch.cat([item.unsqueeze(0) for item in push_loss]).mean() * self.dist_weight
        else:
            push_loss = 0.0 * featmap.mean()  # Fake loss
        return push_loss + pull_loss
@LOSSES.register_module()
class IoULoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(IoULoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, outputs, targets):
        mask = (targets != self.ignore_index).float()
        targets = targets.float()
        num = torch.sum(outputs * targets * mask)
        den = torch.sum(outputs * mask + targets * mask - outputs * targets * mask)
        return 1 - num / den

class BevLaneLoss(nn.Module):
    def __int__(self):
        super(BevLaneLoss, self).__init__()
        # self.off_loss = OffsetLoss()
        # self.seg_loss = FocalLoss()
        # self.haf_loss = RegL1Loss()
        # self.vaf_loss = RegL1Loss()

    def forward(self, labels, preds):
        binary_seg, embedding, haf_pred, vaf_pred, off_pred = preds
        # seg_mask, haf_mask, vaf_mask, mask_offset = labels
        total_loss = dict()
        #............
        # only for test
        seg_mask, _, haf_mask, vaf_mask, mask_offset = preds
        #...........
        device = seg_mask.device
        seg_mask, haf_mask, vaf_mask, mask_offset = seg_mask.to(device), haf_mask.to(device), vaf_mask.to(device), mask_offset.to(device)

        # haf_loss = self.haf_loss(haf_pred, haf_mask, seg_mask)
        # vaf_loss = self.vaf_loss(vaf_pred, vaf_mask, seg_mask)
        # seg_loss = self.seg_loss(binary_seg, seg_mask)

        haf_loss = RegL1Loss().forward(haf_pred, vaf_mask, seg_mask)
        vaf_loss = RegL1Loss().forward(vaf_pred, haf_mask, seg_mask)
        seg_loss = Lane_FocalLoss().forward(binary_seg, haf_mask)

        total_loss['loss'] = seg_loss + haf_loss + vaf_loss

        # total_loss['loss'] = torch.tensor(0.5).requires_grad_(True)
        return total_loss




