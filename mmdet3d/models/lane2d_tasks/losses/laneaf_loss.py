import torch
import torch.nn as nn
import torch.nn.functional as F

# from mmdet3d.models.builder import LOSSES
from mmseg.models.builder import LOSSES
from .focal_loss import FocalLoss
from .dice_loss import DiceLoss


class RegL1Loss(nn.Module):
    def __init__(self, ignore_index=255,reduction='sum'):
        super(RegL1Loss, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, output, target, mask):
        _mask = mask.detach().clone()
        _mask[mask == self.ignore_index] = 0.
        loss = F.l1_loss(output * _mask, target * _mask, reduction=self.reduction)
        loss = loss / (_mask.sum() + 1e-12)
        return loss

class OffsetLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(OffsetLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, pred, target, mask=None):
        if mask is None:
            off_pred = pred
            off_target = target
        else:
            bool_mask = mask > 0
            if len(pred.shape) ==4 and len(mask.shape) ==3:
                channel = pred.shape[1]
                bool_mask = bool_mask.unsqueeze(1)
                bool_mask = torch.repeat_interleave(bool_mask,
                    repeats=channel, dim=1)

            if len(pred.shape) ==4 and len(target.shape) ==3:
                channel = pred.shape[1]
                target = target.unsqueeze(1)
                target = torch.repeat_interleave(target, repeats=channel,
                    dim=1)

            off_pred = torch.masked_select(pred, bool_mask)
            off_target = torch.masked_select(target, bool_mask)

        loss = F.l1_loss(off_pred, off_target, reduction=self.reduction)

        return loss 


@LOSSES.register_module()
class LaneAFLoss(nn.Module):
    """LaneAFLoss including focal loss,dice loss, regl1 loss.
    """

    def __init__(self,
                weight_binary=1.0,
                weight_edge=1.0,
                weight_single=1.0,
                weight_double=1.0,
                weight_hafvaf=0.5,
                use_reg_loss=False):
        super(LaneAFLoss, self).__init__()

        self.weight_binary = weight_binary
        self.weight_edge = weight_edge
        self.weight_single = weight_single
        self.weight_double = weight_double
        self.weight_hafvaf = weight_hafvaf

        self.binary_focal = FocalLoss(gamma=2.0, alpha=0.25)
        self.binary_dice = DiceLoss()

        self.edge_focal = FocalLoss(gamma=2.0, alpha=0.25)
        self.edge_dice = DiceLoss()

        self.single_focal = FocalLoss(gamma=2.0, alpha=0.25)
        self.single_dice = DiceLoss()

        # self.double_focal =FocalLoss(gamma=2.0, alpha=0.25)
        # self.double_dice = DiceLoss()

        if use_reg_loss:
            self.haf = RegL1Loss()
            self.vaf = RegL1Loss()
        else:
            self.haf = OffsetLoss()
            self.vaf = OffsetLoss()

    def forward(self, preds, targets):

        binary, edge, single, haf, vaf = preds['binary'], preds['edge_binary'], \
            preds['single_label'], preds['haf'], preds['vaf']
        binary_target, edge_target, single_target, haf_target, vaf_target \
             = targets['label'], targets['edge_binary'], targets['single_label'], \
                targets['haf'], targets['vaf']
        
        binary_target = binary_target.squeeze_(1)
        single_target = single_target.squeeze_(1)
        edge_target = edge_target.squeeze_(1)

        
        binary_target = torch.gt(binary_target, 0).type(torch.long)
        bloss = self.weight_binary*self.binary_focal(binary, binary_target)\
                + self.weight_binary*self.binary_dice(binary, binary_target)

        eloss = self.weight_edge*self.edge_focal(edge, edge_target)\
                + self.weight_edge*self.edge_dice(edge, edge_target)

        sloss = self.weight_single*self.single_focal(single,single_target)\
                + self.weight_single*self.single_dice(single,single_target)

        # loss += self.weight_double*self.double_focal(double, double_target)
        # loss += self.weight_double*self.double_dice(double, double_target)

        if binary_target.sum().item() == 0:
            hvloss = torch.tensor(0,device='cuda',dtype=torch.float32)
        else:
            hvloss = self.weight_hafvaf*self.haf(haf, haf_target, binary_target)\
                + self.weight_hafvaf*self.vaf(vaf, vaf_target, binary_target)

        return dict(
            binaryloss=bloss,
            edgeloss=eloss,
            singleloss=sloss,
            hvloss=hvloss
        )