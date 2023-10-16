import torch.nn as nn
from mmcv.runner import force_fp32,auto_fp16
from mmdet3d.models.builder import HEADS
# from mmseg.models.builder import HEADS

from .base_head import BaseHead


@HEADS.register_module()
class LaneAFHead(BaseHead):
    def __init__(self, init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform'),
                     **kwargs):
        super(LaneAFHead, self).__init__(num_classes=2, init_cfg=init_cfg, **kwargs)

        def cbrc(out_channels):
            return nn.Sequential(
                nn.Conv2d(self.in_channels, self.channels, 1),
                nn.BatchNorm2d(self.channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.channels, out_channels, 1)
                )

        self.binary = cbrc(2)
        self.haf = cbrc(1)
        self.vaf = cbrc(2)
        self.edge = cbrc(2)
        self.singleline = cbrc(3)
        # self.doubleline = cbrc(5)

    @auto_fp16()
    def forward(self,inputs):
        x = inputs[0]
        binary = self.binary(x)
        haf = self.haf(x)
        vaf = self.vaf(x)
        edge_binary = self.edge(x)
        single_label = self.singleline(x)
        # doubleline = self.doubleline(x)

        return dict(
            binary=binary,
            haf=haf,
            vaf=vaf,
            edge_binary=edge_binary,
            single_label=single_label
        )

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label):
        """Compute loss."""
        
        return self.loss_decode(seg_logit, seg_label)
    
