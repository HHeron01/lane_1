# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from .. import builder
# from ..builder import TASKS
# from thomas.core import add_prefix

# from mmseg.models.segmentors.base import BaseSegmentor
# from mmdet.models.detectors.base import BaseDetector
# from mmdet3d.models.detectors.base import Base3DDetector
# from mmdet3d.models.segmentors.base import Base3DSegmentor

# class SegHead(BaseSegmentor):
#     pass
# class LaneHead(BaseSegmentor):
#     pass
# class Od2dHead(BaseDetector):
#     pass
# class Od3dHead(Base3DDetector):
#     pass
# class Lane3dHead(Base3DSegmentor):
#     pass

# @TASKS.register_module()
# class OneNet():
#     def __init__(self, backbone, neck, seg_head, lane_head, od2d_head,**kwargs) -> None:
#         self.backbone = builder.build_backbone(backbone)
#         self.seg = builder.build_head(seg_head. self.backbone)
#         self.lane = builder.build_head

