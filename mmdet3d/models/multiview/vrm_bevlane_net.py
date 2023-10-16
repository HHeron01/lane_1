# Copyright (c) Phigent Robotics. All rights reserved.
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32, auto_fp16

from mmdet3d.ops.bev_pool_v2.bev_pool import TRTBEVPoolv2
from mmdet.models import DETECTORS
from .. import builder
from mmdet3d.models.detectors.centerpoint import CenterPoint
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d.models.detectors.base import Base3DDetector


@DETECTORS.register_module()
class VRM_BEVLane(Base3DDetector):
    def __init__(self, img_backbone, img_neck, img_view_transformer,
       bev_lane_head, train_cfg, test_cfg, pretrained=None, **kwargs):
        super(VRM_BEVLane, self).__init__(**kwargs)
        if img_backbone:
            self.img_backbone = builder.build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = builder.build_neck(img_neck)
        self.img_view_transformer = builder.build_neck(img_view_transformer)

        self.bev_lane_head = builder.build_head(bev_lane_head)
        # self.bev_lane_loss = builder.build_loss(bev_lane_loss)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.pretrained = pretrained

    # @auto_fp16()
    def image_encoder(self, img):
        imgs = img
        # B, N, C, imH, imW = imgs.shape
        # imgs = imgs.view(B * N, C, imH, imW)
        x = self.img_backbone(imgs)
        # if self.with_img_neck:
        #     x = self.img_neck(x
        #     if type(x) in [list, tuple]:
        #         x = x[0]
        # _, output_dim, ouput_H, output_W = x.shape
        # x = x.view(B, N, output_dim, ouput_H, output_W)
        return x

    def extract_img_feat(self, img):
        """Extract features of images."""
        img_feats = self.image_encoder(img)
        # img_feats = self.img_neck(img_feats)
        bev_feat, img_feat = self.img_view_transformer(img_feats)

        return img_feat, bev_feat

    #   output['img_inputs'] = (imgs, rots, trans, intrins, post_rots,
    #                                  post_trans, bda_rot, extrinsics, undists, gt_lanes)
    # output['maps'] = (gt_mask, mask_haf, mask_vaf, mask_offset)
    def extract_feat(self, img, img_metas):
        """Extract features from images and points."""
        img_feat, bev_feat = self.extract_img_feat(img)
        return img_feat, bev_feat

    def forward_lane_train(self, bev_feat, maps, img_feat=None, **kwargs):
        #NOTE: input = [seg_mask, haf_mask, vaf_mask, mask_offset]
        # array1 = maps[0].detach().cpu().numpy()
        # cv2.imwrite("./map.png", np.argmax(array1, axis=0) * 100)

        # outs = self.bev_lane_head(bev_feat)
        bev_out, img_out = self.bev_lane_head(bev_feat, img_feat)
        loss_inputs = [bev_out, img_out, maps]
        # losses = self.bev_lane_loss(*loss_inputs)
        losses = self.bev_lane_head.loss(*loss_inputs)
        return losses, (bev_out, img_out)

    def forward_lane_test(self, bev_feat, img_feat=None, **kwargs):
        #NOTE: input = [seg_mask, haf_mask, vaf_mask, mask_offset]
        # array1 = maps[0].detach().cpu().numpy()
        # cv2.imwrite("./map.png", np.argmax(array1, axis=0) * 100)
        bev_out, img_out = self.bev_lane_head(bev_feat, img_feat)

        return (bev_out, img_out)

    # @auto_fp16()
    def forward_train(self,
                      image=None,
                      ipm_gt_segment=None,
                      ipm_gt_instance=None,
                      ipm_gt_offset=None,
                      ipm_gt_z=None,
                      img_metas=None,
                      image_gt_segment=None,
                      image_gt_instance=None,
                      **kwargs):

        maps = (ipm_gt_segment, ipm_gt_instance, ipm_gt_offset, ipm_gt_z, image_gt_segment, image_gt_instance)
        img_feat, bev_feat = self.extract_feat(img=image, img_metas=img_metas)
        losses = dict()
        losses_lane, out = self.forward_lane_train(bev_feat, maps, img_feat)
        losses.update(losses_lane)
        return losses, out

    def forward_test(self,
                     image=None,
                     ipm_gt_segment=None,
                     ipm_gt_instance=None,
                     ipm_gt_offset=None,
                     ipm_gt_z=None,
                     img_metas=None,
                      **kwargs):
        return self.simple_test(image,
                    ipm_gt_segment,
                    ipm_gt_instance,
                    ipm_gt_offset,
                    ipm_gt_z,
                    img_metas)

    def aug_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        assert False

    def simple_test(self,
                    image=None,
                    ipm_gt_segment=None,
                    ipm_gt_instance=None,
                    ipm_gt_offset=None,
                    ipm_gt_z=None,
                    img_metas=None,
                    **kwargs):
        img_feats, bev_feat = self.extract_feat(img=image, img_metas=img_metas)
        bev_out, img_out = self.bev_lane_head(bev_feat, img_feats)
        return bev_out, img_out

    def forward_dummy(self,
                    image=None,
                    ipm_gt_segment=None,
                    ipm_gt_instance=None,
                    ipm_gt_offset=None,
                    ipm_gt_z=None,
                    img_metas=None,
                    **kwargs):
        img_feats, bev_feat = self.extract_feat(img=image, img_metas=img_metas)
        bev_out, img_out = self.bev_lane_head(bev_feat, img_feats)
        binary_seg, embedding, offset_feature, z_feature, topdown = bev_out
        binary_seg_2d, embedding_2d = img_out
        # return bev_out, img_out
        return binary_seg, embedding, offset_feature, z_feature, binary_seg_2d, embedding_2d

    @property
    def with_img_neck(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, 'img_neck') and self.img_neck is not None
