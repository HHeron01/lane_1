import copy
import os
import torch
import torch.nn as nn
from mmdet3d.models.detectors import Base3DDetector
from mmdet3d.core import bbox3d2result
from mmdet.core import multi_apply
from mmdet.models import DETECTORS
from mmdet3d.models.multiview.vrm_bevlane_net import VRM_BEVLane
from mmcv.runner import force_fp32, auto_fp16


@DETECTORS.register_module()
class BEVLaneTraced(nn.Module):
    def __init__(self, model):
        super(BEVLaneTraced, self).__init__()
        _model = copy.deepcopy(model)
        self.img_backbone = _model.img_backbone
        # self.img_neck = _model.img_neck
        self.img_view_transformer = _model.img_view_transformer
        self.bev_lane_head = _model.bev_lane_head
        self.loss = _model.bev_lane_head.loss

    def image_encoder(self, img):
        imgs = img
        x = self.img_backbone(imgs)
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
    def extract_feat(self, img, img_metas=None):
        """Extract features from images and points."""
        img_feat, bev_feat = self.extract_img_feat(img)
        return img_feat, bev_feat

    # @auto_fp16()
    def forward(self, image,
                      ipm_gt_segment=None,
                      ipm_gt_instance=None,
                      ipm_gt_offset=None,
                      ipm_gt_z=None,
                      img_metas=None,
                      image_gt_segment=None,
                      image_gt_instance=None,
                      **kwargs):
        x = self.img_backbone(image)
        bev_feat, img_feat = self.img_view_transformer(x)
        # img_feat, bev_feat = self.extract_feat(image)
        bev_out, img_out = self.bev_lane_head(bev_feat, img_feat)
        return bev_out, img_out


@DETECTORS.register_module()
class BEVLaneForward(Base3DDetector):
# class BEVLaneForward(VRM_BEVLane):
# class BEVLaneForward(nn.Module):
    def __init__(self, ori_model, graph_module, return_loss=True,
                 **kwargs):
        super(BEVLaneForward, self).__init__(
                 # img_backbone=None,
                 # img_neck=None,
                 # img_view_transformer=None,
                 # bev_lane_head=None,
                 # train_cfg=None,
                 # test_cfg=None,
                 # pretrained=None,
                 )
        self.graph_module = graph_module
        self.loss = ori_model.loss
        # self.loss = return_loss

    @auto_fp16()
    def extract_feat(self, image):
        bev_out, img_out = self.graph_module(image)
        return bev_out, img_out


    # @force_fp32(apply_to=('loss_inputs',))
    def return_loss(self, loss_inputs):
        losses = self.loss(*loss_inputs)
        return losses

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

        # bev_out, img_out = self.graph_module(image)
        bev_out, img_out = self.extract_feat(image)
        maps = (ipm_gt_segment, ipm_gt_instance, ipm_gt_offset, ipm_gt_z, image_gt_segment, image_gt_instance)
        loss_inputs = [bev_out, img_out, maps]
        losses = self.return_loss(loss_inputs)
        # losses = self.loss(*loss_inputs)
        return losses, (bev_out, img_out)

    def forward(self,
            image=None,
            ipm_gt_segment=None,
            ipm_gt_instance=None,
            ipm_gt_offset=None,
            ipm_gt_z=None,
            img_metas=None,
            image_gt_segment=None,
            image_gt_instance=None,
            **kwargs):
        if self.return_loss:
            return self.forward_train(image=image,
            ipm_gt_segment=ipm_gt_segment,
            ipm_gt_instance=ipm_gt_instance,
            ipm_gt_offset=ipm_gt_offset,
            ipm_gt_z=ipm_gt_z,
            img_metas=img_metas,
            image_gt_segment=image_gt_segment,
            image_gt_instance=image_gt_instance,
            **kwargs)
        else:
            return self.forward_test(image=image)
            # ipm_gt_segment=ipm_gt_segment,
            # ipm_gt_instance=ipm_gt_instance,
            # ipm_gt_offset=ipm_gt_offset,
            # ipm_gt_z=ipm_gt_z,
            # img_metas=img_metas,
            # image_gt_segment=image_gt_segment,
            # image_gt_instance=image_gt_instance,
            # **kwargs)


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
        # img_feats, bev_feat = self.extract_feat(img=image, img_metas=img_metas)
        bev_out, img_out = self.graph_module(image)
        return (bev_out, img_out)


    def forward_test(self,
                     image=None,
                     ):
                     # ipm_gt_segment=None,
                     # ipm_gt_instance=None,
                     # ipm_gt_offset=None,
                     # ipm_gt_z=None,
                     # img_metas=None,
                     # image_gt_segment=None,
                     # image_gt_instance=None,
                     # **kwargs):
        bev_out, img_out = self.graph_module(image)
        return (bev_out, img_out)

    def forward_dummy(self,
                    image=None,
                    ):
        bev_out, img_out = self.graph_module(image)
        binary_seg, embedding, offset_feature, z_feature, topdown = bev_out
        binary_seg_2d, embedding_2d = img_out
        # return bev_out, img_out
        return binary_seg, embedding, offset_feature, z_feature, topdown, binary_seg_2d, embedding_2d
