import cv2
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32

from mmdet.models import DETECTORS
from .. import builder
from mmdet3d.models.detectors.base import Base3DDetector

@DETECTORS.register_module()
class LaneAF2D(Base3DDetector):
    def __init__(self, img_backbone, img_neck, lane_head, train_cfg, test_cfg, pretrained=None, **kwargs):
        super(LaneAF2D, self).__init__(**kwargs)
        if img_backbone:
            self.img_backbone = builder.build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = builder.build_neck(img_neck)
        self.lane_head = builder.build_head(lane_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.pretrained = pretrained

    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        #   N是样本数量
        imgs = imgs.view(B * N, C, imH, imW)
        x = self.img_backbone(imgs)  # [[1, 64, 160, 240], [1, 128, 80, 120], [1, 256, 40, 60] [1, 512, 20, 30]
        if self.with_img_neck:       # [1, 256, 160, 240]
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        if N!=1:
            x = x.view(B, N, output_dim, ouput_H, output_W)
        return x

    def extract_img_feat(self, img, img_metas, **kwargs):
        """Extract features of images."""
        # features, ks, imu2cs, post_rots, post_trans, undists
        ks, imu2cs, post_rots, post_trans, undists = img[1], img[2], img[3], img[4], img[5]
        grid, drop_idx = img[9], img[10]
        x = self.image_encoder(img[0])
        return x

    #   output['img_inputs'] = (imgs, rots, trans, intrins, post_rots,
    #                                  post_trans, bda_rot, extrinsics, undists, gt_lanes)
    # output['maps'] = (gt_mask, mask_haf, mask_vaf, mask_offset)
    def extract_feat(self, img, img_metas, **kwargs):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas, **kwargs)
        return img_feats

    def forward_lane_train(self, img_feats, gt_labels, **kwargs):
        #NOTE: input = [seg_mask, haf_mask, vaf_mask, mask_offset]
        # array1 = maps[0].detach().cpu().numpy()
        # cv2.imwrite("./map.png", np.argmax(array1, axis=0) * 100)
        outs = self.lane_head(img_feats)
        loss_inputs = [outs, gt_labels]
        # losses = self.bev_lane_loss(*loss_inputs)
        losses = self.lane_head.loss(*loss_inputs)
        return losses, outs

    def forward_train(self,
                      img_metas=None,
                      img_inputs=None,
                      maps_bev=None,
                      maps_2d=None,
                      label_bev=None,
                      label_2d=None,
                      **kwargs):

        img_feats = self.extract_feat(img=img_inputs, img_metas=img_metas, **kwargs)
        losses = dict()
        losses_lane, out = self.forward_lane_train(img_feats, maps_2d)
        losses.update(losses_lane)
        return losses, out

    def forward_test(self,
                      img_metas=None,
                      img_inputs=None,
                      maps_bev=None,
                      maps_2d=None,
                      **kwargs):
        return self.simple_test(img_metas, img_inputs, maps_2d)

    def aug_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        assert False

    def simple_test(self,
                      img_metas=None,
                      img_inputs=None,
                      gt_labels=None,
                      **kwargs):
        img_feats = self.extract_feat(img=img_inputs, img_metas=img_metas, **kwargs)
        outs = self.lane_head(img_feats)
        return outs

    def forward_dummy(self,
                      img_metas=None,
                      img_inputs=None,
                      maps_bev=None,
                      maps_2d=None,
                      **kwargs):
        img_feats, _ = self.extract_feat(img=img_inputs, img_metas=img_metas, **kwargs)
        outs = self.lane_head(img_feats)
        return outs

    @property
    def with_img_neck(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, 'img_neck') and self.img_neck is not None



@DETECTORS.register_module()
class LaneATT2D(LaneAF2D):
    def __init__(self, img_backbone, img_neck, lane_head, train_cfg, test_cfg, pretrained=None, **kwargs):
        super(LaneATT2D, self).__init__(img_backbone, img_neck, lane_head, train_cfg, test_cfg, pretrained=None, **kwargs)

    def forward_train(self,
                      img_metas=None,
                      img_inputs=None,
                      maps_bev=None,
                      maps_2d=None,
                      label_bev=None,
                      label_2d=None,
                      **kwargs):

        img_feats = self.extract_feat(img=img_inputs, img_metas=img_metas, **kwargs)
        # 这里他们的模型输出不同
        losses = dict()
        losses_lane, out = self.forward_lane_train(img_feats, label_2d)
        losses.update(losses_lane)
        return losses, out

    def forward_test(self,
                      img_metas=None,
                      img_inputs=None,
                      maps_bev=None,
                      maps_2d=None,
                      label_bev=None,
                      label_2d=None,
                      **kwargs):
        return self.simple_test(img_metas, img_inputs, label_2d)