import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import builder
from mmdet.models import DETECTORS
import numpy as np
import cv2
import warnings

from mmseg.models.segmentors.base import BaseSegmentor
from mmseg.core import add_prefix


@DETECTORS.register_module()
class OneNet(BaseSegmentor):
    """OneNet including seg,lane,od2d,od3d...
    """
    def __init__(self,
                backbone,
                neck,
                use_psp,
                seg_head=None,
                lane_head=None,
                od2d_head=None,
                od3d_head=None,
                init_cfg=None,
                pretrained=None,
                train_cfg=None,
                test_cfg=None):
        super(OneNet, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'backbone set pretrained weight'
            backbone.pretrained = pretrained

        self.backbone = builder.build_backbone(backbone)
        self.neck = builder.build_neck(neck)

        if use_psp:
            warnings.warn("TODO:psp layer need to do after.")
        
        # heads
        self.heads = dict()
        if seg_head is not None:
            self.seg_head = builder.build_head(seg_head)
            self.heads["seg"] = self.seg_head
        if lane_head is not None:
            self.lane_head = builder.build_head(lane_head)
            self.heads['lane'] = self.lane_head
        if od2d_head is not None:
            self.heads['od2d'] = builder.build_head(od2d_head)
        if od3d_head is not None:
            self.heads['od3d'] = builder.build_head(od3d_head)
        
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img):
        x = self.backbone(img)
        x = self.neck(x)
        return x
    
    def encode_decode(self, img, img_metas):
        x = self.extract_feat(img)
        out = dict()
        for head_name in self.heads:
            out_head = self.heads[head_name].forward_test(x, img_metas, self.test_cfg)
            out.update(head_name=out_head)
        return out

    def forward_dummy(self, img):
        output = self.encode_decode(img, None)
        return output

    def forward_train(self, imgs, img_metas, **kwargs):
        x = self.extract_feat(imgs)

        losses = dict()
        for head_name in self.heads:
            loss = self.heads[head_name].forward_train(x, img_metas, 
                                                    {'label':kwargs['label'], 
                                                    'vaf':kwargs['vaf'], 
                                                    'haf':kwargs['haf'],
                                                    'edge_binary':kwargs['edge_binary'], 
                                                    'single_label':kwargs['single_label']
                                                    },
                                                    self.train_cfg)
            losses.update(add_prefix(loss, head_name))
        # self.lane_head.forward_train(x,img_metas,None, self.train_cfg)
        
        return losses
    
    def simple_test(self, img, img_meta, **kwargs):
        x = self.extract_feat(img)
        
        outs = dict(img_shape=img_meta[0]['img_shape'],
                    ori_shape=img_meta[0]['ori_shape'])
        if 'lane' in self.heads:
            lane_maps = self.heads['lane'].forward_test(x, img_meta, self.test_cfg)
    
            outs.update(lane=dict(
                edge=lane_maps['edge_binary'],
                single=lane_maps['single_label'],
                binary=lane_maps['binary'],
                haf=lane_maps['haf'],
                vaf=lane_maps['vaf']
            ))

        return outs
    
    def aug_test(self, imgs, img_metas, **kwargs):
        ValueError("TODO")
    
    def show_result(self, img, result, palette=None, win_name='', show=False, wait_time=0, out_file=None, opacity=0.5):
        if 'lane' in result:
            self.show_result_lane(img, result['lane'], win_name + "_lane", show, wait_time, out_file, opacity)

    def merge_sinlge_result(self, img, seg, opacity, title):
        res = img.copy()
        state = np.random.get_state()
        np.random.seed(42)
        palette = np.random.randint(0, 255, size=(40, 3))
        np.random.set_state(state)
        palette = np.array(palette)
   
        assert 0 < opacity <= 1.0
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            if label == 0:
                continue
            color_seg[seg == label, :] = color
        # convert to BGR
        color_seg = color_seg[..., ::-1]

        res = res * (1 - opacity) + color_seg * opacity
        res = res.astype(np.uint8)

        cv2.putText(res, title, (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        return res
 
    
    def show_result_lane(self, img, result, win_name='', show=False, wait_time=0, out_file=None, opacity=0.5):
        import mmcv
        img = mmcv.imread(img)

        # TODO: delete
        img_edge = img.copy()
        edge = result['edge'].astype(np.uint8)
        edge = cv2.resize(edge, (img_edge.shape[1], img_edge.shape[0]))
        img_edge[edge == 1] = (0,255,0)

        img = img.copy()

        seg = result['instance']
        vaf = result['vaf']
        
        scale_x, scale_y = result['scale']
        img = np.ascontiguousarray(img, dtype=np.uint8)
        seg_color = cv2.applyColorMap(40*seg, cv2.COLORMAP_JET)
        rows, cols = np.nonzero(seg)
        for r, c in zip(rows, cols):
            img = cv2.arrowedLine(img, (int(c*scale_y+0.5), int(r*scale_x+0.5)),(int(0.5+ c*scale_y+vaf[0, r, c]*scale_y*0.75), 
                int(0.5+ r*scale_x+vaf[1, r, c]*scale_x*0.5)), seg_color[r, c, :].tolist(), 1, tipLength=0.4)

        for obj in result['lane']:
            pts = np.array(obj["coordinate"]).reshape(-1,2)
            if obj['property']['category'] == 'SolidLane':
                color = (0, 0, 255)
            elif obj['property']['category'] == 'DalidLane':
                color = (255, 0, 0)
            else:
                color = (0, 255, 0)
            for idx in range(1, pts.shape[0]):
                cv2.line(img, pts[idx-1], pts[idx], color, 2)
        
        # TODO: delete
        img = np.hstack([img, img_edge])

        if show:
            mmcv.imshow(img, win_name, wait_time)
        if out_file is not None:
            mmcv.imwrite(img, out_file)

        if not (show or out_file):
            warnings.warn('show==False and out_file is not specified, only '
                          'result image will be returned')
            return img