
import numpy as np
from mmcv.utils import print_log
import os

import cv2
import numpy as np
import json

from mmdet.datasets import DATASETS
import torchvision.transforms as transforms
import torch
import random
import numbers
import matplotlib.pyplot as plt
from .custom import LaneCustomDataset
# from mmseg.datasets.custom import CustomDataset

@DATASETS.register_module()
class Lane2dDataset(LaneCustomDataset):
    """lane2d dataset.
    """

    CLASSES = ["single", "double", "edge"]
    PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70]]

    def __init__(self,
                 **kwargs
                 ):
        super(Lane2dDataset, self).__init__(**kwargs)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
            split):
        """
        load json files from ingot
        """
        return self.load_annotations_offline(img_dir, ann_dir)
        # return self.load_annotation_online(img_dir, img_suffix, ann_dir, seg_map_suffix, split)
        

    def load_annotations_offline(self, img_dir, ann_dir):
        img_infos = []
        import json
        print(img_dir)
        print(ann_dir)
        ann_files = os.listdir(ann_dir)
        for ann_file in ann_files:
            ann_path = os.path.join(ann_dir, ann_file)
            with open(ann_path) as ft:
                data = json.load(ft)
                img_suffix = data['meta']['sensors'][0]['url']
                img_info = dict(filename=img_suffix)
                img_info['ann'] = dict(seg_map=ann_file)
            img_infos.append(img_info)
        img_infos = sorted(img_infos, key=lambda x:x['filename'])
       
        return img_infos
    
    def load_annotation_online(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
            split):
            # TODO: get annotation from ingot
            # step 1: init_ingot
            # step 2: load json file and get img_paths(from json url) and ann info(json path)
            # step 3: infos[{"filename":url_path,"ann":json_path},..,..]
            pass

    def evaluate(self, results, metric='mIoU', logger=None, gt_seg_maps=None, **kwargs):
        print("only support format results, after evaluate")

    def pre_eval(self, preds, indices):
        print("only support format results, after evaluate")

    def format_results(self, results, imgfile_prefix, indices=None, **kwargs):
        assert imgfile_prefix != self.ann_dir, "cannot overwrite json to ann dir"

        ann_info = self.get_ann_info(indices[0])['seg_map']
        save_path = os.path.join(imgfile_prefix, ann_info)
        with open(save_path, 'w') as ft:
            data = dict(objects=results['lane']['lane'])
            json.dump(data, ft, indent=2)
        
        return save_path