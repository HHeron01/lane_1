import os
import numpy as np
import cv2
import torch
from PIL import Image
from pyquaternion import Quaternion
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

from torch.utils.data import Dataset
from mmdet3d.datasets.multiview_datasets.seg_bevlane.const import CAMS, NUM_CLASSES, IMG_ORIGIN_H, IMG_ORIGIN_W
from mmdet3d.datasets.multiview_datasets.seg_bevlane.rasterize import preprocess_map
from mmdet3d.datasets.multiview_datasets.seg_bevlane.vector_map import VectorizedLocalMap
from ..image import normalize_img, img_transform

from mmdet.datasets import DATASETS
from mmdet3d.datasets.pipelines import Compose

@DATASETS.register_module()
class Nus_online_SegDataset(Dataset):
    def __init__(self, version='v1.0-mini', data_root='', data_config=None,
                 grid_config=None, test_mode=False, pipeline=None, CLASSES=None, use_valid_flag=True):
        super().__init__()
        # print("grid_config:", grid_config)
        patch_h = grid_config['y'][1] - grid_config['y'][0]
        patch_w = grid_config['x'][1] - grid_config['x'][0]
        canvas_h = int(patch_h / grid_config['y'][2])
        canvas_w = int(patch_w / grid_config['x'][2])

        self.use_valid_flag = use_valid_flag
        self.CLASSES = CLASSES
        self.is_train = not test_mode #1 - test_mode
        self.data_config = data_config
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.nusc = NuScenes(version=version, dataroot=data_root, verbose=False)
        self.vector_map = VectorizedLocalMap(data_root, patch_size=self.patch_size, canvas_size=self.canvas_size)
        self.scenes = self.get_scenes(version, self.is_train)
        self.samples = self.get_samples()
        self.thickness = data_config['thickness']
        self.angle_class = data_config['angle_class']
        if pipeline is not None:
             self.pipeline = Compose(pipeline)
        self.flag = np.zeros(len(self.samples), dtype=np.uint8)

    def __len__(self):
        return len(self.samples)

    def get_scenes(self, version, is_train):
        # filter by scene split
        split = {
            'v1.0-trainval': {True: 'train', False: 'val'},
            'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
        }[version][is_train]

        return create_splits_scenes()[split]

    def get_samples(self):
        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [samp for samp in samples if
                   self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples

    def sample_augmentation(self):
        # print("data:", self.data_config)
        fH, fW = self.data_config['input_size']
        resize = (fW / IMG_ORIGIN_W, fH / IMG_ORIGIN_H)
        resize_dims = (fW, fH)
        return resize, resize_dims

    def get_vectors(self, rec):
        location = self.nusc.get('log', self.nusc.get('scene', rec['scene_token'])['log_token'])['location']
        ego_pose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        vectors = self.vector_map.gen_vectorized_samples(location, ego_pose['translation'], ego_pose['rotation'])
        return vectors

    def get_imgs(self, rec):
        imgs = []
        trans = []
        rots = []
        intrins = []
        post_trans = []
        post_rots = []

        visu_path = './vis_pic'
        if not os.path.exists(visu_path):
            os.makedirs(visu_path)
        for cam in CAMS:
            samp = self.nusc.get('sample_data', rec['data'][cam])
            imgname = os.path.join(self.nusc.dataroot, samp['filename'])
            img = Image.open(imgname)
            img.save(visu_path + '/' + cam + '.png')
            resize, resize_dims = self.sample_augmentation()
            img, post_rot, post_tran = img_transform(img, resize, resize_dims)
            # resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
            # img, post_rot, post_tran = img_transform(img, resize, resize_dims, crop, flip, rotate)

            img = normalize_img(img)
            post_trans.append(post_tran)
            post_rots.append(post_rot)
            imgs.append(img)

            sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
            trans.append(torch.Tensor(sens['translation']))
            rots.append(torch.Tensor(Quaternion(sens['rotation']).rotation_matrix))
            intrins.append(torch.Tensor(sens['camera_intrinsic']))
        return torch.stack(imgs), torch.stack(trans), torch.stack(rots), torch.stack(intrins), torch.stack(post_trans), torch.stack(post_rots)

    def get_semantic_map(self, rec):
        vectors = self.get_vectors(rec)
        instance_masks, forward_masks, backward_masks = preprocess_map(vectors, self.patch_size, self.canvas_size,
                                                                       NUM_CLASSES, self.thickness,
                                                                       self.angle_class)
        semantic_masks = instance_masks != 0
        semantic_masks = torch.cat([(~torch.any(semantic_masks, axis=0)).unsqueeze(0), semantic_masks])
        return semantic_masks

    def get_data_info(self, index):
        rec = self.samples[index]
        imgs, trans, rots, intrins, post_trans, post_rots = self.get_imgs(rec)
        # print("trans:", trans)
        semantic_masks = self.get_semantic_map(rec)
        visu_path = './vis_pic'
        if not os.path.exists(visu_path):
            os.makedirs(visu_path)
        array1 = semantic_masks.detach().cpu().numpy()
        cv2.imwrite(visu_path + "/semantic_masks.png", np.argmax(array1, axis=0) * 100)
        # standard protocol modified from SECOND.Pytorch
        input_dict = dict(
            imgs=imgs,
            trans=trans,
            rots=rots,
            intrins=intrins,
            post_trans=post_trans,
            post_rots=post_rots,
            semantic_masks=semantic_masks,
        )
        return input_dict

    def evaluate(self,  results, logger=None, **kwargs):
        # print(".................................")
        print('.......Evaluating  waiting ......')
        eval_res = dict()
        return eval_res


    def __getitem__(self, idx):
        input_dict = self.get_data_info(idx)
        # return input_dict
        data = self.pipeline(input_dict)
        return data


import tqdm
if __name__ == '__main__':

    data_config = {
        'cams': [
            'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
            'CAM_BACK', 'CAM_BACK_RIGHT'
        ],
        'Ncams': 6,
        'input_size': (128, 128),
        'src_size': (900, 1600),
        'thickness': 5,
        'angle_class': 36,

        # Augmentation
        'resize': (-0.06, 0.11),
        'rot': (-5.4, 5.4),
        'flip': True,
        'crop_h': (0.0, 0.0),
        'resize_test': 0.00,
    }

    grid_config = {
        'x': [-30.0, 30.0, 0.15],
        'y': [-15.0, 15.0, 0.15],
        'z': [-5, 3, 8],
        'depth': [1.0, 60.0, 1.0],
    }

    class_names = [
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ]

    dataset = Nus_online_SegDataset(version='v1.0-mini', data_root='./data/nuscenes-mini', data_config=data_config,
                                    grid_config=grid_config, test_mode=False)
    # input_dict = dataset.get_data_info(1)

    for idx in tqdm(range(dataset.__len__())):
        imgs, trans, rots, intrins, post_trans, post_rots, semantic_masks = dataset.__getitem__(idx)
        print(idx)
