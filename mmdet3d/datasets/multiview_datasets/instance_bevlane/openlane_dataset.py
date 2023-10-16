import json
# from utils.utils import *
import os.path as ops
import glob
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
import sys
sys.path.append('/workspace/lane')
from torch.utils.data import Dataset
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.datasets.multiview_datasets.instance_bevlane.openlane_extract import OpenLaneSegMask
from mmdet3d.datasets.multiview_datasets.image import img_transform, normalize_img
from mmdet.datasets import DATASETS
import cv2
import ipdb


@DATASETS.register_module()
class OpenLane_Dataset(Dataset):
    def __init__(self, images_dir, json_file_dir, data_config=None, grid_config=None,
                 test_mode=False, pipeline=None, CLASSES=None, use_valid_flag=False):
        super(OpenLane_Dataset, self).__init__()
        width_range = (grid_config['x'][0], grid_config['x'][1])
        depth_range = (grid_config['y'][0], grid_config['y'][1])
        self.width_res = grid_config['x'][2]   # 格子宽度大小
        self.depth_res = grid_config['y'][2]   # 格子长度大小
        self.IMG_ORIGIN_W, self.IMG_ORIGIN_H = data_config['src_size'] #1920 * 1280
        self.x_min = grid_config['x'][0]
        self.x_max = grid_config['x'][1]
        self.y_min = grid_config['y'][0]
        self.y_max = grid_config['y'][1]
        self.y_samples = np.linspace(self.y_min, self.y_max, num=100, endpoint=False)

        self.zoff = 1.08
        self.use_valid_flag = use_valid_flag
        self.CLASSES = CLASSES
        self.is_train = not test_mode  # 1 - test_mode
        self.data_config = data_config
        self.grid = self.make_grid()

        self.images_dir = images_dir
        self.json_file_dir = json_file_dir
        self.samples = self.init_dataset_3D(json_file_dir)
        self.mask_extract = OpenLaneSegMask(width_range=width_range,
            depth_range=depth_range,
            width_res=self.width_res,
            depth_res=self.depth_res)
        # gt_lanes, mask_seg,mask_offset,mas_haf,mask_vaf,mask_z

        if pipeline is not None:
            self.pipeline = Compose(pipeline)
        self.flag = np.zeros(len(self.samples), dtype=np.uint8)

    def __len__(self):
        return len(self.samples) #// 20

    def make_grid(self):
        xcoords = torch.linspace(self.x_min, self.x_max, int((self.x_max - self.x_min) / self.width_res))
        ycoords = torch.linspace(self.y_min, self.y_max, int((self.y_max - self.y_min) / self.depth_res))
        yy, xx = torch.meshgrid(ycoords, xcoords)
        return torch.stack([xx, yy, torch.full_like(xx, self.zoff)], dim=-1)

    def prune_3d_lane_by_visibility(self, lane_3d, visibility):
        lane_3d = lane_3d[visibility > 0, ...]
        return lane_3d

    def prune_3d_lane_by_range(self, lane_3d, x_min, x_max):
        # TODO: solve hard coded range later
        lane_3d = lane_3d[np.logical_and(lane_3d[:, 1] > 0, lane_3d[:, 1] < 200), ...]

        # remove lane points out of x range
        lane_3d = lane_3d[np.logical_and(lane_3d[:, 0] > x_min,
                                         lane_3d[:, 0] < x_max), ...]
        return lane_3d

    def data_filter(self, gt_lanes, gt_visibility, gt_category):
        gt_lanes = [self.prune_3d_lane_by_visibility(np.array(gt_lane), np.array(gt_visibility[k])) for k, gt_lane in
                    enumerate(gt_lanes)]
        gt_category = [gt_category[k] for k, lane in enumerate(gt_lanes) if lane.shape[0] > 1]
        gt_lanes = [lane for lane in gt_lanes if lane.shape[0] > 1]

        gt_category = [gt_category[k] for k, lane in enumerate(gt_lanes)
                       if lane[0, 1] < self.y_samples[-1] and lane[-1, 1] > self.y_samples[0]]
        gt_lanes = [lane for lane in gt_lanes if lane[0, 1] < self.y_samples[-1] and lane[-1, 1] > self.y_samples[0]]

        gt_lanes = [self.prune_3d_lane_by_range(np.array(lane), self.x_min, self.x_max) for lane in gt_lanes]

        gt_category = [gt_category[k] for k, lane in enumerate(gt_lanes) if lane.shape[0] > 1]
        gt_lanes = [lane for lane in gt_lanes if lane.shape[0] > 1]

        return gt_category, gt_lanes

    def sample_augmentation(self):
        fW, fH = self.data_config['input_size']
        resize = (fW / self.IMG_ORIGIN_W, fH / self.IMG_ORIGIN_H)
        resize_dims = (fW, fH)
        return resize, resize_dims

    def get_seg_mask(self, gt_lanes, gt_category, gt_visibility):

        seg_mask = torch.from_numpy(np.array([]))
        haf_mask = torch.from_numpy(np.array([]))
        haf_masked = torch.from_numpy(np.array([]))
        vaf_mask = torch.from_numpy(np.array([]))
        mask_offset = torch.from_numpy(np.array([]))

        gt_lanes, mask_seg, mask_offset, mask_haf, mask_vaf, mask_z, haf_masked = self.mask_extract(gt_lanes, gt_category, gt_visibility)

        return gt_lanes, mask_seg, mask_offset, mask_haf, mask_vaf, mask_z, haf_masked

    def perspective(self, matrix, vector):
        """Applies perspective projection to a vector using projection matrix."""
        # tmp = torch.zeros_like(vector)
        # tmp[..., 0] = vector[..., 0]
        # tmp[..., 1] = vector[..., 2]
        # tmp[..., 2] = vector[..., 1]
        # vector = tmp
        vector = vector.unsqueeze(-1)
        homogeneous = torch.matmul(matrix[..., :-1], vector) + matrix[..., [-1]]
        homogeneous = homogeneous.squeeze(-1)
        b = (homogeneous[..., -1] > 0).unsqueeze(-1)
        b = torch.cat((b, b, b), -1)
        b[..., -1] = True
        homogeneous = homogeneous * b.float()
        return homogeneous[..., :-1] / homogeneous[..., [-1]], b.float()
    

    # 1. 添加随机阴影
    def add_random_shadow(self, image):
        # 生成随机阴影遮罩
        height, width = image.shape[:2]
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        random_poly = np.array([[int(width/4), int(height/4)], [int(width/2), int(height/4)], [int(width/2), int(height/2)], [int(width/4), int(height/2)]], dtype=np.int32)
        cv2.fillPoly(mask, [random_poly], 1)

        # 随机调整阴影的透明度
        shadow_opacity = np.random.uniform(0.5, 0.8)
        
        # 将阴影叠加到图像上
        image_with_shadow = cv2.addWeighted(image, 1-shadow_opacity, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), shadow_opacity, 0)
        
        return image_with_shadow

    # 2. 添加高斯模糊
    def add_gaussian_blur(self, image):
        # 生成随机高斯核大小3
        kernel_size = np.random.choice([3, 5, 7])
        
        # 应用高斯模糊
        blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        return blurred_image


    # 3. 模拟强曝光逆光，做intensity数据增强
    def simulate_overexposure(self, image):
        # 生成随机曝光增益
        gain = np.random.uniform(1.2, 1.5)
        
        # 进行强曝光逆光操作
        overexposed_image = cv2.convertScaleAbs(image, alpha=gain, beta=0)
        
        return overexposed_image

    # 根据概率随机选择并应用数据增强操作
    def apply_random_augmentation(self, image):
        augmentations = [
    (self.add_random_shadow, 0.6),    # 添加随机阴影，概率为 0.3
    (self.add_gaussian_blur, 0.1),    # 添加高斯模糊，概率为 0.2
    (self.simulate_overexposure, 0.3) # 模拟强曝光逆光，概率为 0.1
    ]

        # 根据概率随机选择并应用数据增强操作
        # 定义保持原始图像不进行数据增强的概率
        no_aug_prob = 0.95

        # 判断是否进行数据增强操作
        if np.random.choice([False, True], p=[no_aug_prob, 1 - no_aug_prob]):
            return image
        
        for augmentation, prob in augmentations:
            if np.random.choice([False, True], p=[1 - prob, prob]):
                image = augmentation(image)
        
        return image
    
    def get_data_info(self, index, debug=True):
        label_json = self.samples[index]
        # label_file_path = self.json_file_dir + '/' + label_json
        label_file_path = ops.join(self.json_file_dir, label_json)

        imgs = []
        trans = []
        rots = []
        intrins = []
        post_trans = []
        post_rots = []
        extrinsics = []
        undists = []


        with open(label_file_path, 'r') as fr:
            info_dict = json.loads(fr.read())

        image_path = ops.join(self.images_dir, info_dict['file_path'])
        assert ops.exists(image_path), '{:s} not exist'.format(image_path)

        img = Image.open(image_path)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (960, 640))

        extrinsic = np.array(info_dict['extrinsic'])
        intrinsic = np.array(info_dict['intrinsic'])
        gt_lanes_packeds = info_dict['lane_lines']
        resize, resize_dims = self.sample_augmentation()
        img, post_rot, post_tran = img_transform(img, resize, resize_dims)

        R_vg = np.array([[0, 1, 0],
                         [-1, 0, 0],
                         [0, 0, 1]], dtype=float)
        R_gc = np.array([[1, 0, 0],
                         [0, 0, 1],
                         [0, -1, 0]], dtype=float)
        extrinsic[:3, :3] = np.matmul(np.matmul(
            np.matmul(np.linalg.inv(R_vg), extrinsic[:3, :3]),
            R_vg), R_gc)
        extrinsic[0:2, 3] = 0.0

        gt_lanes, gt_visibility, gt_category = [], [], []

        for j, gt_lane_packed in enumerate(gt_lanes_packeds):
            # A GT lane can be either 2D or 3D
            # if a GT lane is 3D, the height is intact from 3D GT, so keep it intact here too
            lane = np.array(gt_lane_packed['xyz'])
            lane_visibility = np.array(gt_lane_packed['visibility'])

            lane = np.vstack((lane, np.ones((1, lane.shape[1]))))
            cam_representation = np.linalg.inv(
                np.array([[0, 0, 1, 0],
                          [-1, 0, 0, 0],
                          [0, -1, 0, 0],
                          [0, 0, 0, 1]], dtype=float))
            lane = np.matmul(extrinsic, np.matmul(cam_representation, lane))
            lane = lane[0:3, :].T

            gt_lanes.append(lane)
            gt_visibility.append(lane_visibility)
            gt_category.append(gt_lane_packed['category'])         # 这里gt_lanes是转到3D，然后vis再转到像素
        gt_category, gt_lanes = self.data_filter(gt_lanes, gt_visibility, gt_category)
        img = np.array(img)
        img = self.apply_random_augmentation(img)
        img = normalize_img(img)
        trans.append(torch.Tensor(extrinsic[:3, 3]))
        rots.append(torch.Tensor(extrinsic[:3, :3]))
        extrinsics.append(torch.tensor(extrinsic).float())
        intrins.append(torch.cat((torch.Tensor(intrinsic), torch.zeros((3, 1))), dim=1).float())
        post_trans.append(post_tran)
        post_rots.append(post_rot)
        imgs.append(img)
        undists.append(torch.zeros(7))

        imgs, trans, rots, intrins, post_trans, post_rots, undists, extrinsics = torch.stack(imgs), torch.stack(trans), torch.stack(rots), torch.stack(intrins), torch.stack(
            post_trans), torch.stack(post_rots), torch.stack(undists), torch.stack(extrinsics)
        extrinsics = torch.linalg.inv(extrinsics)# change cam2glob to glob2cam

        gt_lanes, mask_seg, mask_offset, mask_haf, mask_vaf, mask_z, haf_masked = self.get_seg_mask(gt_lanes, gt_category, gt_visibility)

        mask_seg[mask_seg > 0] = 1
        mask_seg = torch.from_numpy(np.array(mask_seg, dtype=np.uint8))
        mask_haf = torch.from_numpy(np.array(mask_haf)).float()
        haf_masked = torch.from_numpy(np.array(haf_masked)).float()
        mask_vaf = np.transpose(mask_vaf, (2, 0, 1))
        mask_vaf = torch.from_numpy(np.array(mask_vaf)).float()
        mask_offset = torch.from_numpy(np.array(mask_offset)).float()
        mask_offset = mask_offset.permute(2, 0, 1)
        mask_z = torch.from_numpy(np.array(mask_z)).float()
        mask_z = mask_z.permute(2, 0, 1)

        if debug:
            visu_path = './vis_pic'
            calib = np.matmul(intrins, extrinsics)
            for gt_lane in gt_lanes:
                gt_lane = torch.tensor(gt_lane).float()
                img_points, _ = self.perspective(calib, gt_lane)

                post_img_points = []
                for img_point in img_points:
                    img_point = torch.matmul(post_rots[0, :2, :2], img_point) + post_trans[0, :2]
                    post_img_points.append(img_point.detach().cpu().numpy())
                post_img_points = np.array(post_img_points)
                x_2d, y_2d = post_img_points[:, 0].astype(np.int32), post_img_points[:, 1].astype(np.int32)
                for k in range(1, img_points.shape[0]):
                    image = cv2.line(image, (x_2d[k - 1], y_2d[k - 1]),
                                     (x_2d[k], y_2d[k]), (0, 0, 255), 4)
            # cv2.imwrite(visu_path + "/img.jpg", image)

        input_dict = dict(
            imgs=imgs,
            trans=trans,
            rots=rots,
            extrinsics=extrinsics,
            intrins=intrins,
            undists=undists,
            post_trans=post_trans,
            post_rots=post_rots,
            semantic_masks=mask_seg,
            mask_offset=mask_offset,
            mask_haf=mask_haf,
            mask_vaf=mask_vaf,
            mask_z=mask_z,
            gt_lanes=gt_lanes,
            grid=self.grid,
            drop_idx=torch.tensor([]),
            file_path=info_dict['file_path'],
            haf_masked=haf_masked,
        )

        return input_dict

    def init_dataset_3D(self, json_file_dir):
        filter_samples = []
        samples = glob.glob(json_file_dir + '**/*.json', recursive=True)
        for i, sample in enumerate(samples):
            label_file_path = ops.join(self.json_file_dir, sample)
            with open(label_file_path, 'r') as fr:
                info_dict = json.loads(fr.read())
            image_path = ops.join(self.images_dir, info_dict['file_path'])
            if not ops.exists(image_path):
                # print('{:s} not exist'.format(image_path))
                continue
            # if i < 1014:
            #     continue
            
            filter_samples.append(sample)
            #if len(filter_samples) > 410:
            #   break
            # print("image_path:", image_path)

        # return samples
        return filter_samples

    def __getitem__(self, idx):
        input_dict = self.get_data_info(idx, debug=False)
        data = self.pipeline(input_dict)
        return data
        # return input_dict

if __name__ == '__main__':
    data_config = {
        'cams': [],
        'Ncams': 1,
        'input_size': (960, 640),
        'src_size': (1920, 1280),
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
        'x': [-10.0, 10.0, 0.15],
        'y': [3.0, 103.0, 0.5],
        'z': [-5, 3, 8],
        'depth': [1.0, 60.0, 1.0],
    }

    images_dir = '/workspace/openlane_all/images'
    json_file_dir = '/workspace/openlane_all/lane3d_300/training/'

    dataset = OpenLane_Dataset(images_dir, json_file_dir, data_config=data_config, grid_config=grid_config,
                 test_mode=False, pipeline=None, CLASSES=None, use_valid_flag=True)

    for idx in tqdm(range(dataset.__len__())):
        input_dict = dataset.__getitem__(idx)
        print(idx)
