import json
# from utils.utils import *
import os.path as ops
import glob
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.datasets.multiview_datasets.instance_bevlane.openlane_extract import OpenLaneSegMask
from mmdet3d.datasets.multiview_datasets.image import img_transform, normalize_img
from mmdet.datasets import DATASETS
import cv2
import math
from mmdet3d.datasets.multiview_datasets.anchor_bevlane import utils
from mmdet3d.datasets.multiview_datasets.instance_bevlane.mesh_grid import MeshGrid
# from mmdet3d.datasets.multiview_datasets.instance_bevlane.openlane_extract import OpenLaneSegMask



@DATASETS.register_module()
class OpenLane_Anchor_Dataset(Dataset):
    def __init__(self, images_dir, json_file_dir, S=96, data_config=None, grid_config=None, use_gen_anchor=False, use_seg_anchor=False,
                 test_mode=False, pipeline=None, CLASSES=None, use_valid_flag=False):
        super(OpenLane_Anchor_Dataset, self).__init__()
        width_range = (grid_config['x'][0], grid_config['x'][1])
        depth_range = (grid_config['y'][0], grid_config['y'][1])
        self.width_res = grid_config['x'][2]
        self.depth_res = grid_config['y'][2]
        self.IMG_ORIGIN_W, self.IMG_ORIGIN_H = data_config['src_size'] #1920 * 1280
        self.x_min = grid_config['x'][0]
        self.x_max = grid_config['x'][1]
        self.y_min = grid_config['y'][0]
        self.y_max = grid_config['y'][1]

        self.bev_w = self.x_max - self.x_min
        self.bev_h = self.y_max - self.y_min
        self.y_samples = np.linspace(self.y_min, self.y_max, num=int(self.bev_h), endpoint=False)

        self.zoff = 1.08
        self.use_valid_flag = use_valid_flag
        self.CLASSES = CLASSES
        self.is_train = not test_mode  # 1 - test_mode
        self.data_config = data_config
        self.grid = self.make_grid()
        self.mesh = MeshGrid(width_range, depth_range, self.width_res, self.depth_res)
        self.mask_extract = OpenLaneSegMask(width_range=width_range,
                                            depth_range=depth_range,
                                            width_res=self.width_res,
                                            depth_res=self.depth_res)

        self.images_dir = images_dir
        self.json_file_dir = json_file_dir
        self.samples = self.init_dataset_3D(json_file_dir)

        self.n_strips = S - 1
        self.n_offsets = S
        # self.anchor_y_steps = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 40, 50, 60, 80, 100])
        self.anchor_y_steps = np.linspace(self.y_min, self.y_max, num=int(self.bev_h / 2.), endpoint=False)#[::-1]
        # self.anchor_y_steps = torch.linspace(1, 0, steps=self.n_offsets, dtype=torch.float32)
        self.num_y_steps = len(self.anchor_y_steps)

        self.max_lanes = 15
        self.num_classes = 1

        self.use_gen_anchor = use_gen_anchor
        self.use_seg_anchor = use_seg_anchor
        # num = (self.x_max - self.x_min) / self.width_res
        if use_gen_anchor:
            self.anchor_x_steps = np.linspace(self.x_min, self.x_max,
                                              num=int(self.bev_w / 0.4) + 1, endpoint=False)
            self.num_x_steps = len(self.anchor_x_steps)
            self.anchor_dim = 3 * self.num_y_steps + self.num_classes

            #anchor config
            self.y_ref = 5.
            self.ref_id = np.argmin(np.abs(self.num_y_steps - self.y_ref))

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

    def perspective(self, matrix, vector):
        """Applies perspective projection to a vector using projection matrix."""
        vector = vector.unsqueeze(-1)
        homogeneous = torch.matmul(matrix[..., :-1], vector) + matrix[..., [-1]]
        homogeneous = homogeneous.squeeze(-1)
        b = (homogeneous[..., -1] > 0).unsqueeze(-1)
        b = torch.cat((b, b, b), -1)
        b[..., -1] = True
        homogeneous = homogeneous * b.float()
        return homogeneous[..., :-1] / homogeneous[..., [-1]], b.float()

    def linear_interpolation(self, x1, y1, x2, y2):
        num = round((y2 - y1) / self.depth_res)
        if num <= 0:
            return []
        xx = np.linspace(x1, x2, num, endpoint=False)
        yy = np.linspace(y1, y2, num, endpoint=False)
        out = []
        for x, y in zip(xx, yy):
            out.append([x, y])
        return out

    def linear_interpolation_3d(self, x1, y1, z1, x2, y2, z2):
        num = round((y2 - y1) / self.depth_res)
        out = []
        if x1 == x2 and y1 == y2:
            out.append([x1, y1, z1])
            return out
        if num <= 0:
            return []
        xx = np.linspace(x1, x2, num, endpoint=False)
        yy = np.linspace(y1, y2, num, endpoint=False)
        zz = np.linspace(z1, z2, num, endpoint=False)

        for x, y, z in zip(xx, yy, zz):
            out.append([x, y, z])
        return out

    def coords_interpolation(self, bev_coords, space="3d"):
        out = []
        if len(bev_coords) < 1:
            return out
        last_poi = bev_coords[0]
        if space == "2d":
            for poi in bev_coords:
                re = self.linear_interpolation(last_poi[0], last_poi[1], poi[0], poi[1])
                last_poi = poi
                out.extend(re)
        else:
            for poi in bev_coords:
                re = self.linear_interpolation_3d(last_poi[0], last_poi[1], last_poi[2], poi[0], poi[1], poi[2])
                last_poi = poi
                out.extend(re)
        return out

    def lane_fit(self, gt_lanes, poly_order=3, sample_step=1, interp=False):
        fit_lanes = []
        for i, gt_lane in enumerate(gt_lanes):

            xs_gt = gt_lane[:, 0]
            ys_gt = gt_lane[:, 1]
            zs_gt = gt_lane[:, 2]

            poly_params_yx = np.polyfit(ys_gt, xs_gt, deg=poly_order)
            poly_params_yz = np.polyfit(ys_gt, zs_gt, deg=poly_order)

            y_min, y_max = np.min(ys_gt), np.max(ys_gt)
            y_min = math.floor(y_min)
            y_max = math.ceil(y_max)
            y_sample = np.array(range(y_min, y_max, sample_step))

            ys_out = np.array(y_sample, dtype=np.float32)

            xs_out = np.polyval(poly_params_yx, ys_out)
            zs_out = np.polyval(poly_params_yz, ys_out)

            fit_lane = np.zeros((len(xs_out), 3))
            fit_lane[:, 0] = xs_out
            fit_lane[:, 1] = ys_out
            fit_lane[:, 2] = zs_out

            mask_idex = (self.x_min <= fit_lane[:, 0]) & (fit_lane[:, 0] <= self.x_max) & (
                        self.y_min <= fit_lane[:, 1]) & (fit_lane[:, 1] <= self.y_max)

            fit_lane = fit_lane[mask_idex]

            if interp:
                fit_lane = self.coords_interpolation(fit_lane)
                fit_lane = np.array(fit_lane)

            if fit_lane.shape[0] <= 2:
                continue

            fit_lanes.append(fit_lane)

        return fit_lanes

    def transform_annotation(self, gt_lanes):

        #filter outside point
        # gt_lanes = self.lane_fit(gt_lanes=gt_lanes, interp=True)

        # new_ge_lanes = []
        #
        # for id, line in enumerate(gt_lanes):
        #     bev_coords = line
        #     if len(bev_coords) < 2:
        #         continue
        #     new_ge_lane = []
        #     for num, pos_imu in enumerate(bev_coords):
        #         if self.mesh.is_pos_outside(pos_imu[0], pos_imu[1]):
        #             continue
        #         u, v = self.mesh.get_index_by_pos(pos_imu[0], pos_imu[1])
        #         z = pos_imu[2]
        #         if self.mesh.is_index_outside(u, v):
        #             continue
        #         new_ge_lane.append([u, v, z])
        #     new_ge_lanes.append(np.array(new_ge_lane))
        #........................................................
        # create tranformed annotations
        # 2 scores, 1 start_y, 1 start_x, 1 start_z, 1 len, S coordinates; score[0] = negative prob, score[1] = positive prob
        lanes = np.ones((self.max_lanes, 2 + 1 + 1 + 1 + 3 * self.n_offsets),
                        dtype=np.float32)# * -1e5
        lanes[:, 0] = 1
        lanes[:, 1] = 0

        frame_x_values, frame_z_values, frame_visibility_vectors = [], [], []
        for lane_idx, gt_lane in enumerate(gt_lanes):
            x_values, z_values, visibility_vec = utils.resample_laneline_in_y(gt_lane, self.anchor_y_steps, self.x_min, out_vis=True)
            # frame_x_values.append(x_values)
            # frame_z_values.append(z_values)
            # frame_visibility_vectors.append(visibility_vec)

            visibility_indexs = np.where(visibility_vec != 0)
            try:
                lanes[lane_idx, 0] = 0
                lanes[lane_idx, 1] = 1
                # lanes[lane_idx, 2] = self.y_samples[visibility_indexs][0]#start y
                lanes[lane_idx, 2] = self.anchor_y_steps[visibility_indexs][0]#start y
                lanes[lane_idx, 3] = np.around(x_values[visibility_indexs][0], decimals=4) #start x
                lanes[lane_idx, 4] = np.around(z_values[visibility_indexs][0], decimals=5) #start z
                lanes[lane_idx, 5:5 + 3 * len(x_values)] = np.hstack((x_values, z_values, visibility_vec))
            except:
                continue

        new_lanes = lanes
        return new_lanes

    # for gen net
    def gen_transform_annotation(self, gt_lanes):
        # lanes = np.ones((self.max_lanes, 2 + 1 + 1 + 1 + 3 * self.n_offsets),
        #                 dtype=np.float32)# * -1e5
        # lanes[:, 0] = 1
        # lanes[:, 1] = 0

        # frame_x_values, frame_z_values, frame_visibility_vectors = [], [], []

        gt_anchor = np.zeros([self.num_x_steps, self.num_classes, self.anchor_dim], dtype=np.float32)

        ass_ids = []

        for lane_idx, gt_lane in enumerate(gt_lanes):
            x_values, z_values, visibility_vec = utils.resample_laneline_in_y(gt_lane, self.anchor_y_steps, self.x_min, out_vis=True)
            # frame_x_values.append(x_values)
            # frame_z_values.append(z_values)
            # frame_visibility_vectors.append(visibility_vec)

            visibility_indexs = np.where(visibility_vec != 0)
            num_visib = len(list(visibility_indexs)[0])
            mid_index = int(num_visib / 2)

            ass_id = np.argmin((self.anchor_x_steps - x_values[mid_index]) ** 2)
            ass_ids.append(ass_id)

            x_off_values = x_values - self.anchor_x_steps[ass_id]

            gt_anchor[ass_id, 0, 0: self.num_y_steps] = x_off_values
            gt_anchor[ass_id, 0, self.num_y_steps: 2 * self.num_y_steps] = z_values
            gt_anchor[ass_id, 0, 2 * self.num_y_steps: 3 * self.num_y_steps] = visibility_vec
            gt_anchor[ass_id, 0, -1] = 1
            # except:
            #     continue

        return gt_anchor, ass_ids


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

        #'/home/slj/Documents/workspace/mmdet3d/data/openlane/example/image/validation/segment-260994483494315994_2797_545_2817_545_with_camera_labels/150723473494688800.jpg'
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
            gt_category.append(gt_lane_packed['category'])

        gt_category, origin_lanes = self.data_filter(gt_lanes, gt_visibility, gt_category)

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

        origin_lanes = self.lane_fit(gt_lanes=origin_lanes, interp=False)
        gt_lanes = self.transform_annotation(origin_lanes)

        gt_anchors = None
        ass_ids = None

        if self.use_gen_anchor:
            gt_anchors, ass_ids = self.gen_transform_annotation(origin_lanes)
        if self.use_seg_anchor:
            mask_seg, mask_offset, mask_z, mask_cls, gt_lanes = self.mask_extract.get_anchorlane_mask(origin_lanes, gt_category, gt_visibility)
        # else:
        #     gt_anchors = None
        #     ass_ids = None

        input_dict = dict(
            imgs=imgs,
            trans=trans,
            rots=rots,
            extrinsics=extrinsics,
            intrins=intrins,
            undists=undists,
            post_trans=post_trans,
            post_rots=post_rots,
            gt_lanes=gt_lanes,
            origin_lanes=origin_lanes,
            gt_anchors=gt_anchors,
            grid=self.grid,
            drop_idx=torch.tensor([]),
            file_path=info_dict['file_path'],
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
            filter_samples.append(sample)
            if len(filter_samples) > 400:
                break
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
    # images_dir = '/home/slj/Documents/workspace/mmdet3d/data/openlane/example/image'
    # json_file_dir = '/home/slj/Documents/workspace/mmdet3d/data/openlane/example/annotations/segment-260994483494315994_2797_545_2817_545_with_camera_labels'

    images_dir = '/workspace/openlane_all/images'
    json_file_dir = '/workspace/openlane_all/lane3d_300/training/'

    dataset = OpenLane_Anchor_Dataset(images_dir, json_file_dir, data_config=data_config, grid_config=grid_config,
                 test_mode=False, pipeline=None, CLASSES=None, use_valid_flag=True)

    for idx in tqdm(range(dataset.__len__())):
        input_dict = dataset.__getitem__(idx)
        print(idx)
