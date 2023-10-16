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
from matplotlib.pyplot import MultipleLocator
import matplotlib.pyplot as plt
import os



class OpenLane_Dataset(Dataset):
    def __init__(self, images_dir, json_file_dir, data_config=None, grid_config=None,
                 test_mode=False, pipeline=None, CLASSES=None, use_valid_flag=False):
        super(OpenLane_Dataset, self).__init__()
        width_range = (grid_config['x'][0], grid_config['x'][1])
        depth_range = (grid_config['y'][0], grid_config['y'][1])
        self.width_res = grid_config['x'][2]
        self.depth_res = grid_config['y'][2]
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
        vaf_mask = torch.from_numpy(np.array([]))
        mask_offset = torch.from_numpy(np.array([]))

        gt_lanes, mask_seg, mask_offset, mask_haf, mask_vaf, mask_z = self.mask_extract(gt_lanes, gt_category, gt_visibility)

        return gt_lanes, mask_seg, mask_offset, mask_haf, mask_vaf, mask_z

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

    def projective_transformation(self, Matrix, x, y, z):
        ones = np.ones((1, len(z)))
        coordinates = np.vstack((x, y, z, ones))
        trans = np.matmul(Matrix, coordinates)

        x_vals = trans[0, :] / trans[2, :]
        y_vals = trans[1, :] / trans[2, :]
        return x_vals, y_vals

    def unprojective_transformation(self, Matrix, x, y):
        # ones = np.ones((1, len(z)))
        # coordinates = np.vstack((x, y, z, ones))
        trans = np.matmul(Matrix, coordinates)

        x_vals = trans[0, :] / trans[2, :]
        y_vals = trans[1, :] / trans[2, :]
        return x_vals, y_vals

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
        fit_lane_nearsts = {}
        zs_out_preds = {}

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

        for i, gt_lane in enumerate(gt_lanes):

            ys_gt = gt_lane[:, 1]
            origin_y_max = np.max(ys_gt)

            mask_idex = gt_lane[:, 1] <= 60.
            if sum(mask_idex) <= 2:
                continue
            nearst_lane = gt_lane[mask_idex]
            nearst_xs_gt = nearst_lane[:, 0]
            nearst_ys_gt = nearst_lane[:, 1]
            nearst_zs_gt = nearst_lane[:, 2]

            poly_params_yx = np.polyfit(nearst_ys_gt, nearst_xs_gt, deg=poly_order)
            poly_params_yz = np.polyfit(nearst_ys_gt, nearst_zs_gt, deg=poly_order)

            y_min, y_max = np.min(nearst_ys_gt), np.max(nearst_ys_gt)
            y_min = math.floor(y_min)
            y_max = math.ceil(y_max)
            y_sample = np.array(range(y_min, y_max, sample_step))
            ys_out = np.array(y_sample, dtype=np.float32)

            xs_out = np.polyval(poly_params_yx, ys_out)
            zs_out = np.polyval(poly_params_yz, ys_out)

            if origin_y_max >= 60:
                origin_max = max(origin_y_max, 95)
                pred_y_sample = np.array(range(60, int(origin_max) + 1, sample_step))
                pred_ys_out = np.array(pred_y_sample, dtype=np.float32)
                zs_out_pred = np.polyval(poly_params_yz, pred_ys_out)

                # zs_out_preds[i] = zs_out_pred
                # zs_out_preds[i] = np.full([len(pred_y_sample)], zs_out_pred[0])
                # zs_out_preds[i] = np.array([np.median(zs_out)] * len(pred_y_sample))
                zs_out_preds[i] = np.array([np.mean(zs_out)] * len(pred_y_sample))
                # zs_out_preds[i] = np.array([zs_out_pred[0]] * len(pred_y_sample))
            fit_lane = np.zeros((len(xs_out), 3), dtype=np.float32)
            fit_lane[:, 0] = xs_out
            fit_lane[:, 1] = ys_out
            fit_lane[:, 2] = zs_out

            mask_idex = (self.x_min <= fit_lane[:, 0]) & (fit_lane[:, 0] <= self.x_max) & (
                    self.y_min <= fit_lane[:, 1]) & (fit_lane[:, 1] <= self.y_max)

            fit_lane = fit_lane[mask_idex]

            # .............................................
            if interp:
                fit_lane = self.coords_interpolation(fit_lane)
                fit_lane = np.array(fit_lane)

            if fit_lane.shape[0] <= 2:
                continue
            # fit_lane_nearsts.append(fit_lane)
            fit_lane_nearsts[i] = fit_lane

        pred_fit_lanes = []

        for i, fit_lane in enumerate(fit_lanes):
            pred_lane = np.zeros_like(fit_lane)
            pred_lane[:, 0] = fit_lane[:, 0]
            pred_lane[:, 1] = fit_lane[:, 1]
            pred_lane[:, 2] = fit_lane[:, 2]

            if i in zs_out_preds.keys():
                # cat_gt = fit_lane_nearsts[i][:, 2].astype('float32')
                mask_idex = pred_lane[:, 1] < 60.
                cat_gt = pred_lane[mask_idex][:, 2]
                cat_pred = zs_out_preds[i].astype('float32')
                cat_gt = cat_gt.tolist()
                cat_pred = cat_pred.tolist()

                exe_2 = cat_gt + cat_pred
                exe_3 = np.array(exe_2)
                # fit_lane[:, 2] = exe_3[:len(fit_lane[:, 2])]
                pred_lane[:, 2] = exe_3[:len(pred_lane[:, 2])]
                # try:
                #     fit_lane[:, 2] = exe_3
                # except:
                #     print(i)
                # exe = fit_lane[:, 2]
                pred_lane = np.array(pred_lane)

                # fit_lane[i, 2] = np.stack((cat_gt, cat_pred), axis=1)
                # fit_lane[:, 2] = np.concatenate((cat_gt, cat_pred), axis=1)
            pred_fit_lanes.append(pred_lane)


        return fit_lanes, pred_fit_lanes, zs_out_preds

    def draw_lane_imu_z(self, gt_z, pred_z):
        filepath = "mmdet3d/core/hook/imu_z_compare.png"
        # print("filepath: " + filepath)
        fig_2d = plt.figure(figsize=(6.4, 6.4))
        plt.grid(linestyle='--', color='y', linewidth=0.5)
        x_major_locator = MultipleLocator(1)
        y_major_locator = MultipleLocator(4)
        ax = plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)
        ax.yaxis.set_major_locator(y_major_locator)
        # plt.xticks(range(-19, 19, 1))

        for line_id, line in enumerate(gt_z):
            x_data = []
            y_data = []
            for poi in line:
                x_data.append(poi[0])
                y_data.append(poi[1])
            plt.plot(x_data, y_data, linestyle='-', color='b', linewidth=1)

        for line_id, line in enumerate(pred_z):
            x_data = []
            y_data = []
            if len(line['coordinates']) < 2:
                continue
            for poi in line['coordinates']:
                x_data.append(poi[0])
                y_data.append(poi[1])
            plt.plot(x_data, y_data, linestyle='-', color='r', linewidth=1)

        plt.xlabel('X: offset, res=0.2')
        plt.ylabel('Y: distance')
        plt.title("Only show X_Y : GroundTruth: Blue __;   Prediction: Red __")
        plt.savefig(filepath)
        plt.cla()
        plt.close()
        return filepath


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

        gt_category, gt_lanes = self.data_filter(gt_lanes, gt_visibility, gt_category)

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

        fit_lanes, pred_fit_lanes, zs_out_preds = self.lane_fit(gt_lanes, interp=False)

        calib = np.matmul(intrins, extrinsics)


        for fit_lane in fit_lanes:
            fit_lane = torch.tensor(fit_lane).float()
            img_points, _ = self.perspective(calib, fit_lane)

            post_img_points = []
            for img_point in img_points:
                img_point = torch.matmul(post_rots[0, :2, :2], img_point) + post_trans[0, :2]
                post_img_points.append(img_point.detach().cpu().numpy())
            post_img_points = np.array(post_img_points)
            x_2d, y_2d = post_img_points[:, 0].astype(np.int32), post_img_points[:, 1].astype(np.int32)

            gt_img_points = torch.zeros([len(x_2d), 2])
            gt_img_points[:, 0] = x_2d
            gt_img_points[:, 1] = y_2d

            for k in range(1, img_points.shape[0]):
                image = cv2.line(image, (x_2d[k - 1], y_2d[k - 1]),
                                 (x_2d[k], y_2d[k]), (0, 0, 255), 3)

        for pred_fit_lane in pred_fit_lanes:
            pred_fit_lane = torch.tensor(pred_fit_lane).float()
            img_points, _ = self.perspective(calib, pred_fit_lane)

            post_img_points = []
            for img_point in img_points:
                img_point = torch.matmul(post_rots[0, :2, :2], img_point) + post_trans[0, :2]
                post_img_points.append(img_point.detach().cpu().numpy())
            post_img_points = np.array(post_img_points)
            x_2d, y_2d = post_img_points[:, 0].astype(np.int32), post_img_points[:, 1].astype(np.int32)

            for k in range(1, img_points.shape[0]):
                image = cv2.line(image, (x_2d[k - 1], y_2d[k - 1]),
                                 (x_2d[k], y_2d[k]), (255, 0, 0), 2)

        calib = torch.linalg.inv(calib)# change glob2cam to cam2glob

        pred_3d = torch.matmul(calib, gt_img_points)



        visu_path = './test_for_labels'
        # img_path = os.path.join(visu_path, *img_root)
        # print("img_path:", img_path)
        if not os.path.exists(visu_path):
            os.makedirs(visu_path)

        cv2.imwrite(visu_path + '/' + str(index) + "_img.jpg", image)


        input_dict = dict()
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
            if len(filter_samples) > 41000:
                break
            # print("image_path:", image_path)

        # return samples
        return filter_samples

    def __getitem__(self, idx):
        input_dict = self.get_data_info(idx, debug=False)
        # data = self.pipeline(input_dict)
        # return data
        return input_dict

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

    images_dir = '/home/slj/data/openlane/openlane_all/images'
    json_file_dir = '/home/slj/data/openlane/openlane_all/lane3d_300/training/'

    dataset = OpenLane_Dataset(images_dir, json_file_dir, data_config=data_config, grid_config=grid_config,
                 test_mode=False, pipeline=None, CLASSES=None, use_valid_flag=True)

    for idx in tqdm(range(dataset.__len__())):
        input_dict = dataset.__getitem__(idx)
        print(idx)
