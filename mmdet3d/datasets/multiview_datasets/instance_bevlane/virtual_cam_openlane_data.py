import copy
import json
import os

import cv2
import numpy as np
import math
import torch
from scipy.interpolate import interp1d
from torch.utils.data import Dataset
from ..coord_util import ego2image, IPM2ego_matrix
from ..standard_camera_cpu import Standard_camera
from mmdet.datasets import DATASETS
from mmdet3d.datasets.pipelines import Compose
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os.path as ops
import glob
from PIL import Image
from mmdet3d.datasets.multiview_datasets.instance_bevlane.generate_affinity_field import GenerateHAFAndVAF
from mmdet3d.datasets.multiview_datasets.image import img_transform, normalize_img
from mmdet3d.datasets.multiview_datasets.instance_bevlane.openlane_extract import OpenLaneSegMask
from mmdet3d.datasets.multiview_datasets.instance_bevlane.mesh_grid import MeshGrid


@DATASETS.register_module()
class Virtual_Cam_OpenLane_Dataset(Dataset):
    def __init__(self, image_paths,
                   gt_paths,
                   x_range,
                   y_range,
                meter_per_pixel,
                  input_shape=None,
                 output_2d_shape=None,
                 virtual_camera_config=None,
                 pipeline=None,
                 test_mode=False,
                 CLASSES=None,
                 use_valid_flag=None,
                 ):

        self.x_range = x_range
        self.y_range = y_range
        self.meter_per_pixel = meter_per_pixel
        self.image_paths = image_paths
        self.gt_paths = gt_paths

        self.width_res = meter_per_pixel
        self.depth_res = meter_per_pixel

        self.lane3d_thick = 1
        self.lane2d_thick = 3
        self.lane_length_threshold = 3  #

        self.samples = self.init_dataset_3D(self.gt_paths)

        ''' virtual camera paramter'''
        self.use_virtual_camera = virtual_camera_config['use_virtual_camera']
        self.vc_intrinsic = virtual_camera_config['vc_intrinsic']
        self.vc_extrinsics = virtual_camera_config['vc_extrinsics']
        self.vc_image_shape = virtual_camera_config['vc_image_shape']

        ''' transform loader '''
        # self.output2d_size = output_2d_shape
        self.ipm_h, self.ipm_w = int((self.x_range[1] - self.x_range[0]) / self.meter_per_pixel), int(
            (self.y_range[1] - self.y_range[0]) / self.meter_per_pixel)

        self.CLASSES = CLASSES

        if pipeline is not None:
            self.pipeline = Compose(pipeline)
        self.use_valid_flag = use_valid_flag

        # self.flag = np.zeros(len(self.cnt_list), dtype=np.uint8)
        self.flag = np.zeros(len(self.samples), dtype=np.uint8)

        self.test_mode = test_mode

        self.trans_image = A.Compose([
            A.Resize(height=input_shape[0], width=input_shape[1]),
            A.MotionBlur(p=0.2),
            A.RandomBrightnessContrast(),
            A.ColorJitter(p=0.1),
            A.Normalize(),
            ToTensorV2()]
        )

        self.output2d_size = output_2d_shape
        self.input_shape = input_shape

        self.engine_af = GenerateHAFAndVAF()

        self.mask_extract = OpenLaneSegMask(width_range=self.y_range,
            depth_range=self.x_range,
            width_res=self.meter_per_pixel,
            depth_res=self.meter_per_pixel)

        # self.mesh = MeshGrid(width_range, depth_range, width_res, depth_res)

    def get_y_offset_and_z(self, res_d):
        '''
        :param res_d: res_d
        :param instance_seg:
        :return:
        '''

        def caculate_distance(base_points, lane_points, lane_z, lane_points_set):
            '''
            :param base_points: base_points n * 2
            :param lane_points:
            :return:
            '''
            condition = np.where(
                (lane_points_set[0] == int(base_points[0])) & (lane_points_set[1] == int(base_points[1])))
            if len(condition[0]) == 0:
                return None, None
            lane_points_selected = lane_points.T[condition]  #
            lane_z_selected = lane_z.T[condition]
            offset_y = np.mean(lane_points_selected[:, 1]) - base_points[1]
            z = np.mean(lane_z_selected[:, 1])
            return offset_y, z

        # instance_seg = np.zeros((450, 120), dtype=np.uint8)
        res_lane_points = {}
        res_lane_points_z = {}
        res_lane_points_bin = {}
        res_lane_points_set = {}
        # for idx, res in enumerate(res_d):
        for idx in res_d:
            ipm_points_ = np.array(res_d[idx])
            ipm_points = ipm_points_.T[np.where((ipm_points_[1] >= 0) & (ipm_points_[1] < self.ipm_h))].T  #
            if len(ipm_points[0]) <= 1:
                continue
            x, y, z = ipm_points[1], ipm_points[0], ipm_points[2]
            base_points = np.linspace(x.min(), x.max(),
                                      int((x.max() - x.min()) // 0.05))
            base_points_bin = np.linspace(int(x.min()), int(x.max()),
                                          int(int(x.max()) - int(x.min())) + 1)  # .astype(np.int)
            if len(x) <= 1:
                continue
            elif len(x) <= 2:
                function1 = interp1d(x, y, kind='linear',
                                     fill_value="extrapolate")  #
                function2 = interp1d(x, z, kind='linear')
            elif len(x) <= 3:
                function1 = interp1d(x, y, kind='quadratic', fill_value="extrapolate")
                function2 = interp1d(x, z, kind='quadratic')
            else:
                function1 = interp1d(x, y, kind='cubic', fill_value="extrapolate")
                function2 = interp1d(x, z, kind='cubic')
            y_points = function1(base_points)
            y_points_bin = function1(base_points_bin)
            z_points = function2(base_points)
            res_lane_points[idx] = np.array([base_points, y_points])  #
            res_lane_points_z[idx] = np.array([base_points, z_points])
            res_lane_points_bin[idx] = np.array([base_points_bin, y_points_bin]).astype(np.int)
            res_lane_points_set[idx] = np.array([base_points, y_points]).astype(
                np.int)

        offset_map = np.zeros((self.ipm_h, self.ipm_w))
        z_map = np.zeros((self.ipm_h, self.ipm_w))
        ipm_image = np.zeros((self.ipm_h, self.ipm_w))
        for idx in res_lane_points_bin:
            lane_bin = res_lane_points_bin[idx].T
        # for idx in res_d:
        #     lane_bin = res_d[idx].T
            for point in lane_bin:
                row, col = point[0], point[1]
                if not (0 < row < self.ipm_h and 0 < col < self.ipm_w):  #
                    continue
                ipm_image[row, col] = idx
                center = np.array([row, col])
                offset_y, z = caculate_distance(center, res_lane_points[idx], res_lane_points_z[idx],
                                                res_lane_points_set[idx])  #
                if offset_y is None:  #
                    ipm_image[row, col] = 0
                    continue
                if offset_y > 1:
                    offset_y = 1
                if offset_y < 0:
                    offset_y = 0
                offset_map[row][col] = offset_y
                z_map[row][col] = z

        return ipm_image, offset_map, z_map

    def get_laneline_offset(self, gt_lanes, draw_type='cv2line'):
        """
        gt_points fit
        """
        # gt_lanes = self.lane_fit(gt_lanes)
        """
        get lanes keypoints offset to anchor and gt_mask
        """
        mask_seg = np.zeros((self.bev_height, self.bev_width, 3), dtype=np.uint8)
        mask_offset = np.zeros((self.bev_height, self.bev_width, 2))
        mask_z = np.zeros((self.bev_height, self.bev_width, 1))


        ins_id = 0
        for id, line in enumerate(gt_lanes):
            bev_coords = line
            ins_id += 1
            # category = gt_category[id]
            # gt_visibility = gt_visibility[id]

            if len(bev_coords) < 2:
                continue
            last_poi_x = 0
            last_poi_y = 0

            for num, pos_imu in enumerate(bev_coords):
                if self.mesh.is_pos_outside(pos_imu[0], pos_imu[1]):
                    continue
                u, v = self.mesh.get_index_by_pos(pos_imu[0], pos_imu[1])
                z = pos_imu[2]
                if self.mesh.is_index_outside(u, v):
                    continue
                offset_x, offset_y = self.mesh.get_offset_with_cell_ld(pos_imu[0], pos_imu[1])

                if draw_type == "cv2circle":
                    ins_color = self.color_map[ins_id % self.color_num]
                    rgb_value = ins_id * 10
                    color = (rgb_value, rgb_value, rgb_value)
                    cv2.circle(mask_seg,
                               (u, v),
                               radius=1,
                               color=color,
                               thickness=self.laneline_width)
                elif draw_type == "cv2line":
                    if num == 0:
                        last_poi_x = int(u)
                        last_poi_y = int(v)
                    #draw thickness is attention
                    cv2.line(mask_seg, (last_poi_x, last_poi_y),
                              (int(u), int(v)), color=(ins_id, ins_id, ins_id),
                             thickness=1)
                    last_poi_x = int(u)
                    last_poi_y = int(v)
                else:
                    # print('mask_seg', mask_seg.shape)
                    value = int(ins_id) if ins_id < 255 else 255
                    mask_seg[v, u, :] = value
                    if u > 0 and self.width_res < 0.6:
                        mask_seg[v, u - 1, :] = value
                    if u + 1 < mask_seg.shape[1] and self.width_res < 0.6:
                        mask_seg[v, u + 1, :] = value

                mask_offset[v, u, 0] = offset_x
                mask_offset[v, u, 1] = offset_y

                mask_z[v, u, 0] = z

        return mask_seg, mask_offset, mask_z


    def prune_3d_lane_by_range(self, lane_3d, x_min, x_max):
        # TODO: solve hard coded range later
        lane_3d = lane_3d[np.logical_and(lane_3d[:, 1] > 0, lane_3d[:, 1] < 200), ...]

        # remove lane points out of x range
        lane_3d = lane_3d[np.logical_and(lane_3d[:, 0] > x_min,
                                         lane_3d[:, 0] < x_max), ...]
        return lane_3d

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

            xs_gt = gt_lane[:, 1]
            ys_gt = gt_lane[:, 0]
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

            if len(xs_out) < 2:
                continue

            fit_lane = np.zeros((len(xs_out), 3))
            fit_lane[:, 0] = xs_out
            fit_lane[:, 1] = ys_out
            fit_lane[:, 2] = zs_out

            mask_idex = (self.y_range[0] <= fit_lane[:, 0]) & (fit_lane[:, 0] <= self.y_range[1]) & (self.x_range[0] <= fit_lane[:, 1]) & (fit_lane[:, 1] <= self.x_range[1])

            fit_lane = fit_lane[mask_idex]

            if interp:
                fit_lane = self.coords_interpolation(fit_lane)
                fit_lane = np.array(fit_lane)

            fit_lanes.append(fit_lane)

        return fit_lanes

    def get_seg_offset(self, idx, smooth=True, debug=True):
        index = idx
        # gt_path = os.path.join(self.gt_paths, self.cnt_list[idx][0], self.cnt_list[idx][1])
        gt_path = self.samples[idx]
        gt_path = ops.join(self.gt_paths, gt_path)

        with open(gt_path, 'r') as fr:
            info_dict = json.loads(fr.read())

        image_path = ops.join(self.image_paths, info_dict['file_path'])
        # if not self.test_mode:
        #     self.image_paths = self.image_paths + '/training'
        # else:
        #     self.image_paths = self.image_paths + '/validation'

        image = cv2.imread(image_path)
        image_draw = image.copy()

        image_ori = Image.open(image_path)
        # print("path:", image_path)
        image_h, image_w, _ = image.shape
        with open(gt_path, 'r') as f:
            gt = json.load(f)
        cam_w_extrinsics = np.array(gt['extrinsic'])
        maxtrix_camera2camera_w = np.array([[0, 0, 1, 0],
                                            [-1, 0, 0, 0],
                                            [0, -1, 0, 0],
                                            [0, 0, 0, 1]], dtype=float)
        cam_extrinsics = cam_w_extrinsics @ maxtrix_camera2camera_w  #
        R_vg = np.array([[0, 1, 0],
                         [-1, 0, 0],
                         [0, 0, 1]], dtype=float)
        R_gc = np.array([[1, 0, 0],
                         [0, 0, 1],
                         [0, -1, 0]], dtype=float)
        cam_extrinsics_persformer = copy.deepcopy(cam_w_extrinsics)
        cam_extrinsics_persformer[:3, :3] = np.matmul(np.matmul(
            np.matmul(np.linalg.inv(R_vg), cam_extrinsics_persformer[:3, :3]),
            R_vg), R_gc)
        cam_extrinsics_persformer[0:2, 3] = 0.0
        matrix_lane2persformer = cam_extrinsics_persformer @ np.linalg.inv(maxtrix_camera2camera_w)

        cam_intrinsic = np.array(gt['intrinsic'])
        lanes = gt['lane_lines']

        matrix_IPM2ego = IPM2ego_matrix(
            ipm_center=(int(self.x_range[1] / self.meter_per_pixel), int(self.y_range[1] / self.meter_per_pixel)),
            m_per_pixel=self.meter_per_pixel)  #


        image_gt = np.zeros((image_h, image_w), dtype=np.uint8)
        res_points_d = {}
        vis_res_points_d = []
        lane_ego_persformers = []
        vis_res_points_ipms = []
        lane_egos = []
        for idx in range(len(lanes)):
            lane1 = lanes[idx]
            lane_camera_w = np.array(lane1['xyz']).T[np.array(lane1['visibility']) == 1.0].T
            lane_camera_w = np.vstack((lane_camera_w, np.ones((1, lane_camera_w.shape[1]))))
            lane_ego_persformer = matrix_lane2persformer @ lane_camera_w  #
            lane_ego_persformer[0], lane_ego_persformer[1] = lane_ego_persformer[1], -1 * lane_ego_persformer[0]
            lane_ego = cam_w_extrinsics @ lane_camera_w  #
            ''' plot uv '''
            uv1 = ego2image(lane_ego[:3], cam_intrinsic, cam_extrinsics)
            cv2.polylines(image_gt, [uv1[0:2, :].T.astype(np.int)], False, idx + 1, self.lane2d_thick)

            distance = np.sqrt((lane_ego_persformer[1][0] - lane_ego_persformer[1][-1]) ** 2 + (
                    lane_ego_persformer[0][0] - lane_ego_persformer[0][-1]) ** 2)
            if distance < self.lane_length_threshold:
                continue
            y = lane_ego_persformer[1]
            x = lane_ego_persformer[0]
            z = lane_ego_persformer[2]

            lane_ego_persformers.append(lane_ego_persformer[0:3, :].T)
            lane_egos.append(lane_ego[:3].T)

        fit_lanes = self.lane_fit(lane_ego_persformers)

        lane_egos = self.lane_fit(lane_egos)

        for idx, fit_lane in enumerate(fit_lanes):
            fit_lane = fit_lane.T
            z = fit_lane[2, :]
            ipm_points = np.linalg.inv(matrix_IPM2ego[:, :2]) @ (fit_lane[:2][::-1] - matrix_IPM2ego[:, 2].reshape(2, 1))
            ipm_points_ = np.zeros_like(ipm_points)
            ipm_points_[0] = ipm_points[1]
            ipm_points_[1] = ipm_points[0]
            res_points = np.concatenate([ipm_points_, np.array([z])], axis=0)

            # res_points_d[idx + 1] = res_points

            # res_points_vis = res_points

            # vis_res_points_d.append(res_points_vis.T)

            # ipm_points = np.linalg.inv(matrix_IPM2ego[:, :2]) @ (res_points[:2] - matrix_IPM2ego[:, 2].reshape(2, 1))
            # ipm_points_ = np.zeros_like(ipm_points)
            # ipm_points_[0] = ipm_points[1]
            # ipm_points_[1] = ipm_points[0]
            # res_points = np.concatenate([ipm_points_, np.array([z])], axis=0)

            res_points_d[idx + 1] = res_points

            res_points_vis = res_points

            vis_res_points_d.append(res_points_vis.T)

            # vis_ipm_points = np.zeros_like(res_points_vis.T)
            # vis_ipm_points[:, 0] = res_points_vis.T[:, 0] * self.meter_per_pixel
            # vis_ipm_points[:, 1] = res_points_vis.T[:, 1] * self.meter_per_pixel
            # vis_ipm_points[:, 2] = res_points_vis.T[:, 2]
            #
            # vis_res_points_ipms.append(vis_ipm_points)

        ipm_gt, offset_y_map, z_map = self.get_y_offset_and_z(res_points_d)

        # _ = self.mask_extract(vis_res_points_ipms, gt_category=None, gt_visibility=None)

        ''' virtual camera '''
        if self.use_virtual_camera:
            sc = Standard_camera(self.vc_intrinsic, self.vc_extrinsics, self.vc_image_shape,
                                 cam_intrinsic, cam_extrinsics, (image.shape[0], image.shape[1]))
            trans_matrix = sc.get_matrix(height=0)
            # trans_matrix_test = np.linalg.inv(trans_matrix)
            image = cv2.warpPerspective(image, trans_matrix, self.vc_image_shape)
            # cv2.imwrite("/home/slj/data/openlane/calib_imgs/" + str(index) + '.jpg', image)
            image_gt = cv2.warpPerspective(image_gt, trans_matrix, self.vc_image_shape)
            #..................................
            # sc_test = Standard_camera(self.vc_intrinsic, self.vc_extrinsics, (1024, 576),
            #                      cam_intrinsic, cam_extrinsics, (576, 1024))
            # trans_matrix_test = sc_test.get_matrix(height=0)

        intrins = []
        post_trans = []
        post_rots = []
        extrinsics = []
        if debug:
            extrinsics.append(torch.tensor(self.vc_extrinsics).float())
            intrins.append(torch.cat((torch.Tensor(self.vc_intrinsic), torch.zeros((3, 1))), dim=1).float())
            # extrinsics.append(torch.tensor(cam_extrinsics).float())
            # intrins.append(torch.cat((torch.Tensor(cam_intrinsic), torch.zeros((3, 1))), dim=1).float())

            extrinsics = torch.stack(extrinsics)
            extrinsics = torch.linalg.inv(extrinsics)
            intrins = torch.stack(intrins)
            image_draw = image.copy()
            visu_path = './test_vis'
            calib = np.matmul(intrins, extrinsics)

            for gt_lane in vis_res_points_d:
                z = gt_lane[:, 2]
                gt_lane = gt_lane.T
                gt_lane = gt_lane[:2, :][::-1]
                gt_lane = matrix_IPM2ego[:, :2] @ gt_lane
                gt_lane = gt_lane + matrix_IPM2ego[:, 2].reshape(2, 1)
                gt_lane = np.concatenate([gt_lane, np.array([z])], axis=0)
                gt_lane = torch.tensor(gt_lane).float()
                gt_lane = gt_lane.T
                img_points, _ = self.perspective(calib, gt_lane)

                post_img_points = img_points
                # post_img_points = []
                # for img_point in img_points:
                #     img_point = torch.matmul(post_rots[0, :2, :2], img_point) + post_trans[0, :2]
                #     post_img_points.append(img_point.detach().cpu().numpy())
                post_img_points = np.array(post_img_points)
                x_2d, y_2d = post_img_points[:, 0].astype(np.int32), post_img_points[:, 1].astype(np.int32)
                for k in range(1, img_points.shape[0]):
                    image_draw = cv2.line(image_draw, (x_2d[k - 1], y_2d[k - 1]),
                                     (x_2d[k], y_2d[k]), (0, 0, 255), 4)
            cv2.imwrite(visu_path + "/debug_img.jpg", image_draw)

        return image_ori, image, image_gt, ipm_gt, offset_y_map, z_map, cam_extrinsics, cam_intrinsic, fit_lanes, info_dict['file_path'], trans_matrix#vis_res_points_ipms
        # return image_ori, image, image_gt, ipm_gt, offset_y_map, z_map, mask_haf, mask_vaf, cam_extrinsics, cam_intrinsic, vis_res_points_d



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

    def sample_augmentation(self):
        fW, fH = self.input_shape[1], self.input_shape[0]
        resize = (fW / self.vc_image_shape[0], fH / self.vc_image_shape[1])
        resize_dims = (fW, fH)
        return resize, resize_dims

    def img_transform(self, img, resize, resize_dims):
        post_rot2 = torch.eye(2)
        post_tran2 = torch.zeros(2)

        img = img.resize(resize_dims)

        rot_resize = torch.Tensor([[resize[0], 0],
                                   [0, resize[1]]])
        post_rot2 = rot_resize @ post_rot2
        post_tran2 = rot_resize @ post_tran2

        post_tran = torch.zeros(3)
        post_rot = torch.eye(3)
        post_tran[:2] = post_tran2
        post_rot[:2, :2] = post_rot2
        return img, post_rot, post_tran


    def get_data_info(self, idx):
        '''
        :param idx:
        :return:
        '''
        image_ori, image, image_gt, ipm_gt, offset_y_map, z_map, cam_extrinsic, cam_intrinsic, res_points_d, file_path, trans_matrix = self.get_seg_offset(idx)
        # image_ori, image, image_gt, ipm_gt, offset_y_map, z_map, mask_haf, mask_vaf, cam_extrinsics, cam_intrinsic, res_points_d = self.get_seg_offset(idx)
        # cv2.imwrite("./test_vis/img.jpg", image)

        transformed = self.trans_image(image=image)
        image = transformed["image"]
        # image_vis = image_ori.copy()

        resize, resize_dims = self.sample_augmentation()
        image_ori, post_rot, post_tran = self.img_transform(image_ori, resize, resize_dims)
        image_ori = normalize_img(image_ori)

        cam_intrinsics = []
        cam_extrinsics = []
        post_trans = []
        post_rots = []
        cam_extrinsics.append(torch.tensor(cam_extrinsic).float())
        cam_intrinsics.append(torch.cat((torch.Tensor(cam_intrinsic), torch.zeros((3, 1))), dim=1).float())
        post_trans.append(post_tran)
        post_rots.append(post_rot)
        post_trans, post_rots, cam_extrinsics, cam_intrinsic = torch.stack(post_trans), torch.stack(post_rots),\
                                            torch.stack(cam_extrinsics), torch.stack(cam_intrinsics)
        cam_extrinsics = torch.linalg.inv(cam_extrinsics)  # change cam2glob to glob2cam

        ''' 2d gt'''
        image_gt = cv2.resize(image_gt, (self.output2d_size[1], self.output2d_size[0]), interpolation=cv2.INTER_NEAREST)
        image_gt_instance = torch.tensor(image_gt).unsqueeze(0)  # h, w, c
        image_gt_segment = torch.clone(image_gt_instance)
        image_gt_segment[image_gt_segment > 0] = 1
        ''' 3d gt'''
        ipm_gt_instance = torch.tensor(ipm_gt).unsqueeze(0)  # h, w, c0
        ipm_gt_offset = torch.tensor(offset_y_map).unsqueeze(0)
        ipm_gt_z = torch.tensor(z_map).unsqueeze(0)
        ipm_gt_segment = torch.clone(ipm_gt_instance)
        ipm_gt_segment[ipm_gt_segment > 0] = 1

        # mask_haf, mask_vaf = self.engine_af(ipm_gt_segment.detach().cpu().numpy()[0])
        # mask_vaf = np.transpose(mask_vaf, (2, 0, 1))

        # mask_haf = torch.tensor(mask_haf)
        # mask_vaf = torch.tensor(mask_vaf)

        cv2.imwrite('./test_vis/ipm_gt_segment.png', ipm_gt_segment.detach().cpu().numpy()[0] * 100)
        cv2.imwrite('./test_vis/mask_offset.png', ipm_gt_offset.detach().cpu().numpy()[0] * 1000)
        cv2.imwrite("./test_vis/mask_z.png", -ipm_gt_z.detach().cpu().numpy()[0] * 100000)

        # cv2.imwrite("./test_vis/mask_haf.png", -mask_haf.detach().cpu().numpy() * 100)
        # cv2.imwrite("./test_vis/mask_vaf_0.png",-mask_vaf.detach().cpu().numpy()[0] * 100)
        # cv2.imwrite("./test_vis/mask_vaf_1.png", -mask_vaf.detach().cpu().numpy()[1] * 100)


        input_dict = dict(
            image_ori=image_ori,
            image=image,
            extrinsic=cam_extrinsics,
            intrin=cam_intrinsic,
            post_tran=post_trans,
            post_rot=post_rots,
            ipm_gt_segment=ipm_gt_segment.float(),
            ipm_gt_instance=ipm_gt_instance.float(),
            ipm_gt_offset=ipm_gt_offset.float(),
            ipm_gt_z=ipm_gt_z.float(),
            res_points_d=res_points_d,
            file_path=file_path,
            image_gt_segment=image_gt_segment.float(),
            image_gt_instance=image_gt_instance.float(),
            trans_matrix=trans_matrix,
        )

        return input_dict


    def init_dataset_3D(self, json_file_dir):
        filter_samples = []
        samples = glob.glob(json_file_dir + '**/*.json', recursive=True)
        for i, sample in enumerate(samples):
            label_file_path = ops.join(json_file_dir, sample)
            with open(label_file_path, 'r') as fr:
                info_dict = json.loads(fr.read())
            image_path = ops.join(self.image_paths, info_dict['file_path'])
            if not ops.exists(image_path):
                # print('{:s} not exist'.format(image_path))
                continue
            # if i < 1014:
            #     continue
            filter_samples.append(sample)
            # if len(filter_samples) > 4000:
            if len(filter_samples) > 47000:
                break
            # print("image_path:", image_path)

        # return samples
        return filter_samples

    def __getitem__(self, idx):
        # print("idx:", idx)
        input_dict = self.get_data_info(idx)
        data = self.pipeline(input_dict)
        return data

    def __len__(self):
        return len(self.samples)  # // 20



if __name__ == "__main__":
    pass