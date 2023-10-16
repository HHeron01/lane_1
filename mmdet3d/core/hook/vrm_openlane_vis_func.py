import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import torch
import os
from .bevlane_post import BevLanePost
from mmcv.runner.hooks import HOOKS, Hook
import torchvision
from matplotlib.pyplot import MultipleLocator
from .cluster import embedding_post
from .post_process import bev_instance2points_with_offset_z
from sklearn.cluster import MeanShift, estimate_bandwidth
from torch.nn import functional as F
# from mmdet3d.datasets.multiview_datasets.coord_util import image2ego_byheight
# from mmdet3d.datasets.multiview_datasets.standard_camera_cpu import Standard_camera


def torch_inputs_to_imgs(inputs, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    inputs_cpu = np.transpose(inputs.data.cpu().numpy(), [1, 2, 0])
    img = ((inputs_cpu * std * mean) * 255).astype(np.uint8).copy()
    return img

class NormalizeInverse(torchvision.transforms.Normalize):
    #  https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/8
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())

denormalize_img = torchvision.transforms.Compose((
    NormalizeInverse(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    torchvision.transforms.ToPILImage(),
))


def IPM2ego_matrix(ipm_center=None, m_per_pixel=None, ipm_points=None, ego_points=None):
    if ipm_points is None:
        center_x, center_y = ipm_center[0] * m_per_pixel, ipm_center[1] * m_per_pixel
        M = np.array([[-m_per_pixel, 0, center_x], [0, -m_per_pixel, center_y]])
    else:
        pts1 = np.float32(ipm_points)
        pts2 = np.float32(ego_points)
        M = cv2.getAffineTransform(pts1, pts2)
    return M


# @HOOKS.register_module()
class Vrm_LaneVisFunc(object):

    def __init__(self, use_offset=False, use_off_z=True) -> None:
        self.color_map = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (100, 255, 0), (100, 0, 255), (255, 100, 0),
            (0, 100, 255), (255, 0, 100), (0, 255, 100),
            (255, 255, 255), (0, 0, 0), (0, 100, 100)
        ]

        depth_range = (3, 103)
        width_range = (-12, 12)

        # y_range = (3, 103)
        # x_range = (-12, 12)
        self.meter_per_pixel = 0.5  # grid size
        bev_shape = (int((depth_range[1] - depth_range[0]) / self.meter_per_pixel), int((width_range[1] - width_range[0]) / self.meter_per_pixel))

        self.width_res = 0.5
        self.depth_res = 0.5

        self.width_range = width_range
        self.depth_range = depth_range
        self.use_offset = use_offset
        self.use_off_z = use_off_z

        self.post_conf = -0.7
        self.post_emb_margin = 6.0
        self.post_min_cluster_size = 15


        # NOTE: postprocessor need Init
        # self.post_engine = BevLanePost(None, width_range, depth_range, self.meter_per_pixel, self.meter_per_pixel, use_offset=self.use_offset, use_off_z=self.use_off_z)
        self.pred_z = -0.3

        self.vc_intrinsic = np.array([[2081.5212033927246, 0.0, 934.7111248349433],
                                     [0.0, 2081.5212033927246, 646.3389987785433],
                                     [0.0, 0.0, 1.0]])
        self.vc_extrinsics = np.array([[-0.002122161262459438, 0.010697496358766389, 0.9999405282331697, 1.5441039498273286],
            [-0.9999378331046326, -0.010968621415360667, -0.0020048117763292747, -0.023774034344867204],
            [0.010946522625388108, -0.9998826195688676, 0.01072010851209982, 2.1157397903843567],
            [0.0, 0.0, 0.0, 1.0]])

        self.matrix_IPM2ego = IPM2ego_matrix(
            ipm_center=(int(self.depth_range[1] / self.meter_per_pixel), int(self.width_range[1] / self.meter_per_pixel)),
            m_per_pixel=self.meter_per_pixel)  #

    # @torchsnooper.snoop()
    def get_cam_imgs(self, img):
        """
        inputs is from dataloader single frame, get images from "imgs"
        """
        # NOTE: inputs["imgs"]
        # NOTE: torch size is [batch=8, N=6, T=1, C=3, H=256, W=480]
        # print('get_cam_imgs inputs["imgs"]', inputs["imgs"].shape)

        # cam_img = torch_inputs_to_imgs(torch.squeeze(img[0], dim=0))
        img = denormalize_img(torch.squeeze(img[0], dim=0))
        # img = denormalize_img(img)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img


    def post_process(self, results):    # NOTE: results = [odhead_preds, lanehead_preds]
        # NOTE: lanehead_pred = [(bin_seg, embedding, haf, vaf, hoff, voff), bev_feat]
        binary_seg, embedding, offset_y, z_pred, topdown = results
        binary_seg = binary_seg.detach().cpu().numpy()
        embedding = embedding.detach().cpu().numpy()
        offset_y = torch.sigmoid(offset_y).detach().cpu().numpy()[0][0]
        z_pred = z_pred.detach().cpu().numpy()[0][0]

        prediction = (binary_seg, embedding)

        canvas, ids = embedding_post(prediction, conf=self.post_conf, emb_margin=self.post_emb_margin,
                                     min_cluster_size=self.post_min_cluster_size, canvas_color=False)
        lines = bev_instance2points_with_offset_z(canvas, max_x=self.depth_range[1],
                                                  meter_per_pixal=(self.meter_per_pixel, self.meter_per_pixel),
                                                  offset_y=offset_y, Z=z_pred)

        frame_lanes_pred = []

        for lane in lines:
            pred_in_persformer = np.array([-1 * lane[1], lane[0], lane[2]])
            frame_lanes_pred.append(pred_in_persformer.T)

        # with open(os.path.join(self.postprocess_save_path, files[1] + '.json'), 'w') as f1:
        #     json.dump([frame_lanes_pred, frame_lanes_gt], f1)

        fit_lanes = self.lane_fit(frame_lanes_pred)

        return fit_lanes

    def q_post_process(self, results):    # NOTE: results = [odhead_preds, lanehead_preds]
        # NOTE: lanehead_pred = [(bin_seg, embedding, haf, vaf, hoff, voff), bev_feat]
        # binary_seg, embedding, offset_y, z_pred, topdown = results
        topdown, binary_seg, embedding, offset_y, z_pred, _, _ = results
        # binary_seg, embedding, offset_y, z_pred, _, _ = results
        # binary_seg = binary_seg.detach().cpu().numpy()
        # embedding = embedding.detach().cpu().numpy()
        offset_y = torch.sigmoid(torch.tensor(offset_y)).cpu().numpy()[0][0]
        z_pred = z_pred[0][0]

        prediction = (binary_seg, embedding)

        canvas, ids = embedding_post(prediction, conf=self.post_conf, emb_margin=self.post_emb_margin,
                                     min_cluster_size=self.post_min_cluster_size, canvas_color=False)
        lines = bev_instance2points_with_offset_z(canvas, max_x=self.depth_range[1],
                                                  meter_per_pixal=(self.meter_per_pixel, self.meter_per_pixel),
                                                  offset_y=offset_y, Z=z_pred)

        frame_lanes_pred = []

        for lane in lines:
            pred_in_persformer = np.array([-1 * lane[1], lane[0], lane[2]])
            frame_lanes_pred.append(pred_in_persformer.T)

        fit_lanes = self.lane_fit(frame_lanes_pred)

        return fit_lanes

    def img_post_process(self, results, ipm_img, post_rots, post_trans, extrinsics, intrins):
        # NOTE: img post
        # post_rots = post_rots.cuda()
        # post_trans = post_trans.cuda()

        color = np.array([[255, 125, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255]], dtype='uint8')
        # ipm_img = cv2.cvtColor(ipm_img, cv2.COLOR_BGR2RGB)
        # seg_img = np.zeros_like(ipm_img)
        seg_img = np.zeros([576, 1024, 3])
        bin_seg, embedding = results

        bin_seg = F.interpolate(bin_seg, (576, 1024), mode='bilinear')
        embedding = F.interpolate(embedding, (576, 1024), mode='bilinear')

        bin_seg, embedding = bin_seg[0], embedding[0]

        embedding = embedding.detach().cpu().numpy()
        embedding = np.transpose(embedding, (1, 2, 0))
        bin_seg = bin_seg.detach().cpu().numpy()
        # bin_seg = np.argmax(bin_seg, axis=0)
        bin_seg[bin_seg < self.post_conf] = 0
        # bin_seg[bin_seg >= self.post_conf] = 1
        # print("bin_seg:", np.unique(bin_seg))

        cluster_result = np.zeros(bin_seg.shape, dtype=np.int32)

        cluster_list = embedding[bin_seg[0] > 0]
        fit_lanes = []
        if len(cluster_list) == 0:
            return ipm_img, fit_lanes

        mean_shift = MeanShift(bandwidth=1.5, bin_seeding=True, n_jobs=-1)
        mean_shift.fit(cluster_list)

        labels = mean_shift.labels_
        cluster_result[bin_seg > 0] = labels + 1

        cluster_lanes = []

        cluster_result[cluster_result > 10] = 0
        for idx in np.unique(cluster_result):
            if len(cluster_result[cluster_result==idx]) < 20:
                cluster_result[cluster_result==idx] = 0

        # cluster_result = np.expand_dims(cluster_result[0], axis=-1)
        # cluster_result = np.concatenate((cluster_result, cluster_result, cluster_result), axis=-1)
        #
        # cluster_result = cv2.warpPerspective(cluster_result[0].astype(np.uint8), trans_matrix, (1920, 1280))

        for i, lane_idx in enumerate(np.unique(cluster_result)):
            if lane_idx==0:
                continue
            # seg_img[cluster_result == lane_idx] = self.color_map[lane_idx % len(self.color_map)]
            seg_img[cluster_result[0] == lane_idx] = self.color_map[lane_idx % len(self.color_map)]
            # print("cluster_result", lane_idx, np.argwhere(cluster_result[0] == lane_idx))

            cluster_lanes.append(np.argwhere(cluster_result[0] == lane_idx))

        # image = cv2.addWeighted(src1=seg_img, alpha=0.8, src2=ipm_img, beta=1., gamma=0.)
        # cv2.putText(image, "2D_PRED", (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        # fit_lanes = []
        for cluster_lane in cluster_lanes:
            # xs_gt = cluster_lane[:, 0]
            # ys_gt = cluster_lane[:, 1]

            xs_gt = cluster_lane[:, 1]
            ys_gt = cluster_lane[:, 0]

            poly_params_yx = np.polyfit(ys_gt, xs_gt, deg=3)

            y_min, y_max = np.min(ys_gt), np.max(ys_gt)
            y_min = math.floor(y_min)
            y_max = math.ceil(y_max)
            y_sample = np.array(range(y_min, y_max, 1))
            ys_out = np.array(y_sample, dtype=np.float32)

            xs_out = np.polyval(poly_params_yx, ys_out)

            fit_lane = np.zeros((len(xs_out), 2))

            fit_lane[:, 0] = xs_out
            fit_lane[:, 1] = ys_out
            # fit_lane[:, 0] = ys_out
            # fit_lane[:, 1] = xs_out

            input_shape = (576, 1024)  # h, w
            mask_idex = (0 <= fit_lane[:, 0]) & (fit_lane[:, 0] < 1024) & (
                        0 <= fit_lane[:, 1]) & (fit_lane[:, 1] <= 576)

            if not any(mask_idex):
                continue

            fit_lane = fit_lane[mask_idex]

            post_img_points = []
            for img_point in fit_lane:
                # img_point = torch.matmul(post_rots[0, :2, :2], torch.tensor(img_point.astype(np.float32))) + post_trans[0, :2]
                # post_img_points.append(img_point.detach().cpu().numpy())
                img_point = torch.matmul(post_rots[0, :2, :2].inverse(), torch.tensor(img_point.astype(np.float32))) - post_trans[0, :2]
                # img_point = torch.matmul(post_rots[0, :2, :2],  torch.tensor(img_point.astype(np.float32))) + post_trans[0, :2]
                # image_gt = cv2.warpPerspective(image_gt, trans_matrix, self.vc_image_shape)
                img_point = img_point.detach().cpu().numpy()
                # img_point = np.stack((img_point, 1), 1)
                # img_point = img_point @ trans_matrix[:2, :2]
                post_img_points.append(img_point)
            post_img_points = np.array(post_img_points)

            # fit_lanes.append(fit_lane)
            fit_lanes.append(post_img_points)

        for lane_idx, lane in enumerate(fit_lanes):
            x_2d = lane[:, 0]
            y_2d = lane[:, 1]
            for k in range(1, lane.shape[0]):
                ipm_img = cv2.line(ipm_img, (int(x_2d[k - 1]), int(y_2d[k - 1])),
                                 (int(x_2d[k]), int(y_2d[k])), self.color_map[lane_idx % len(self.color_map)], 2)

        ego_lanes = []
        for img_lane in fit_lanes:
            one_np = np.ones((img_lane.shape[0], 1)).astype(np.float32)
            img_lane = np.concatenate((img_lane, one_np), axis=1)
            # img_lane = img_lane[:, [1, 0, 2]]
            # ego_lane = self.image2ego(img_lane.T, np.array(self.vc_intrinsic), np.array(self.vc_extrinsics),)
            # ego_lane = self.image2ego(img_lane.T, intrins[0][0], extrinsics[0][0],)
            # cam_intrinsics.append(torch.cat((torch.Tensor(cam_intrinsic), torch.zeros((3, 1))), dim=1).float())
            # intrins = torch.cat((intrins[0][:, :3].inverse(), torch.zeros(3, 1)), dim=1)
            # intrins = intrins.unsqueeze(0)
            # calib = np.matmul(intrins, extrinsics.inverse())
            # ego_lanes = self.inverse_perspective(calib, img_lane)

            # ego_lane = self.imageview2ego(img_lane.T, intrins[0][:, :3], extrinsics[0])
            ego_lane = self.imageview2ego(img_lane.T, self.vc_intrinsic, np.linalg.inv(self.vc_extrinsics))
            ego_lanes.append(ego_lane.T)

        fit_lanes = self.lane_fit(ego_lanes)
        # fit_ego_lanes = []
        # for ego_lane in fit_lanes:
        #     axis_egos = np.array([-1 * ego_lane[:, 1], ego_lane[:, 0], ego_lane[0, 2]])
        #     fit_ego_lanes.append(axis_egos)

        # return image
        # return ipm_img, ego_lanes
        return ipm_img, fit_lanes

    def ego2image(self, ego_points, camera_intrinsic, camera2ego_matrix):
        """
        :param ego_points:  3*n
        :param camera_intrinsic: 3*3
        :param camera2ego_matrix:  4*4
        :return:
        """
        ego2camera_matrix = np.linalg.inv(camera2ego_matrix)
        camera_points = np.dot(ego2camera_matrix[:3, :3], ego_points) + \
                        ego2camera_matrix[:3, 3].reshape(3, 1)
        image_points_ = camera_intrinsic @ camera_points
        image_points = image_points_ / image_points_[2]
        return image_points

    def image2ego(self, image_points, camera_intrinsic, camera2ego_matrix):

        camera_points_ = np.linalg.inv(camera_intrinsic) @ image_points
        ego2camera_matrix_T = np.linalg.inv(camera2ego_matrix)[:3, 3].reshape(3, 1)
        camera_points = camera_points_ - ego2camera_matrix_T
        ego_points = np.dot(camera2ego_matrix[:3, :3], camera_points)
        ego_points = ego_points[:, [1, 0, 2]]
        return ego_points

    def imageview2ego(self, image_view_points, camera_intrinsic, ego2camera_matrix, height=0):
        camera_intrinsic_inv = np.linalg.inv(camera_intrinsic)
        # height = np.linalg.inv(ego2camera_matrix)[2, 3]
        # print("height:", height)
        # camera_intrinsic_inv = camera_intrinsic
        R_inv = ego2camera_matrix[:3, :3].T

        T = ego2camera_matrix[:3, 3]
        mat1 = np.dot(np.dot(R_inv, camera_intrinsic_inv), image_view_points)
        mat2 = np.dot(R_inv, T)

        Zc = (height + mat2[2]) / mat1[2]
        points_ego = Zc * mat1 - np.expand_dims(mat2, 1)

        axis_egos = np.array([-1 * points_ego[1, :], points_ego[0, :], points_ego[2, :]])
        # return points_ego
        return axis_egos

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

    def inverse_perspective(self, matrix, vector):
        """Applies perspective projection to a vector using projection matrix."""
        # tmp = torch.zeros_like(vector)
        # tmp[..., 0] = vector[..., 0]
        # tmp[..., 1] = vector[..., 2]
        # tmp[..., 2] = vector[..., 1]
        # vector = tmp
        matrix = torch.tensor(matrix)
        vector = torch.tensor(vector)

        # matrix = matrix.inverse()
        vector = vector.unsqueeze(-1) - matrix[..., [-1]]

        ego_vector = torch.matmul(matrix[..., :-1], vector)
        ego_vector = ego_vector.squeeze(-1)
        ego_vector = ego_vector.detach().cpu().numpy()

        return ego_vector


    def img_trans(self, features):
        cv2.normalize(features, features, 0., 1., cv2.NORM_MINMAX)
        norm_img = np.asarray(features * 255, dtype=np.uint8)
        out_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)
        return out_img

    def get_pred_bev_feat(self, topdown):
        intermidate = torch.max(topdown, dim=1)[0]
        intermidate = intermidate.detach().cpu().numpy()
        heat_img = self.img_trans(intermidate[0])
        heat_img = np.ascontiguousarray(heat_img)
        return heat_img


    def draw_pic_2d(self, image, gt_lanes, intrin, extrinsic,
                    post_rots, post_trans, pred=True):
        # intrins = []
        # extrinsics = []
        # post_trans = []
        # post_rots = []
        # extrinsics.append(extrinsic.float())
        # intrins.append(torch.cat((intrin, torch.zeros((3, 1))), dim=1).float())
        # post_trans.append(post_tran)
        # post_rots.append(post_rot)
        #
        # extrinsics = torch.stack(extrinsics)
        # extrinsics = torch.linalg.inv(extrinsics)
        # intrins = torch.stack(intrins)
        # post_trans = torch.stack(post_trans)
        # post_rots = torch.stack(post_rots)

        calib = np.matmul(intrin, extrinsic)
        for line_id, gt_lane in enumerate(gt_lanes):
            # gt_lane = gt_lane.T
            color = self.color_map[line_id % len(self.color_map)]
            if pred:
                cv2.putText(image, "PRED", (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                # gt_lane = gt_lane['coordinates']
                if len(gt_lane) < 2:
                    continue
                gt_lane = np.array(gt_lane)
                a = np.ones([gt_lane.shape[0], 1])
                if not self.use_off_z:
                    a = np.full((gt_lane.shape[0], 1), fill_value=self.pred_z)
                    gt_lane = np.concatenate((gt_lane, a), axis=1)
            else:
                if len(gt_lane) < 2:
                    continue
                cv2.putText(image, "GT", (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
            gt_lane = gt_lane[:, [1, 0, 2]]

            # gt_lane[:, 2] = 0.
            gt_lane = torch.tensor(gt_lane).float()
            img_points, _ = self.perspective(calib, gt_lane)

            # post_img_points = img_points
            post_img_points = []
            for img_point in img_points:
                img_point = torch.matmul(post_rots[0, :2, :2], img_point) + post_trans[0, :2]
                post_img_points.append(img_point.detach().cpu().numpy())
            post_img_points = np.array(post_img_points)
            x_2d, y_2d = post_img_points[:, 0].astype(np.int32), post_img_points[:, 1].astype(np.int32)
            for k in range(1, img_points.shape[0]):
                image = cv2.line(image, (x_2d[k - 1], y_2d[k - 1]),
                                 (x_2d[k], y_2d[k]), color, 2)
        return image


    def get_bevlines_img(self, pred_img, bevlines):
        img = copy.deepcopy(pred_img)
        img = img.transpose(1, 2, 0)
        img = np.concatenate((img, img, img), axis=-1)
        img = img * 150

        cv2.putText(img, "PRED", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

        for line_id, line in enumerate(bevlines):
            color = self.color_map[line_id % len(self.color_map)]
            # for point in line['bevindexs']:
            for point in line:
                u = point[0]
                v = point[1]
                # NOTE: color one grid-cell
                try:
                    cv2.circle(img, (u, v), 1, color)
                except:
                    print("u_v:", u, v)
        return img

    def get_points_img(self, gt_img, points):
        img = copy.deepcopy(gt_img)
        img = np.expand_dims(img, axis=0)
        img = img.transpose(1, 2, 0)
        img = np.concatenate((img, img, img), axis=-1)
        img = img * 150
        width_res = self.width_res
        depth_res = self.depth_res
        width_range = self.width_range
        depth_range = self.depth_range

        # cv2.putText(img, "GT", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)

        for line_id, line in enumerate(points):
            color = self.color_map[line_id % len(self.color_map)]
            for point_imu in line:
                x = point_imu[0] #* self.bev_upsample
                y = point_imu[1] #* self.bev_upsample

                # TODO: use self param
                u = int((x - width_range[0]) / width_res)
                v = int((depth_range[1] - y) / depth_res)
                # NOTE: color one grid-cell
                cv2.circle(img, (u, v), 1, color)
        return img

    def get_lane_imu_img_2D(self, points, bevlines, iego_lanes=None):
        filepath = "mmdet3d/core/hook/imu_2d_compare.png"
        # print("filepath: " + filepath)
        fig_2d = plt.figure(figsize=(6.4, 6.4))
        plt.grid(linestyle='--', color='y', linewidth=0.5)
        x_major_locator = MultipleLocator(1)
        y_major_locator = MultipleLocator(4)
        ax = plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)
        ax.yaxis.set_major_locator(y_major_locator)
        # plt.xticks(range(-19, 19, 1))

        for line_id, line in enumerate(points):

            x_data = []
            y_data = []
            for poi in line:
                x_data.append(poi[0])
                y_data.append(poi[1])
            plt.plot(x_data, y_data, linestyle='-', color='b', linewidth=1)

        for line_id, line in enumerate(bevlines):
            x_data = []
            y_data = []
            if len(line) < 2:
                continue
            for poi in line:
                x_data.append(poi[0])
                y_data.append(poi[1])
            plt.plot(x_data, y_data, linestyle='-', color='r', linewidth=1)

        if iego_lanes is not None:
            for line_id, line in enumerate(iego_lanes):
                x_data = []
                y_data = []
                if len(line) < 2:
                    continue
                for poi in line:
                    x_data.append(poi[0])
                    y_data.append(poi[1])

                    # x_data.append(-poi[1])
                    # y_data.append(poi[0])
                plt.plot(x_data, y_data, linestyle='-', color='g', linewidth=1)

        plt.xlabel('X: offset, res=0.5 * 0.5')
        plt.ylabel('Y: distance')
        plt.title("Only show X_Y : GroundTruth: Blue __;   Prediction: Red __； Ipm: Green __")
        plt.savefig(filepath)
        plt.cla()
        plt.close()
        return filepath

    def get_lane_imu_img_3D(self, points, bevlines, iego_lanes=None):
        filepath = "mmdet3d/core/hook/imu_3d_compare.png"
        z_min = -1
        z_max = 1
        fig_3d = plt.figure(dpi=100, figsize=(6.4, 6.4))
        plt.style.use('seaborn-white')
        plt.rc('font', family='Times New Roman', size=10)
        ax = fig_3d.gca(projection='3d')
        line1, line2 = None, None
        plot_lines = {}
        plot_lines["gt"] = []
        plot_lines["pred"] = []
        plot_lines["ipm"] = []
        for gt_lane in points:
            # print("gt_lane:", gt_lane)
            if len(gt_lane.shape) < 2:
                continue
            if gt_lane.shape == (0, 3):
                # print("gt_:", gt_lane)
                continue
            line_gt = {}
            line_gt["x_3d"] = gt_lane[:, 0].tolist()
            line_gt["y_3d"] = gt_lane[:, 1].tolist()
            line_gt["z_3d"] = gt_lane[:, 2].tolist()
            plot_lines["gt"].append(line_gt)
            fit1 = np.polyfit(gt_lane[:, 1], gt_lane[:, 0], 2)
            fit2 = np.polyfit(gt_lane[:, 1], gt_lane[:, 2], 2)
            f_xy = np.poly1d(fit1)
            f_zy = np.poly1d(fit2)
            y_g = np.linspace(min(gt_lane[:, 1]), max(gt_lane[:, 1]), 5 * len(gt_lane[:, 1]))
            x_g = f_xy(y_g)
            z_g = f_zy(y_g)
            if z_min == -1 and z_max == 1:
                z_max = max(z_g)
                z_min = min(z_g)
            else:
                if max(z_g) > z_max:
                    z_max = max(z_g)
                if min(z_g) < z_min:
                    z_min = min(z_g)
            line1, = ax.plot(x_g, y_g, z_g, lw=2, c='blue', alpha=1, label='GroundTruth')

        for pred_lane in bevlines:
            # pred_lane = pred_lane['coordinates']
            if len(pred_lane) < 2:
                continue
            pred_lane = np.array(pred_lane)
            line_pred = {}
            line_pred["x_3d"] = pred_lane[:, 0].tolist()
            line_pred["y_3d"] = pred_lane[:, 1].tolist()
            line_pred["z_3d"] = pred_lane[:, 2].tolist()
            plot_lines["pred"].append(pred_lane)
            fit1 = np.polyfit(pred_lane[:, 1], pred_lane[:, 0], 2)
            fit2 = np.polyfit(pred_lane[:, 1], pred_lane[:, 2], 2)
            f_xy = np.poly1d(fit1)
            f_zy = np.poly1d(fit2)
            y_g = np.linspace(min(pred_lane[:, 1]), max(pred_lane[:, 1]), 5 * len(pred_lane[:, 1]))
            x_g = f_xy(y_g)
            z_g = f_zy(y_g)
            if z_min == -1 and z_max == 1:
                z_max = max(z_g)
                z_min = min(z_g)
            else:
                if max(z_g) > z_max:
                    z_max = max(z_g)
                if min(z_g) < z_min:
                    z_min = min(z_g)
            line2, = ax.plot(x_g, y_g, z_g, lw=2, c='red', alpha=1, label='Prediction')

        if iego_lanes is not None:
            for iego_lane in iego_lanes:
                # pred_lane = pred_lane['coordinates']
                if len(iego_lane) < 2:
                    continue
                iego_lane = np.array(iego_lane)
                line_ipm = {}
                line_ipm["x_3d"] = iego_lane[:, 0].tolist()
                line_ipm["y_3d"] = iego_lane[:, 1].tolist()
                line_ipm["z_3d"] = iego_lane[:, 2].tolist()
                plot_lines["ipm"].append(iego_lane)
                fit1 = np.polyfit(iego_lane[:, 1], iego_lane[:, 0], 2)
                fit2 = np.polyfit(iego_lane[:, 1], iego_lane[:, 2], 2)
                f_xy = np.poly1d(fit1)
                f_zy = np.poly1d(fit2)
                y_g = np.linspace(min(iego_lane[:, 1]), max(iego_lane[:, 1]), 5 * len(iego_lane[:, 1]))
                x_g = f_xy(y_g)
                z_g = f_zy(y_g)
                if z_min == -1 and z_max == 1:
                    z_max = max(z_g)
                    z_min = min(z_g)
                else:
                    if max(z_g) > z_max:
                        z_max = max(z_g)
                    if min(z_g) < z_min:
                        z_min = min(z_g)
                line3, = ax.plot(x_g, y_g, z_g, lw=2, c='green', alpha=1, label='IPM')


        ax.set_xlabel('x-axis', labelpad=10)
        # ax.set_xlim(-10, 10)
        ax.set_ylabel('y-axis', labelpad=10)
        # ax.set_ylim(0, 100)
        ax.set_zlabel('z-axis')
        ax.set_zlim(2 * z_min - z_max, 2 * z_max - z_min)
        ax.zaxis.set_major_locator(plt.MultipleLocator(max(0.1, round((z_max - z_min) * 2 / 5, 1))))
        ax.view_init(20, -60)
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        # plt.gca().set_box_aspect((1,1.5,0.5))
        ax.set_box_aspect((1, 1.5, 0.5))
        if iego_lanes is not None:
            plt.legend([line1, line2, line3], ['gt', 'pred', 'green'], loc=(0.75, 0.7), fontsize=15)
            plt.tick_params(pad=0)
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0, 0)
        else:
            plt.legend([line1, line2], ['gt', 'pred'], loc=(0.75, 0.7), fontsize=15)
            plt.tick_params(pad=0)
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0, 0)
        plt.savefig(filepath)
        plt.cla()
        plt.close()
        return filepath

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

            if len(xs_out) < 2:
                continue

            fit_lane = np.zeros((len(xs_out), 3))
            fit_lane[:, 0] = xs_out
            fit_lane[:, 1] = ys_out
            fit_lane[:, 2] = zs_out

            mask_idex = (self.width_range[0] <= fit_lane[:, 0]) & (fit_lane[:, 0] <= self.width_range[1]) & (self.depth_range[0] <= fit_lane[:, 1]) & (fit_lane[:, 1] <= self.depth_range[1])

            if not any(mask_idex):
                continue

            fit_lane = fit_lane[mask_idex]

            if interp:
                fit_lane = self.coords_interpolation(fit_lane)
                fit_lane = np.array(fit_lane)

            fit_lanes.append(fit_lane)

        return fit_lanes

    def get_disp_img(self, pred_img, gt_img, pred_mask, gt_mask, imu_2d_compare, imu_3d_compare, ipm_img=None):#, bev_add_xy, bev_add_xyz):

        gt_mask = np.expand_dims(gt_mask, axis=-1)
        gt_mask = np.concatenate((gt_mask, gt_mask, gt_mask), axis=-1)

        pred_mask = np.expand_dims(pred_mask, axis=-1)
        pred_mask = np.concatenate((pred_mask, pred_mask, pred_mask), axis=-1)
        # pred_mask = pred_mask.transpose(1, 2, 0)

        h, w, c = gt_img.shape
        mask_h, mask_w, _ = pred_mask.shape
        mask_new_shape = (int((h / mask_h) * mask_w), h)

        compare_h, compare_w, _ = imu_2d_compare.shape
        compare_new_shape = (int((h / compare_h) * compare_w), h)

        pred_mask = cv2.resize(pred_mask, mask_new_shape)
        gt_mask = cv2.resize(gt_mask, mask_new_shape)

        # pred_instance = cv2.resize(pred_instance, mask_new_shape)
        # gt_instance = cv2.resize(gt_instance, mask_new_shape)

        imu_2d_compare = cv2.resize(imu_2d_compare, compare_new_shape)
        imu_3d_compare = cv2.resize(imu_3d_compare, compare_new_shape)

        # disp_pred = np.concatenate((pred_img, pred_mask, pred_instance, imu_2d_compare), axis=1)
        # disp_gt = np.concatenate((gt_img, gt_mask, gt_instance, imu_3d_compare), axis=1)

        disp_pred = np.concatenate((pred_img, pred_mask, imu_2d_compare), axis=1)
        disp_gt = np.concatenate((gt_img, gt_mask, imu_3d_compare), axis=1)

        disp_img_plus = np.concatenate((disp_pred, disp_gt), axis=0)

        if ipm_img is not None:
            disp_img_plus_h, disp_img_plus_w, _ = disp_img_plus.shape
            ipm_h, ipm_w, _ = ipm_img.shape
            disp_img_plus_new_shape = (int((disp_img_plus_h / ipm_h) * ipm_w), disp_img_plus_h)

            ipm_img = cv2.resize(ipm_img, disp_img_plus_new_shape)
            disp_img_plus = np.concatenate((ipm_img, disp_img_plus), axis=1)

        return disp_img_plus


    def __call__(self, runner):

        root_path = runner.work_dir
        # visu_path = os.path.join(root_path, 'vis_pic/')
        visu_path = root_path + '/vis_pic/'
        if not os.path.exists(visu_path):
            os.makedirs(visu_path)
        # get ground truth
        img_metas = runner.data_batch.get('img_metas')

        gt_lanes = img_metas.data[0][0].get('res_points_d')
        file_path = img_metas.data[0][0].get('file_path')
        trans_matrix = img_metas.data[0][0].get('trans_matrix')
        axis_gt_lanes = []
        for gt_lane in gt_lanes:
            # pred_in_persformer = np.array([-1 * lane[1], lane[0], lane[2]])
            axis_gt_lane = np.array([-1 * gt_lane[:, 0], gt_lane[:, 1], gt_lane[:, 2]])
            axis_gt_lanes.append(axis_gt_lane.T)

        # img_input = runner.data_batch.get("img_inputs")
        # imgs, intrins, extrinsics, post_rots, post_trans, undists, bda_rot, rots, trans, grid, drop_idx = img_input
        image_oris, imgs, intrins, extrinsics, post_rots, post_trans = runner.data_batch.get("image_ori"), runner.data_batch.get("image"), runner.data_batch.get("intrin"), runner.data_batch.get("extrinsic"),\
                                                           runner.data_batch.get("post_rot"), runner.data_batch.get("post_tran"),

        # intrins = runner.meta.
        epoch = runner.epoch
        # map_input = runner.data_batch.get("maps")
        gt_mask, gt_instance, mask_offset, mask_z = runner.data_batch.get("ipm_gt_segment"), runner.data_batch.get("ipm_gt_instance"), runner.data_batch.get("ipm_gt_offset"),\
                                                           runner.data_batch.get("ipm_gt_z"),

        # net_out = runner.outputs.get('net_out')
        bev_net_out, net_out_2d = runner.outputs.get('net_out')

        binary_seg, embedding, offset_feature, z_feature, topdown = bev_net_out

        ori_img = cv2.imread('/home/slj/data/openlane/openlane_all/images/' + file_path)
        ori_img = cv2.warpPerspective(ori_img, trans_matrix, (1920, 1280))  # self.vc_image_shape)
        img = self.get_cam_imgs(image_oris)
        # img = cv2.resize(img, (1024, 576))
        ipm_img = self.get_cam_imgs(imgs)
        img_draw_pred = img.copy()
        img_draw_gt = img.copy()
        # ground truth
        mask_gt = gt_mask[0].detach().cpu().numpy()
        # pred
        bevlines = self.post_process(bev_net_out)

        ipm_img_show, iego_lanes = self.img_post_process(net_out_2d, ori_img, post_rots[0],
                                                             post_trans[0], extrinsics[0], intrins[0])

        # ipm_img_show = self.img_post_process(net_out_2d, ipm_img)

        axis_bevlines = []
        for bevline in bevlines:
            # pred_in_persformer = np.array([-1 * lane[1], lane[0], lane[2]])
            axis_bevline = np.array([-1 * bevline[:, 0], bevline[:, 1], bevline[:, 2]])
            axis_bevlines.append(axis_bevline.T)

        mask_pred = binary_seg[0].detach().cpu().numpy()
        mask_pred[mask_pred > 0] = 1.0
        # bevlines_img = self.get_bevlines_img(mask_pred, bevlines)
        # points_img = self.get_points_img(mask_gt[0], gt_lanes)
        heat_img = self.get_pred_bev_feat(topdown)
        img_draw_pred = self.draw_pic_2d(img_draw_pred, axis_bevlines, intrins[0], extrinsics[0], post_rots[0],
                                             post_trans[0], pred=True)
        img_draw_gt = self.draw_pic_2d(img_draw_gt, gt_lanes, intrins[0], extrinsics[0], post_rots[0],
                                           post_trans[0], pred=False)

        filepath_2d = self.get_lane_imu_img_2D(axis_gt_lanes, bevlines)
        imu_2d_compare = cv2.imread(filepath_2d)

        filepath_3d = self.get_lane_imu_img_3D(axis_gt_lanes, bevlines)
        imu_3d_compare = cv2.imread(filepath_3d)

        disp_img = self.get_disp_img(img_draw_pred, img_draw_gt, mask_pred[0] * 100, mask_gt[0] * 100, imu_2d_compare, imu_3d_compare)


        cv2.imwrite(visu_path + str(epoch) + '_disp_img_plus.jpg', disp_img)

        # cv2.imwrite(visu_path + str(epoch) + '_mask_pred.jpg', mask_pred[0] * 100)
        #
        # cv2.imwrite(visu_path + str(epoch) + '_mask_gt.jpg', mask_gt[0] * 100)
        #
        # cv2.imwrite(visu_path + str(epoch) + '_img_draw_pred.jpg', img_draw_pred)
        # cv2.imwrite(visu_path + str(epoch) + '_img_draw_gt.jpg', img_draw_gt)
        #
        # cv2.imwrite(visu_path + str(epoch) + '_imu_2d_compare.jpg', imu_2d_compare)
        #
        # cv2.imwrite(visu_path + str(epoch) + '_imu_3d_compare.jpg', imu_3d_compare)
        #
        cv2.imwrite(visu_path + str(epoch) + "_val_heat.png", heat_img)
        # cv2.imwrite(visu_path + str(epoch) + "_val_2d_img.png", ipm_img_show)
        # cv2.imwrite(visu_path + str(epoch) + "ipm_img.png", ipm_img)






















