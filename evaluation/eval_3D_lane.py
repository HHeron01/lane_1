"""
Evaluation metrics includes:
    Average Precision (AP)
    Max F-scores
    x error close (0 - 40 m)
    x error far (0 - 100 m)
    z error close (0 - 40 m)
    z error far (0 - 100 m)

Author: Linjun Shi
Date: March, 2023
"""

import numpy as np
import cv2
import os
import os.path as ops
import copy
import math
import ujson as json
from scipy.interpolate import interp1d
import matplotlib
from utils import *
from MinCostFlow import SolveMinCostFlow
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (35, 30)
plt.rcParams.update({'font.size': 25})
plt.rcParams.update({'font.weight': 'semibold'})

color = [[0, 0, 255],  # red
         [0, 255, 0],  # green
         [255, 0, 255],  # purple
         [255, 255, 0]]  # cyan

vis_min_y = 5
vis_max_y = 80


class LaneEval(object):
    def __init__(self, args):
        self.x_min = -19.2
        self.x_max = 19.2
        self.y_min = 0
        self.y_max = 96
        self.y_samples = np.linspace(self.y_min, self.y_max, num=100, endpoint=False)
        self.dist_th = 1.5
        self.ratio_th = 0.75
        self.close_range = 40

    def vis_3d(self, gt_lanes):
        z_min = -1
        z_max = 1
        fig = plt.figure(dpi=100, figsize=(12, 6))
        plt.style.use('seaborn-white')
        plt.rc('font', family='Times New Roman', size=10)
        ax = fig.gca(projection='3d')
        line1, line2 = None, None
        plot_lines = {}
        plot_lines["gt"] = []
        for gt_lane in gt_lanes:
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
            line2, = ax.plot(x_g, y_g, z_g, lw=2, c='yellowgreen', alpha=1, label='gt')

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
        if line2:
            plt.legend([line2], ['gt'], loc=(0.75, 0.7), fontsize=15)
        plt.tick_params(pad=0)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0, 0)
        fig.savefig('3d_lane.png')
        return fig

    def vis_2d(self, gt_lanes, P_g2im, raw_file):
        # print("self.dataset_dir:", self.dataset_dir, raw_file)
        path = '/home/slj/Documents/git_workspace/openlane/OpenLane-main/eval/LANE_evaluation/lane3d/example/image/validation/segment-260994483494315994_2797_545_2817_545_with_camera_labels/150723475443834500.jpg'
        # print(ops.join(self.images_dir, raw_file))
        # img = cv2.imread(self.dataset_dir + raw_file)
        img = cv2.imread(path)
        # cv2.imshow("frame", img)
        # img = cv2.imread(path)
        for gt_lane in gt_lanes:
            line_gt = {}
            line_gt["x_3d"] = gt_lane[:, 0].tolist()
            line_gt["y_3d"] = gt_lane[:, 1].tolist()
            line_gt["z_3d"] = gt_lane[:, 2].tolist()

            x_2d, y_2d = self.projective_transformation(P_g2im, gt_lane[:, 0], gt_lane[:, 1], gt_lane[:, 2])
            x_2d = x_2d.astype(np.int32)
            y_2d = y_2d.astype(np.int32)
            # print("x_2d:", x_2d)

            for k in range(1, x_2d.shape[0]):
                img = cv2.line(img, (x_2d[k - 1], y_2d[k - 1]), (x_2d[k], y_2d[k]), (0, 0, 255), 4)
        cv2.imwrite('./img.jpg', img)

    def bench(self, pred_lanes, pred_category, gt_lanes, gt_visibility, gt_category):
        """
            Matching predicted lanes and ground-truth lanes in their IPM projection, ignoring z attributes.
            x error, y_error, and z error are all considered, although the matching does not rely on z
            The input of prediction and ground-truth lanes are in ground coordinate, x-right, y-forward, z-up
            The fundamental assumption is: 1. there are no two points from different lanes with identical x, y
                                              but different z's
                                           2. there are no two points from a single lane having identical x, y
                                              but different z's
            If the interest area is within the current drivable road, the above assumptions are almost always valid.

        :param pred_lanes: N X 2 or N X 3 lists depending on 2D or 3D
        :param gt_lanes: N X 2 or N X 3 lists depending on 2D or 3D
        :param raw_file: file path rooted in dataset folder
        :param gt_cam_height: camera height given in ground-truth data
        :param gt_cam_pitch: camera pitch given in ground-truth data
        :return:
        """
        # change this properly
        close_range_idx = np.where(self.y_samples > self.close_range)[0][0]

        r_lane, p_lane, c_lane = 0., 0., 0.
        x_error_close = []
        x_error_far = []
        z_error_close = []
        z_error_far = []

        # only keep the visible portion
        gt_lanes = [prune_3d_lane_by_visibility(np.array(gt_lane), np.array(gt_visibility[k])) for k, gt_lane in
                    enumerate(gt_lanes)]
        # if 'openlane' in self.dataset_name:
        gt_category = [gt_category[k] for k, lane in enumerate(gt_lanes) if lane.shape[0] > 1]
        gt_lanes = [lane for lane in gt_lanes if lane.shape[0] > 1]

        # only consider those pred lanes overlapping with sampling range
        pred_category = [pred_category[k] for k, lane in enumerate(pred_lanes)
                         if lane[0, 1] < self.y_samples[-1] and lane[-1, 1] > self.y_samples[0]]
        pred_lanes = [lane for lane in pred_lanes if
                      lane[0, 1] < self.y_samples[-1] and lane[-1, 1] > self.y_samples[0]]

        pred_lanes = [prune_3d_lane_by_range(np.array(lane), self.x_min, self.x_max) for lane in pred_lanes]

        pred_category = [pred_category[k] for k, lane in enumerate(pred_lanes) if lane.shape[0] > 1]
        pred_lanes = [lane for lane in pred_lanes if lane.shape[0] > 1]

        # only consider those gt lanes overlapping with sampling range
        gt_category = [gt_category[k] for k, lane in enumerate(gt_lanes)
                       if lane[0, 1] < self.y_samples[-1] and lane[-1, 1] > self.y_samples[0]]
        gt_lanes = [lane for lane in gt_lanes if lane[0, 1] < self.y_samples[-1] and lane[-1, 1] > self.y_samples[0]]

        gt_lanes = [prune_3d_lane_by_range(np.array(lane), self.x_min, self.x_max) for lane in gt_lanes]

        gt_category = [gt_category[k] for k, lane in enumerate(gt_lanes) if lane.shape[0] > 1]
        gt_lanes = [lane for lane in gt_lanes if lane.shape[0] > 1]

        # print("gt_lanes:", len(gt_lanes))

        # _ = self.vis_3d(gt_lanes)
        # _ = self.vis_2d(gt_lanes)

        cnt_gt = len(gt_lanes)
        cnt_pred = len(pred_lanes)

        gt_visibility_mat = np.zeros((cnt_gt, 100))
        pred_visibility_mat = np.zeros((cnt_pred, 100))

        # resample gt and pred at y_samples
        for i in range(cnt_gt):
            min_y = np.min(np.array(gt_lanes[i])[:, 1])
            max_y = np.max(np.array(gt_lanes[i])[:, 1])
            x_values, z_values, visibility_vec = resample_laneline_in_y(np.array(gt_lanes[i]), self.y_samples,
                                                                        out_vis=True)
            gt_lanes[i] = np.vstack([x_values, z_values]).T
            gt_visibility_mat[i, :] = np.logical_and(x_values >= self.x_min, np.logical_and(x_values <= self.x_max,
                                                                                            np.logical_and(
                                                                                                self.y_samples >= min_y,
                                                                                                self.y_samples <= max_y)))
            gt_visibility_mat[i, :] = np.logical_and(gt_visibility_mat[i, :], visibility_vec)

        for i in range(cnt_pred):
            # # ATTENTION: ensure y mono increase before interpolation: but it can reduce size
            # pred_lanes[i] = make_lane_y_mono_inc(np.array(pred_lanes[i]))
            # pred_lane = prune_3d_lane_by_range(np.array(pred_lanes[i]), self.x_min, self.x_max)
            min_y = np.min(np.array(pred_lanes[i])[:, 1])
            max_y = np.max(np.array(pred_lanes[i])[:, 1])
            x_values, z_values, visibility_vec = resample_laneline_in_y(np.array(pred_lanes[i]), self.y_samples,
                                                                        out_vis=True)
            pred_lanes[i] = np.vstack([x_values, z_values]).T
            pred_visibility_mat[i, :] = np.logical_and(x_values >= self.x_min, np.logical_and(x_values <= self.x_max,
                                                                                              np.logical_and(
                                                                                                  self.y_samples >= min_y,
                                                                                                  self.y_samples <= max_y)))
            pred_visibility_mat[i, :] = np.logical_and(pred_visibility_mat[i, :], visibility_vec)
            # pred_visibility_mat[i, :] = np.logical_and(x_values >= self.x_min, x_values <= self.x_max)

        # at least two-points for both gt and pred
        gt_lanes = [gt_lanes[k] for k in range(cnt_gt) if np.sum(gt_visibility_mat[k, :]) > 1]
        gt_category = [gt_category[k] for k in range(cnt_gt) if np.sum(gt_visibility_mat[k, :]) > 1]
        gt_visibility_mat = gt_visibility_mat[np.sum(gt_visibility_mat, axis=-1) > 1, :]
        cnt_gt = len(gt_lanes)

        pred_lanes = [pred_lanes[k] for k in range(cnt_pred) if np.sum(pred_visibility_mat[k, :]) > 1]
        pred_category = [pred_category[k] for k in range(cnt_pred) if np.sum(pred_visibility_mat[k, :]) > 1]
        pred_visibility_mat = pred_visibility_mat[np.sum(pred_visibility_mat, axis=-1) > 1, :]
        cnt_pred = len(pred_lanes)

        adj_mat = np.zeros((cnt_gt, cnt_pred), dtype=int)
        cost_mat = np.zeros((cnt_gt, cnt_pred), dtype=int)
        cost_mat.fill(1000)
        num_match_mat = np.zeros((cnt_gt, cnt_pred), dtype=float)
        x_dist_mat_close = np.zeros((cnt_gt, cnt_pred), dtype=float)
        x_dist_mat_close.fill(1000.)
        x_dist_mat_far = np.zeros((cnt_gt, cnt_pred), dtype=float)
        x_dist_mat_far.fill(1000.)
        z_dist_mat_close = np.zeros((cnt_gt, cnt_pred), dtype=float)
        z_dist_mat_close.fill(1000.)
        z_dist_mat_far = np.zeros((cnt_gt, cnt_pred), dtype=float)
        z_dist_mat_far.fill(1000.)

        # compute curve to curve distance
        for i in range(cnt_gt):
            for j in range(cnt_pred):
                x_dist = np.abs(gt_lanes[i][:, 0] - pred_lanes[j][:, 0])
                z_dist = np.abs(gt_lanes[i][:, 1] - pred_lanes[j][:, 1])

                # apply visibility to penalize different partial matching accordingly
                both_visible_indices = np.logical_and(gt_visibility_mat[i, :] >= 0.5, pred_visibility_mat[j, :] >= 0.5)
                both_invisible_indices = np.logical_and(gt_visibility_mat[i, :] < 0.5, pred_visibility_mat[j, :] < 0.5)
                other_indices = np.logical_not(np.logical_or(both_visible_indices, both_invisible_indices))

                euclidean_dist = np.sqrt(x_dist ** 2 + z_dist ** 2)
                euclidean_dist[both_invisible_indices] = 0
                euclidean_dist[other_indices] = self.dist_th

                # if np.average(euclidean_dist) < 2*self.dist_th: # don't prune here to encourage finding perfect match
                num_match_mat[i, j] = np.sum(euclidean_dist < self.dist_th) - np.sum(both_invisible_indices)
                adj_mat[i, j] = 1
                # ATTENTION: use the sum as int type to meet the requirements of min cost flow optimization (int type)
                # using num_match_mat as cost does not work?
                # make sure cost is not set to 0 when it's smaller than 1
                cost_ = np.sum(euclidean_dist)
                if cost_ < 1 and cost_ > 0:
                    cost_ = 1
                else:
                    cost_ = (cost_).astype(int)
                cost_mat[i, j] = cost_
                # cost_mat[i, j] = np.sum(euclidean_dist)
                # cost_mat[i, j] = num_match_mat[i, j]

                # use the both visible portion to calculate distance error
                if np.sum(both_visible_indices[:close_range_idx]) > 0:
                    x_dist_mat_close[i, j] = np.sum(
                        x_dist[:close_range_idx] * both_visible_indices[:close_range_idx]) / np.sum(
                        both_visible_indices[:close_range_idx])
                    z_dist_mat_close[i, j] = np.sum(
                        z_dist[:close_range_idx] * both_visible_indices[:close_range_idx]) / np.sum(
                        both_visible_indices[:close_range_idx])
                else:
                    x_dist_mat_close[i, j] = -1
                    z_dist_mat_close[i, j] = -1

                if np.sum(both_visible_indices[close_range_idx:]) > 0:
                    x_dist_mat_far[i, j] = np.sum(
                        x_dist[close_range_idx:] * both_visible_indices[close_range_idx:]) / np.sum(
                        both_visible_indices[close_range_idx:])
                    z_dist_mat_far[i, j] = np.sum(
                        z_dist[close_range_idx:] * both_visible_indices[close_range_idx:]) / np.sum(
                        both_visible_indices[close_range_idx:])
                else:
                    x_dist_mat_far[i, j] = -1
                    z_dist_mat_far[i, j] = -1

        # solve bipartite matching vis min cost flow solver
        match_results = SolveMinCostFlow(adj_mat, cost_mat)
        match_results = np.array(match_results)

        # only a match with avg cost < self.dist_th is consider valid one
        match_gt_ids = []
        match_pred_ids = []
        match_num = 0
        if match_results.shape[0] > 0:
            for i in range(len(match_results)):
                if match_results[i, 2] < self.dist_th * self.y_samples.shape[0]:
                    match_num += 1
                    gt_i = match_results[i, 0]
                    pred_i = match_results[i, 1]
                    # consider match when the matched points is above a ratio
                    if num_match_mat[gt_i, pred_i] / np.sum(gt_visibility_mat[gt_i, :]) >= self.ratio_th:
                        r_lane += 1
                        match_gt_ids.append(gt_i)
                    if num_match_mat[gt_i, pred_i] / np.sum(pred_visibility_mat[pred_i, :]) >= self.ratio_th:
                        p_lane += 1
                        match_pred_ids.append(pred_i)
                    if pred_category != []:
                        if pred_category[pred_i] == gt_category[gt_i] or (
                                pred_category[pred_i] == 20 and gt_category[gt_i] == 21):
                            c_lane += 1  # category matched num
                    x_error_close.append(x_dist_mat_close[gt_i, pred_i])
                    x_error_far.append(x_dist_mat_far[gt_i, pred_i])
                    z_error_close.append(z_dist_mat_close[gt_i, pred_i])
                    z_error_far.append(z_dist_mat_far[gt_i, pred_i])
        # # Visulization to be added
        # if vis:
        #     pass
        return r_lane, p_lane, c_lane, cnt_gt, cnt_pred, match_num, x_error_close, x_error_far, z_error_close, z_error_far

    def bench_one_submit(self, pred_dir, gt_dir, test_txt, prob_th=0.5, vis=False):
        pred_lines = open(test_txt).readlines()
        gt_lines = pred_lines

        json_pred = []
        json_gt = []

        print("Loading pred json ...")
        for pred_file_path in pred_lines:
            pred_lines = pred_dir + pred_file_path.strip('\n').replace('jpg', 'json')

            with open(pred_lines, 'r') as fp:
                json_pred.append(json.load(fp))

        print("Loading gt json ...")
        for gt_file_path in gt_lines:
            gt_lines = gt_dir + gt_file_path.strip('\n').replace('jpg', 'json')

            with open(gt_lines, 'r') as fp:
                json_gt.append(json.load(fp))

        if len(json_gt) != len(json_pred):
            raise Exception('We do not get the predictions of all the test tasks')

        gts = {l['file_path']: l for l in json_gt}

        laneline_stats = []
        laneline_x_error_close = []
        laneline_x_error_far = []
        laneline_z_error_close = []
        laneline_z_error_far = []
        for i, pred in enumerate(json_pred):
            if i % 1000 == 0 or i == len(json_pred) - 1:
                print('eval:{}/{}'.format(i + 1, len(json_pred)))
            if 'file_path' not in pred or 'lane_lines' not in pred:
                raise Exception('file_path or lane_lines not in some predictions.')
            raw_file = pred['file_path']

            pred_lanelines = pred['lane_lines']
            pred_lanes = [np.array(lane['xyz']) for i, lane in enumerate(pred_lanelines)]
            pred_category = [int(lane['category']) for i, lane in enumerate(pred_lanelines)]

            if raw_file not in gts:
                raise Exception('Some raw_file from your predictions do not exist in the test tasks.')
            gt = gts[raw_file]

            # evaluate lanelines
            cam_extrinsics = np.array(gt['extrinsic'])
            # Re-calculate extrinsic matrix based on ground coordinate
            R_vg = np.array([[0, 1, 0],
                             [-1, 0, 0],
                             [0, 0, 1]], dtype=float)
            R_gc = np.array([[1, 0, 0],
                             [0, 0, 1],
                             [0, -1, 0]], dtype=float)
            cam_extrinsics[:3, :3] = np.matmul(np.matmul(
                np.matmul(np.linalg.inv(R_vg), cam_extrinsics[:3, :3]),
                R_vg), R_gc)
            gt_cam_height = cam_extrinsics[2, 3]
            gt_cam_pitch = 0

            cam_extrinsics[0:2, 3] = 0.0
            # cam_extrinsics[2, 3] = gt_cam_height

            cam_intrinsics = gt['intrinsic']
            cam_intrinsics = np.array(cam_intrinsics)

            try:
                gt_lanes_packed = gt['lane_lines']
            except:
                print("error 'lane_lines' in gt: ", gt['file_path'])

            gt_lanes, gt_visibility, gt_category = [], [], []
            for j, gt_lane_packed in enumerate(gt_lanes_packed):
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
                lane = np.matmul(cam_extrinsics, np.matmul(cam_representation, lane))
                lane = lane[0:3, :].T

                gt_lanes.append(lane)
                gt_visibility.append(lane_visibility)
                gt_category.append(gt_lane_packed['category'])

            P_g2im = projection_g2im_extrinsic(cam_extrinsics, cam_intrinsics)

            # N to N matching of lanelines
            r_lane, p_lane, c_lane, cnt_gt, cnt_pred, match_num, \
            x_error_close, x_error_far, \
            z_error_close, z_error_far = self.bench(pred_lanes,
                                                    pred_category,
                                                    gt_lanes,
                                                    gt_visibility,
                                                    gt_category,
                                                    )
            laneline_stats.append(np.array([r_lane, p_lane, c_lane, cnt_gt, cnt_pred, match_num]))
            # consider x_error z_error only for the matched lanes
            # if r_lane > 0 and p_lane > 0:
            laneline_x_error_close.extend(x_error_close)
            laneline_x_error_far.extend(x_error_far)
            laneline_z_error_close.extend(z_error_close)
            laneline_z_error_far.extend(z_error_far)

        output_stats = []
        laneline_stats = np.array(laneline_stats)
        laneline_x_error_close = np.array(laneline_x_error_close)
        laneline_x_error_far = np.array(laneline_x_error_far)
        laneline_z_error_close = np.array(laneline_z_error_close)
        laneline_z_error_far = np.array(laneline_z_error_far)

        if np.sum(laneline_stats[:, 3]) != 0:
            R_lane = np.sum(laneline_stats[:, 0]) / (np.sum(laneline_stats[:, 3]))
        else:
            R_lane = np.sum(laneline_stats[:, 0]) / (np.sum(laneline_stats[:, 3]) + 1e-6)  # recall = TP / (TP+FN)
        if np.sum(laneline_stats[:, 4]) != 0:
            P_lane = np.sum(laneline_stats[:, 1]) / (np.sum(laneline_stats[:, 4]))
        else:
            P_lane = np.sum(laneline_stats[:, 1]) / (np.sum(laneline_stats[:, 4]) + 1e-6)  # precision = TP / (TP+FP)
        if np.sum(laneline_stats[:, 5]) != 0:
            C_lane = np.sum(laneline_stats[:, 2]) / (np.sum(laneline_stats[:, 5]))
        else:
            C_lane = np.sum(laneline_stats[:, 2]) / (np.sum(laneline_stats[:, 5]) + 1e-6)  # category_accuracy
        if R_lane + P_lane != 0:
            F_lane = 2 * R_lane * P_lane / (R_lane + P_lane)
        else:
            F_lane = 2 * R_lane * P_lane / (R_lane + P_lane + 1e-6)
        x_error_close_avg = np.average(laneline_x_error_close[laneline_x_error_close > -1 + 1e-6])
        x_error_far_avg = np.average(laneline_x_error_far[laneline_x_error_far > -1 + 1e-6])
        z_error_close_avg = np.average(laneline_z_error_close[laneline_z_error_close > -1 + 1e-6])
        z_error_far_avg = np.average(laneline_z_error_far[laneline_z_error_far > -1 + 1e-6])

        output_stats.append(F_lane)
        output_stats.append(R_lane)
        output_stats.append(P_lane)
        output_stats.append(C_lane)
        output_stats.append(x_error_close_avg)
        output_stats.append(x_error_far_avg)
        output_stats.append(z_error_close_avg)
        output_stats.append(z_error_far_avg)
        output_stats.append(np.sum(laneline_stats[:, 0]))  # 8
        output_stats.append(np.sum(laneline_stats[:, 1]))  # 9
        output_stats.append(np.sum(laneline_stats[:, 2]))  # 10
        output_stats.append(np.sum(laneline_stats[:, 3]))  # 11
        output_stats.append(np.sum(laneline_stats[:, 4]))  # 12
        output_stats.append(np.sum(laneline_stats[:, 5]))  # 13

        return output_stats


if __name__ == '__main__':
    parser = define_args()
    args = parser.parse_args()

    # Prediction results path of your model
    pred_dir = args.pred_dir

    # Data (Annotation) path of OpenLane dataset
    gt_dir = args.dataset_dir

    # Image list file(.txt) which contains relative path of every image
    test_txt = args.test_list

    # Initialize evaluator
    evaluator = LaneEval(args)

    # Evaluation
    eval_stats = evaluator.bench_one_submit(pred_dir, gt_dir, test_txt, prob_th=0.5)

    print("===> Evaluation on validation set: \n"
          "laneline F-measure {:.8} \n"
          "laneline Recall  {:.8} \n"
          "laneline Precision  {:.8} \n"
          "laneline Category Accuracy  {:.8} \n"
          "laneline x error (close)  {:.8} m\n"
          "laneline x error (far)  {:.8} m\n"
          "laneline z error (close)  {:.8} m\n"
          "laneline z error (far)  {:.8} m\n".format(eval_stats[0], eval_stats[1],
                                                     eval_stats[2], eval_stats[3],
                                                     eval_stats[4], eval_stats[5],
                                                     eval_stats[6], eval_stats[7]))

