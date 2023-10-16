import os
import cv2
import math
import numpy as np
from typing import List, Dict
from scipy.interpolate import InterpolatedUnivariateSpline, UnivariateSpline
from mmdet3d.datasets.multiview_datasets.instance_bevlane.mesh_grid import MeshGrid
from mmdet3d.datasets.multiview_datasets.instance_bevlane.time_cost import TimeCost
from mmdet3d.datasets.multiview_datasets.instance_bevlane.generate_affinity_field import GenerateHAFAndVAF
        
        
class OpenLaneDetAnno(object):
    def __init__(self, width_range, depth_range, width_res, depth_res, data_config, max_lanes=20, S=72, laneline_width=1) -> None:
        super().__init__()
        self.tc = TimeCost()
        self.max_lanes = max_lanes
        self.update_grid_params(width_range, depth_range, width_res, depth_res)
        self.mesh = MeshGrid(width_range, depth_range, width_res, depth_res)
        self.color_map = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (100, 255, 0), (100, 0, 255), (255, 100, 0),
            (0, 100, 255), (255, 0, 100), (0, 255, 100),
            (255, 255, 255), (0, 0, 0)
        ]
        self.color_num = len(self.color_map)
        self.engine_af = GenerateHAFAndVAF()

        self.x_min = width_range[0]
        self.x_max = width_range[1]
        self.y_min = depth_range[0]
        self.y_max = depth_range[1]

        self.IMG_ORIGIN_W, self.IMG_ORIGIN_H = data_config['src_size'] #1920 * 1280
        self.input_w, self.input_h = data_config['input_size'] # 960.640
        
        self.n_strips = S - 1
        self.n_offsets = S
        # bev space
        self.y_bev = round((self.y_max-self.y_min)/depth_res)
        self.x_bev = round((self.x_max-self.x_min)/width_res)
        self.strip_size_bev = self.y_bev / self.n_strips
        self.strip_size = self.input_h / self.n_strips
        # y at each x offset
        self.offsets_ys_bev = np.arange(self.y_bev , -1, -self.strip_size_bev)
        self.offsets_ys = np.arange(self.input_h, -1, -self.strip_size)

    def update_grid_params(self, width_range, depth_range, width_res, depth_res):
        self.width_range = width_range
        self.depth_range = depth_range
        self.width_res = width_res
        self.depth_res = depth_res
        self.bev_height = round((self.depth_range[1] - self.depth_range[0]) / self.depth_res)
        self.bev_width = round((self.width_range[1] - self.width_range[0]) / self.width_res)

    def filter_lane(self, lane):
        assert lane[-1][1] <= lane[0][1]
        filtered_lane = []
        used = set()
        for p in lane:
            if p[1] not in used:
                filtered_lane.append(p)
                used.add(p[1])
        return filtered_lane
    
    def get_seg_mask(self, gt_lanes):
        mask_seg = np.zeros((self.input_h, self.input_w, 3))
        ins_id = 0
        for id, line in enumerate(gt_lanes):
            coords = line
            ins_id += 1

            if len(coords) < 1:
                continue
            last_poi_x = 0
            last_poi_y = 0
            for num, pos_pixel in enumerate(coords):
                u, v = pos_pixel[0], pos_pixel[1]
                if num == 0:
                    last_poi_x = int(u)
                    last_poi_y = int(v)
                #draw thickness is attention
                cv2.line(mask_seg, (last_poi_x, last_poi_y),
                            (int(u), int(v)), color=(ins_id, ins_id, ins_id),
                            thickness=1)
                last_poi_x = int(u)
                last_poi_y = int(v)
        return mask_seg

    def get_bev_annotation(self, get_lanes_3d):
        bev_lanes = []
        mask_seg = np.zeros((self.bev_height, self.bev_width, 3), dtype=np.uint8)
        mask_z = np.zeros((self.bev_height, self.bev_width, 1), dtype=np.uint8)
        
        ins_id = 0
        for id, line in enumerate(get_lanes_3d): # 遍历每条车到线
            bev_coords = line
            ins_id += 1
            if len(bev_coords) < 1:
                continue
            bev_lane = []
            last_poi_x = 0
            last_poi_y = 0
            for num, pos_imu in enumerate(bev_coords): # 过去自车坐标系车道线坐标
                u, v = self.mesh.get_index_by_pos(pos_imu[0], pos_imu[1]) # 获取投影到bev空间坐标
                if self.mesh.is_index_outside(u, v): # 判断坐标点是否在设定的bev范围内
                    continue
                z = pos_imu[2] # 高度信息
                bev_lane.append((u, v, z))

                if num == 0:
                    last_poi_x = int(u)
                    last_poi_y = int(v)
                cv2.line(mask_seg, (last_poi_x, last_poi_y),
                            (int(u), int(v)), color=(ins_id, ins_id, ins_id),
                            thickness=1)
                mask_z[v, u, 0] = z

            if len(bev_lane) > 0:
                bev_lanes.append(bev_lane)
        
        label_BEV = self.transform_annotation_bev(mask_seg[...,0], mask_z)
        return label_BEV
                
    def transform_annotation_bev(self, mask_seg, mask_z):

        # 删除掉小于两个点的车道线
        # gt_lanes_3d = filter(lambda x: len(x) > 1, gt_lanes_3d)
        # # 将车道线左边按y坐标降序排列，在bev中为从远到近
        # gt_lanes_3d = [sorted(lane, key=lambda x: -x[1]) for lane in gt_lanes_3d]
        # # 删除有相同y的车道线坐标， 每种距离只保留一个横坐标
        # gt_lanes_3d = [self.filter_lane(lane) for lane in gt_lanes_3d]

        nums_lanes = mask_seg.max()
        # 创建转换后的标注
        lanes = np.ones((self.max_lanes, 2 + 1 + 1 + 1 + self.n_offsets*2),
                        dtype=np.float32) * -1e5  # 2 scores, 1 start_y, 1 start_x, 1 length, S+1 coordinates
        # 车道线初始化为负样本
        lanes[:, 0] = 1
        lanes[:, 1] = 0
        # for lane_idx, lane in enumerate(gt_lanes_3d[:self.max_lanes]): # 遍历每条车道线
        for lane_idx in range(min(nums_lanes, self.max_lanes)):
            try:   # 在车道线上均匀采样点，最多为72个点，起点全部补到底边，会有一部分点到图像外
                xs_outside_image, xs_inside_image, \
                zs_inside_image, zs_outside_image = self.sample_lane_bev_mask(lane_idx, mask_seg, mask_z, self.offsets_ys_bev)
            except AssertionError:
                continue
            if len(xs_inside_image) == 0:
                continue
            all_xs = np.hstack((xs_outside_image, xs_inside_image))
            all_zs = np.hstack((zs_inside_image, zs_outside_image))
            # 将转换后的值填充到转换后的矩阵中
            lanes[lane_idx, 0] = 0
            lanes[lane_idx, 1] = 1
            lanes[lane_idx, 2] = len(xs_outside_image) / self.n_strips
            lanes[lane_idx, 3] = xs_inside_image[0]
            lanes[lane_idx, 4] = len(xs_inside_image)
            lanes[lane_idx, 5:5 + len(all_xs)] = all_xs
            lanes[lane_idx, 5+self.n_offsets: 5+self.n_offsets+len(all_zs)] = all_zs

        return lanes

    def sample_lane_bev(self, points, sample_ys):
        # this function expects the points to be sorted
        points = np.array(points)
        if not np.all(points[1:, 1] < points[:-1, 1]):
            raise Exception('Annotaion points have to be sorted')
        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        # interpolate points inside domain
        assert len(points) > 1
        interp = InterpolatedUnivariateSpline(y[::-1], x[::-1], k=min(3, len(points) - 1))
        # interp = UnivariateSpline(y[::-1], x[::-1], k=min(3, len(points) - 1))
        interp_z = InterpolatedUnivariateSpline(y[::-1], z[::-1], k=min(3, len(points) - 1))
        # interp_z = UnivariateSpline(y[::-1], z[::-1], k=min(3, len(points) - 1))
        domain_min_y = y.min()
        domain_max_y = y.max()
        sample_ys_inside_domain = sample_ys[(sample_ys >= domain_min_y) & (sample_ys <= domain_max_y)] # 取出在车道线高度范围内的采样点
        assert len(sample_ys_inside_domain) > 0
        interp_xs = interp(sample_ys_inside_domain) # 根据车道线平滑曲线公式计算离散采样点的x坐标
        interp_zs = interp_z(sample_ys_inside_domain)

        # extrapolate lane to the bottom of the image with a straight line using the 2 points closest to the bottom
        two_closest_points = points[:min(10, len(points)-1)]
        extrap = np.polyfit(two_closest_points[:, 1], two_closest_points[:, 0], deg=1)
        extrap_z = np.polyfit(two_closest_points[:, 1], two_closest_points[:, 2], deg=1)
        
        extrap_ys = sample_ys[sample_ys > domain_max_y]
        extrap_xs = np.polyval(extrap, extrap_ys)
        extrap_zs = np.polyval(extrap_z, extrap_ys)

        all_xs = np.hstack((extrap_xs, interp_xs))
        all_zs = np.hstack((extrap_zs, interp_zs))

        # separate between inside and outside points
        inside_mask = (all_xs >= 0) & (all_xs < self.x_bev)
        xs_inside_image = all_xs[inside_mask]
        xs_outside_image = all_xs[~inside_mask]
        
        zs_inside_image = all_zs[inside_mask]
        zs_outside_image = all_zs[~inside_mask]

        return xs_outside_image, xs_inside_image, zs_inside_image, zs_outside_image

    def sample_lane_bev_mask(self, lane_idx, mask_seg, mask_z, sample_ys):
        
        y, x = np.where(mask_seg==lane_idx)
        z = mask_z[y, x, 0]

        lane = np.vstack((x, y, z)).T.round().astype(int)
        lane = sorted(lane, key=lambda x: -x[1])
        assert len(lane) > 1
        # 删除有相同y的车道线坐标， 每种高度只保留一个横坐标
        lane = np.array(self.filter_lane(lane))
        assert len(lane) > 1

        domain_min_y = y.min()
        domain_max_y = y.max()
        # sample_ys = list(map(int, sample_ys))
        sample_ys = sample_ys.astype(np.int32)
        sample_ys_inside_domain = sample_ys[(sample_ys >= domain_min_y) & (sample_ys <= domain_max_y)] # 取出在车道线高度范围内的采样点
        # sample_ys_outside_domain = sample_ys[sample_ys > domain_max_y]

        interp_xs = []
        interp_zs = []
        for point in lane:
            if point[1] in sample_ys_inside_domain:
                interp_xs.append(point[0])
                interp_zs.append(point[2])

        closest_points = lane[:min(15, len(lane)-1)]
        extrap = np.polyfit(closest_points[:, 1], closest_points[:, 0], deg=1)
        extrap_z = np.polyfit(closest_points[:, 1], closest_points[:, 2], deg=1)
        
        extrap_ys = sample_ys[sample_ys > domain_max_y]
        extrap_xs = np.polyval(extrap, extrap_ys)
        extrap_zs = np.polyval(extrap_z, extrap_ys)

        all_xs = np.hstack((extrap_xs, interp_xs))
        all_zs = np.hstack((extrap_zs, interp_zs))

        # separate between inside and outside points
        inside_mask = (all_xs >= 0) & (all_xs < self.x_bev)
        xs_inside_image = all_xs[inside_mask]
        xs_outside_image = all_xs[~inside_mask]
        
        zs_inside_image = all_zs[inside_mask]
        zs_outside_image = all_zs[~inside_mask]

        return xs_outside_image, xs_inside_image, zs_inside_image, zs_outside_image

    def transform_annotation(self, gt_lanes_2d):

        # 删除掉小于两个点的车道线
        gt_lanes_2d = filter(lambda x: len(x) > 1, gt_lanes_2d)
        # 将车道线左边按y坐标降序排列，在图像中为从下到上
        # gt_lanes_2d = [sorted(lane, key=lambda x: -x[1]) for lane in gt_lanes_2d]
        # # 删除有相同y的车道线坐标， 每种高度只保留一个横坐标
        # gt_lanes_2d = [self.filter_lane(lane) for lane in gt_lanes_2d]
        # 将坐标缩放到目标尺寸
        gt_lanes_2d = [[[x * self.input_w / float(self.IMG_ORIGIN_W), y * self.input_h / float(self.IMG_ORIGIN_H)] for x, y in lane]
                     for lane in gt_lanes_2d]
        seg_mask = self.get_seg_mask(gt_lanes_2d)[...,0]
        # 创建转换后的标注
        lanes = np.ones((self.max_lanes, 2 + 1 + 1 + 1 + self.n_offsets),
                        dtype=np.float32) * -1e5  # 2 scores, 1 start_y, 1 start_x, 1 length, S+1 coordinates
        # 车道线初始化为负样本
        lanes[:, 0] = 1
        lanes[:, 1] = 0
        for lane_idx, lane in enumerate(gt_lanes_2d[:self.max_lanes]): # 遍历每条车道线
            try:   # 在车道线上均匀采样点，最多为72个点，起点全部补到底边，会有一部分点到图像外
                # xs_outside_image, xs_inside_image = self.sample_lane(lane, self.offsets_ys)
                xs_outside_image, xs_inside_image,ys_outside_domain = self.sample_lane_mask(seg_mask, lane_idx+1, self.offsets_ys)
            except AssertionError:
                continue
            if len(xs_inside_image) == 0:
                continue
            all_xs = np.hstack((xs_outside_image, xs_inside_image))
            # 将转换后的值填充到转换后的矩阵中
            lanes[lane_idx, 0] = 0
            lanes[lane_idx, 1] = 1
            # lanes[lane_idx, 2] = len(xs_outside_image) / self.n_strips
            lanes[lane_idx, 2] = len(ys_outside_domain) / self.n_strips
            lanes[lane_idx, 3] = xs_inside_image[0]
            lanes[lane_idx, 4] = len(xs_inside_image)
            lanes[lane_idx, 5+len(ys_outside_domain):5+len(ys_outside_domain)+len(all_xs)] = all_xs
            # lanes[lane_idx, 5:5+len(all_xs)] = all_xs
        # debug = True
        # if debug:
        #     img = np.zeros((self.input_h, self.input_w, 3), dtype=np.uint8)
        #     targets = lanes[lanes[:, 1] == 1]
        #     for i,target in enumerate(targets):
        #         gt = gt_lanes_2d[i]
        #         target = target[5:]
        #         mask = target>-1e5
        #         points = np.vstack((target[mask], self.offsets_ys[mask])).T.round().astype(int)

        #         for p_curr, p_next in zip(points[:-1], points[1:]):
        #             img = cv2.line(img, tuple(p_curr), tuple(p_next), color=(0, 255, 0), thickness=3)
                

        #         points_gt = np.array(gt).round().astype(int)
        #         for p_curr, p_next in zip(points_gt[:-1], points_gt[1:]):
        #             img = cv2.line(img, tuple(p_curr), tuple(p_next), color=(0, 0, 255), thickness=3)
        #         # for point in points:
        #         #     img = cv2.circle(img, (point[0], point[1]), 3, color=(0, 255, 0), thickness=3)
        #         # points_gt = np.array(gt).round().astype(int)
        #         # for point in points_gt:
        #         #     img = cv2.circle(img, (point[0], point[1]), 3, color=(0, 0, 255), thickness=3)
        #     cv2.imwrite('/root/autodl-tmp/model_log/work_dirs/sample_debug/debug.jpg', img)
        #     img
        return lanes

    def sample_lane(self, points, sample_ys):
        # this function expects the points to be sorted
        points = np.array(points)
        if not np.all(points[1:, 1] < points[:-1, 1]):
            raise Exception('Annotaion points have to be sorted')
        x, y = points[:, 0], points[:, 1]

        # interpolate points inside domain
        assert len(points) > 1
        # interp = InterpolatedUnivariateSpline(y[::-1], x[::-1], k=min(3, len(points) - 1))
        interp = UnivariateSpline(y[::-1], x[::-1], k=min(5, len(points) - 1))
        domain_min_y = y.min()
        domain_max_y = y.max()
        sample_ys_inside_domain = sample_ys[(sample_ys >= domain_min_y) & (sample_ys <= domain_max_y)] # 取出在车道线高度范围内的采样点
        assert len(sample_ys_inside_domain) > 0
        interp_xs = interp(sample_ys_inside_domain) # 根据车道线平滑曲线公式计算离散采样点的x坐标

        # extrapolate lane to the bottom of the image with a straight line using the 2 points closest to the bottom
        two_closest_points = points[:min(15, len(points)-1)]
        # two_closest_points = points[:2]
        extrap = np.polyfit(two_closest_points[:, 1], two_closest_points[:, 0], deg=1)
        extrap_ys = sample_ys[sample_ys > domain_max_y]
        extrap_xs = np.polyval(extrap, extrap_ys)
        all_xs = np.hstack((extrap_xs, interp_xs))

        # separate between inside and outside points
        inside_mask = (all_xs >= 0) & (all_xs < self.input_w)
        xs_inside_image = all_xs[inside_mask]
        xs_outside_image = all_xs[~inside_mask]

        return xs_outside_image, xs_inside_image

    def sample_lane_mask(self, seg_mask, lane_idx, sample_ys):
        y, x = np.where(seg_mask==lane_idx)
        lane = np.vstack((x, y)).T.round().astype(int)
        lane = sorted(lane, key=lambda x: -x[1])
        # 删除有相同y的车道线坐标， 每种高度只保留一个横坐标
        lane = np.array(self.filter_lane(lane))
        assert len(lane) > 1

        domain_min_y = y.min()
        domain_max_y = y.max()
        # sample_ys = list(map(int, sample_ys))
        sample_ys = sample_ys.astype(np.int32)
        sample_ys_inside_domain = sample_ys[(sample_ys >= domain_min_y) & (sample_ys <= domain_max_y)] # 取出在车道线高度范围内的采样点
        sample_ys_outside_domain = sample_ys[sample_ys > domain_max_y]

        interp_xs = []
        for point in lane:
            if point[1] in sample_ys_inside_domain:
                interp_xs.append(point[0])

        # two_closest_points = lane[:min(15, len(lane)-1)]
        # extrap = np.polyfit(two_closest_points[:, 1], two_closest_points[:, 0], deg=1)
        # extrap_ys = sample_ys[sample_ys > domain_max_y]
        # extrap_xs = np.polyval(extrap, extrap_ys)
        # all_xs = np.hstack((extrap_xs, interp_xs))
        all_xs = np.array(interp_xs)

        # separate between inside and outside points
        inside_mask = (all_xs >= 0) & (all_xs < self.input_w)
        xs_inside_image = all_xs[inside_mask]
        xs_outside_image = all_xs[~inside_mask]

        return xs_outside_image, xs_inside_image, sample_ys_outside_domain

    def __call__(self, gt_lanes_3d, gt_lanes_2d):
        '''
            将3d坐标转换到bev空间下的laneATT label格式
            将2d坐标转换到图像空间下的laneATT label格式
        '''
        label_2d = self.transform_annotation(gt_lanes_2d)
        label_bev = self.get_bev_annotation(gt_lanes_3d)
        return label_2d, label_bev

def test_function(frame, engine):
    lanes = engine.get_bev_lanes(frame)
    engine.tc.add_tag("get_bev_lanes end")

    points = engine.get_laneline_points(lanes)
    for lane_id, lane in enumerate(points):
        poi_num = len(lane)
        print('lane id = {}, poi num is {}'.format(lane_id, poi_num))
    engine.tc.add_tag("get_laneline_points end")

    mask_seg, mask_offset_h, mask_offset_v = engine.get_laneline_offset(lanes)
    engine.tc.add_tag("get_laneline_offset end")

    engine.show_img(mask_seg, window_name="mask_seg", pic_name="mask_seg")
    engine.show_img(mask_offset_h, window_name="mask_offset_h", pic_name="mask_offset_h")
    engine.show_img(mask_offset_v, window_name="mask_offset_v", pic_name="mask_offset_v")
    engine.tc.add_tag("show_img end")


if __name__ == '__main__':
   pass

    