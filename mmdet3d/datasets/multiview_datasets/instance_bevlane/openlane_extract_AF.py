import os
import cv2
import math
import numpy as np
from typing import List, Dict
from scipy.interpolate import InterpolatedUnivariateSpline, UnivariateSpline
from mmdet3d.datasets.multiview_datasets.instance_bevlane.mesh_grid import MeshGrid
from mmdet3d.datasets.multiview_datasets.instance_bevlane.time_cost import TimeCost
from mmdet3d.datasets.multiview_datasets.instance_bevlane.generate_affinity_field import GenerateHAFAndVAF


class OpenLaneSegMask(object):
    def __init__(self, width_range, depth_range, width_res, depth_res, data_config, laneline_width=1) -> None:
        super().__init__()
        self.tc = TimeCost()
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
        self.engine_af = GenerateHAFAndVAF()  # 进行编码

        self.x_min = width_range[0]
        self.x_max = width_range[1]
        self.y_min = depth_range[0]
        self.y_max = depth_range[1]

        self.IMG_ORIGIN_W, self.IMG_ORIGIN_H = data_config['src_size'] #1920 * 1280
        self.input_w, self.input_h = data_config['input_size'] #1920 * 1280
        # self.x_min = depth_range[0]
        # self.x_max = depth_range[1]
        # self.y_min = width_range[0]
        # self.y_max = width_range[1]
        self.downsample = 4

    def update_grid_params(self, width_range, depth_range, width_res, depth_res):
        self.width_range = width_range
        self.depth_range = depth_range
        self.width_res = width_res
        self.depth_res = depth_res
        self.bev_height = round((self.depth_range[1] - self.depth_range[0]) / self.depth_res)
        self.bev_width = round((self.width_range[1] - self.width_range[0]) / self.width_res)

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

    def lane_fit(self, gt_lanes, poly_order=3, sample_step=1, interp=True):
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

            mask_idex = (self.x_min <= fit_lane[:, 0]) & (fit_lane[:, 0] <= self.x_max) & (self.y_min <= fit_lane[:, 1]) & (fit_lane[:, 1] <= self.y_max)

            fit_lane = fit_lane[mask_idex]

            if interp:
                fit_lane = self.coords_interpolation(fit_lane)
                fit_lane = np.array(fit_lane)

            fit_lanes.append(fit_lane)

        return fit_lanes

    def get_laneline_offset_bev(self, gt_lanes, gt_category, gt_visibility, draw_type='cv2line'):
        """
        根据gt_lanes获取车道线关键点的偏移量和遮罩图像
        """
        mask_seg = np.zeros((self.bev_height, self.bev_width, 3), dtype=np.uint8)
        mask_offset = np.zeros((self.bev_height, self.bev_width, 2))
        mask_z = np.zeros((self.bev_height, self.bev_width, 1))

        ins_id = 0
        for id, line in enumerate(gt_lanes):
            bev_coords = line
            ins_id += 1

            if len(bev_coords) < 2:
                continue

            last_poi_x = 0
            last_poi_y = 0

            for num, pos_imu in enumerate(bev_coords):
                if self.mesh.is_pos_outside(pos_imu[0], pos_imu[1]):   # 判断坐标是否在范围内
                    continue
                u, v = self.mesh.get_index_by_pos(pos_imu[0], pos_imu[1])   # 将坐标转换为图像索引

                z = pos_imu[2]   # 获取高度信息

                if self.mesh.is_index_outside(u, v):   # 判断索引是否在范围内
                    continue

                offset_x, offset_y = self.mesh.get_offset_with_cell_ld(pos_imu[0], pos_imu[1])   # 获取相对于cell的偏移量

                if draw_type == "cv2circle":
                    # 使用cv2.circle绘制圆点
                    ins_color = self.color_map[ins_id % self.color_num]
                    rgb_value = ins_id * 10
                    color = (rgb_value, rgb_value, rgb_value)
                    cv2.circle(mask_seg, (u, v), radius=1, color=color, thickness=self.laneline_width)
                elif draw_type == "cv2line":
                    if num == 0:
                        last_poi_x = int(u)
                        last_poi_y = int(v)
                    # 使用cv2.line绘制线段
                    cv2.line(mask_seg, (last_poi_x, last_poi_y), (int(u), int(v)), color=(ins_id, ins_id, ins_id), thickness=1)
                    last_poi_x = int(u)
                    last_poi_y = int(v)
                else:
                        # 设置遮罩图像像素的值
                        value = int(ins_id) if ins_id < 255 else 255
                        mask_seg[v, u, :] = value
                        if u > 0 and self.width_res < 0.6:  # 检查索引是否在边界内
                            mask_seg[v, u - 1, :] = value    # 设置前一个索引处的像素值
                        if u + 1 < mask_seg.shape[1] and self.width_res < 0.6:  # 检查索引是否在边界内
                            mask_seg[v, u + 1, :] = value    # 设置后一个索引处的像素值

                mask_offset[v, u, 0] = offset_x   # 设置偏移量X
                mask_offset[v, u, 1] = offset_y   # 设置偏移量Y

                mask_z[v, u, 0] = z   # 设置高度值

        return mask_seg, mask_offset, mask_z

    def get_seg_mask(self, gt_lanes):
        mask_seg = np.zeros((self.input_h//self.downsample, self.input_w//self.downsample, 3))
        ins_id = 0
        for id, line in enumerate(gt_lanes):
            coords = line
            ins_id += 1

            if len(coords) < 2:
                continue
            last_poi_x = 0
            last_poi_y = 0

            for num, pos_pixel in enumerate(coords):
                u, v = (pos_pixel[0]/self.IMG_ORIGIN_W)*(self.input_w//self.downsample),\
                       (pos_pixel[1]/self.IMG_ORIGIN_H)*(self.input_h//self.downsample)

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

    def get_anchorlane_mask(self, gt_lanes, gt_category, gt_visibility, draw_type='cv2line'):
        """
        gt_points fit
        """
        # gt_lanes = self.lane_fit(gt_lanes)
        """
        get lanes keypoints offset and gt_mask
        """
        mask_seg = np.zeros((self.bev_height, self.bev_width, 3), dtype=np.uint8)
        mask_offset = np.zeros((self.bev_height, self.bev_width, 1))
        mask_z = np.zeros((self.bev_height, self.bev_width, 1))
        mask_cls = np.zeros(1, self.bev_width)

        ins_id = 0
        for id, line in enumerate(gt_lanes):
            bev_coords = line
            ins_id += 1

            if len(bev_coords) < 2:
                continue
            last_poi_x = 0
            last_poi_y = 0

            first_x = bev_coords[0][0]
            first_y = bev_coords[0][1]

            u, v = self.mesh.get_index_by_pos(first_x, first_y)
            mask_cls[0, u] = 1

            for num, pos_imu in enumerate(bev_coords):
                if self.mesh.is_pos_outside(pos_imu[0], pos_imu[1]):
                    continue
                u, v = self.mesh.get_index_by_pos(pos_imu[0], pos_imu[1])
                z = pos_imu[2]
                if self.mesh.is_index_outside(u, v):
                    continue
                # offset_x, offset_y = self.mesh.get_offset_with_cell_ld(pos_imu[0], pos_imu[1])
                offset_x = pos_imu[0] - first_x

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

                mask_z[v, u, 0] = z

        return mask_seg, mask_offset, mask_z, mask_cls, gt_lanes

    def show_img(self, img, window_name='test mask', pic_name="714", amp_shape=None, amp_value=None):
        if amp_shape is not None:
            img = cv2.resize(img, (img.shape[1] * amp_shape, img.shape[0] * amp_shape))
        if amp_value is not None:
            img *= amp_value

        cv2.imwrite(os.path.join("./datasets/multiview/bevlane/{}.png".format(pic_name)), img)

        # cv2.imshow(window_name, img)
        # cv2.waitKey(3000)  # keep 100ms
        # cv2.destroyAllWindows()

    def __call__(self, gt_lanes_3d, gt_lanes_2d, gt_category, gt_visibility, source="openlane"):
        '''
        API:
            input: single frame from get_item
            output: binary segment mask
        '''
        mask_seg_bev, mask_offset_bev, mask_z_bev = self.get_laneline_offset_bev(gt_lanes_3d, gt_category, gt_visibility)
        mask_seg_2d = self.get_seg_mask(gt_lanes_2d)

        mask_seg_bev = mask_seg_bev[:, :, 0]
        mask_seg_2d = mask_seg_2d[:, :, 0]

        mask_haf_bev, mask_vaf_bev = self.engine_af(mask_seg_bev)
        mask_haf_2d, mask_vaf_2d = self.engine_af(mask_seg_2d)

        return (mask_seg_bev, mask_haf_bev, mask_vaf_bev, mask_offset_bev, mask_z_bev),\
               (mask_seg_2d,  mask_haf_2d,  mask_vaf_2d)

        # cv2.imwrite('./test_vis/mask_seg.png', mask_seg * 100)
        # # cv2.imwrite('./test_vis/mask_offset_0.png', mask_offset[:, :, 0] * 10000)
        # # cv2.imwrite('./test_vis/mask_offset_1.png', mask_offset[:, :, 1] * 10000)
        # cv2.imwrite("./test_vis/mask_z_.png", -mask_z * 100000)
        # # cv2.imwrite('./test_vis/mask_haf.png', mask_haf * 100)
        # # cv2.imwrite('./test_vis/mask_vaf.png', mask_vaf[:, :, 0] * 100)


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
