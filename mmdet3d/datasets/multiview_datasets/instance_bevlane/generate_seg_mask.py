import os
import cv2
import math
import numpy as np
from typing import List, Dict
from mesh_grid import MeshGrid
from time_cost import TimeCost



class Lane3dSegMask(object):
    """
    NOTE:
    Generate laneline label which is BEV instance segmentation mask in imu
    coordinate, that is right-forward-up(RFU).

    e.g.
    bev_grid_map param:
        # WIDTH_RANGE = (-20., 20.)  # unit:m
        # DEPTH_RANGE = (-30., 70.)  # unit:m
        # WIDTH_RES = 0.2  # unit:m
        # DEPTH_RES = 0.5  # unit:m
        # LANELINE_WIDTH = 2  # unit: pixel
    """

    def __init__(self, width_range, depth_range, width_res, depth_res, laneline_width=1) -> None:
        super().__init__()
        self.tc = TimeCost()
        self.proc_categories = ['line', 'road_edge']
        self.line_instance_id_offset = 40
        self.road_edge_instance_id_offset = 120
        self.update_grid_params(width_range, depth_range, width_res, depth_res)
        self.update_laneline_width(laneline_width)
        self.mesh = MeshGrid(width_range, depth_range, width_res, depth_res)
        self.color_map = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (100, 255, 0), (100, 0, 255), (255, 100, 0),
            (0, 100, 255), (255, 0, 100), (0, 255, 100),
            (255, 255, 255), (0, 0, 0)
        ]
        self.color_num = len(self.color_map)

    def update_grid_params(self, width_range, depth_range, width_res, depth_res):
        self.width_range = width_range
        self.depth_range = depth_range
        self.width_res = width_res
        self.depth_res = depth_res
        self.bev_height = round((self.depth_range[1] - self.depth_range[0]) / self.depth_res)
        self.bev_width = round((self.width_range[1] - self.width_range[0]) / self.width_res) - 1
        # self.bev_height = math.floor((self.depth_range[1] - self.depth_range[0]) / self.depth_res)
        # self.bev_width = math.floor((self.width_range[1] - self.width_range[0]) / self.width_res)

    def update_laneline_width(self, laneline_width):
        self.laneline_width = laneline_width

    def query_db(self, db_manager: Ingot, collections: List[str], limit=1000):
        self.db_manager = db_manager
        self.collections = collections

        records = dict()
        for collection in self.collections:
            params = {
                'collection': collection,
                'params': {},
                'keys_to_return': {
                    '_id': True,
                    'meta': True,
                    'payload_path': True,
                    'objects': True
                }
            }
            records[collection] = self.db_manager.query_return_keys(**params, limit=limit)
        return records

    def get_single_frame(self, recodes, index=0):
        # NOTE: first collection, first frame
        collection = self.collections[0]
        frame = recodes[collection][index]
        # test read frame
        objs = frame['objects']
        return frame

    def get_bev_lanes(self, frame):
        objs = frame['objects']
        lines = dict()
        for obj in objs:
            if obj['type'] == 'POLYLINE_3D':
                ins_id = obj['property']['instance_id']
                if ins_id not in lines:
                    lines[ins_id] = []
                lines[ins_id].append(obj)
        line_list = list()
        for ins_id, line_segs in lines.items():
            line_list.append(self.connect_lines(line_segs))
        return line_list
        # lanes = dict()
        # if len(lines) > 0:
        #     lanes = {'lines': line_list}
        # return lanes

    def get_bev_lanes_from_ecarxper(self, frame):
        lines3d_imu = frame['lines3d_imu'][-1]
        lines_property = frame['lines_property'][-1]

        ins_map = dict()
        for id, property in enumerate(lines_property):
            if property['category'] == 'Ground_SolidLane':
                ins_id = property['instance_id']
                if ins_id not in ins_map:
                    ins_map[ins_id] = []
                ins_map[ins_id].append(id)

        line_list = list()
        for ins_id, indexs in ins_map.items():
            new_line = self.connect_lines_from_ecarxper(lines3d_imu, lines_property, indexs)
            if len(new_line['bev_coords']) == 0:
                continue
            line_list.append(new_line)
            line_list[-1]['property']['instance_id'] = ins_id
        return line_list
        # lanes = dict()
        # if len(line_list) > 0:
        #     lanes = {'lines': line_list}
        # return lanes

    def connect_lines(self, line_segments: List):
        bev_line = {
            'type': 'POLYLINE_3D',
            'bev_coords': [],
            'property': {
                'category': None,
                'instance_id': None
            }
        }

        ins_id = line_segments[0]['property']['instance_id']
        bev_coords = []
        for seg in line_segments:
            for imu_pt in seg['coordinate']:
                x = imu_pt[0]
                y = imu_pt[1]
                z = imu_pt[2]
                bev_coords.append([x, y, z])
        bev_line['property']['instance_id'] = ins_id
        bev_line['bev_coords'] = self.coords_interpolation(bev_coords)
        # bev_line['bev_coords'] = bev_coords
        return bev_line

    def connect_lines_from_ecarxper(self, lines3d_imu, lines_property, indexs):
        bev_line = {
            'type': 'POLYLINE_3D',
            'bev_coords': [],
            'property': {
                'category': None,
                'instance_id': None
            }
        }

        bev_coords = []
        for index in indexs:
            for imu_pt in lines3d_imu[index]:
                x = imu_pt[0]
                y = imu_pt[1]
                z = imu_pt[2]
                # TODO use filter function instead
                if y > 300 or y < -300:
                    continue
                if x > 50 or x < -50:
                    continue
                if z > 30 or z < -30:
                    continue
                bev_coords.append([x, y, z])
            bev_line['property']['category'] = lines_property[index]['category']
        # bev_line['bev_coords'] = bev_coords
        bev_line['bev_coords'] = self.coords_interpolation(bev_coords)
        return bev_line

    def linear_interpolation(self, x1, y1, x2, y2):
        """
        function:
            interpolation use linear type
        input:
            [x1, y1] is the first point
            [x2, y2] is the second point
        """
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
        """
        function:
            interpolation use linear type
        input:
            [x1, y1, z1] is the first point
            [x2, y2, z2] is the second point
        """
        num = round((y2 - y1) / self.depth_res)
        if num <= 0:
            return []
        xx = np.linspace(x1, x2, num, endpoint=False)
        yy = np.linspace(y1, y2, num, endpoint=False)
        zz = np.linspace(z1, z2, num, endpoint=False)
        out = []
        for x, y, z in zip(xx, yy, zz):
            out.append([x, y, z])
        return out

    def coords_interpolation(self, bev_coords, type="linear", space="3d"):
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

    def get_laneline_points(self, inputs):
        """
        get lanes points for visu
        """
        points = []
        for line in inputs:
            points.append(line['bev_coords'])
        return points

    def get_laneline_offset(self, inputs, draw_type='matrics'):
        """
        get lanes keypoints offset
        """
        # print("self.bev_heigh:", self.bev_height, self.bev_width)
        mask_seg = np.zeros((self.bev_height, self.bev_width, 3), dtype=np.uint8)
        # mask_offset_h = np.zeros((self.bev_height, self.bev_width, 3), dtype=np.uint8)
        # mask_offset_v = np.zeros((self.bev_height, self.bev_width, 3), dtype=np.uint8)
        mask_offset = np.zeros((self.bev_height, self.bev_width, 2))

        ins_id = 0
        for id, line in enumerate(inputs):
            bev_coords = line['bev_coords']
            category = line['property']['category']
            ins_id = line['property']['instance_id']
            # ins_id += 1

            if len(bev_coords) < 2:
                continue
            last_poi_x = 0
            last_poi_y = 0
            for num, pos_imu in enumerate(bev_coords):
                if self.mesh.is_pos_outside(pos_imu[0], pos_imu[1]):
                    continue
                u, v = self.mesh.get_index_by_pos(pos_imu[0], pos_imu[1])
                if self.mesh.is_index_outside(u, v):
                    continue
                # offset_x, offset_y = self.mesh.get_offset_with_cell_ld(pos_imu[0], pos_imu[1])
                offset_x, offset_y = self.mesh.get_offset_with_cell_center(pos_imu[0], pos_imu[1])

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
                        last_poi_x = int(pos_imu[0])
                        last_poi_y = int(pos_imu[1])
                    cv2.line(mask_seg,
                             (last_poi_x, last_poi_y),
                             (int(pos_imu[0]), int(pos_imu[1])),
                             color=(ins_id, ins_id, ins_id),
                             thickness=1)
                    last_poi_x = int(pos_imu[0])
                    last_poi_y = int(pos_imu[1])
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

        # return mask_seg, mask_offset_h, mask_offset_v
        return mask_seg, mask_offset

    def get_laneline_mask(self, inputs: Dict):
        """
        coords-axis description :
        imu sensor acts as coordinate center, x-axis direction is the same with
        mask coordinate, so do y-axis. that means imu_x_axis -> mask_x_axis,
        imu_y_axis -> -mask_y_axis
        """

        def get_mask_pos(imu_coord, mask_tl_pos):
            # imu coords：[y, x]
            mask_y = math.floor((imu_coord[0] - mask_tl_pos[0]) / self.depth_res)
            mask_x = math.floor((imu_coord[1] - mask_tl_pos[1]) / self.width_res)
            return mask_y, mask_x

        label_mask = np.zeros((self.bev_height, self.bev_width, 3), dtype=np.uint8)
        ext_shape = (10 * self.bev_height, 10 * self.bev_width, 3)
        ext_label_mask = np.zeros(ext_shape, dtype=np.uint8)
        # [y, x],
        mask_tl_pos = [-self.depth_res * (ext_shape[0] / 2), -self.width_res * (ext_shape[1] / 2)]

        for line in inputs['lines']:
            bev_coords = line['bev_coords']
            cat = line['property']['category']
            ins_id = line['property']['instance_id']
            if cat == 'line':
                ins_id += self.line_instance_id_offset
            elif cat == 'road_edge':
                ins_id += self.road_edge_instance_id_offset

            valid_pts = list()
            for imu_pt in bev_coords:
                redef_imu_pt = [-imu_pt[1], imu_pt[0]]
                mask_y, mask_x = get_mask_pos(redef_imu_pt, mask_tl_pos)
                if (mask_y >= 0 and mask_y < ext_shape[0]) and \
                        (mask_x >= 0 and mask_x < ext_shape[1]):
                    valid_pts.append([mask_y, mask_x])

            # draw line
            if len(valid_pts) < 2:
                continue
            for i in range(len(valid_pts) - 1):
                # cv2.line(ext_label_mask,
                #          (valid_pts[i][1], valid_pts[i][0]),
                #          (valid_pts[i + 1][1], valid_pts[i + 1][0]),
                #          color=(ins_id, ins_id, ins_id),
                #          thickness=self.laneline_width)
                cv2.circle(ext_label_mask,
                           (valid_pts[i][1], valid_pts[i][0]),
                           radius=1,
                           color=(0, ins_id, 0),
                           thickness=1)

        label_tl_pos = [-self.depth_range[1], self.width_range[0]]
        tl_y, tl_x = get_mask_pos(label_tl_pos, mask_tl_pos)
        label_mask = ext_label_mask[tl_y: tl_y + label_mask.shape[0],
                     tl_x: tl_x + label_mask.shape[1], :]
        # print('label_mask', label_mask.shape)
        return label_mask, ext_label_mask

    def show_img(self, img, window_name='test mask', pic_name="714", amp_shape=None, amp_value=None):
        if amp_shape is not None:
            img = cv2.resize(img, (img.shape[1] * amp_shape, img.shape[0] * amp_shape))
        if amp_value is not None:
            img *= amp_value

        cv2.imwrite(os.path.join("./flint/datasets/multiview/bevlane/{}.png".format(pic_name)), img)

        # cv2.imshow(window_name, img)
        # cv2.waitKey(3000)  # keep 100ms
        # cv2.destroyAllWindows()

    def __call__(self, frame, source="ecarxper"):
        '''
        API:
            input: single frame from get_item
            output: binary segment mask
        '''
        if source == "ecarxper":
            lanes = self.get_bev_lanes_from_ecarxper(frame)
            points = self.get_laneline_points(lanes)
            # for lane_id, lane in enumerate(points):
            #     poi_num = len(lane)
            #     print('lane id = {}, poi num is {}'.format(lane_id, poi_num))
            # mask_seg, mask_offset_h, mask_offset_v = self.get_laneline_offset(lanes)
            mask_seg, mask_offset = self.get_laneline_offset(lanes)
            # self.show_img(mask_seg, window_name="online_mask_seg", pic_name="online_mask_seg")
            # return points, mask_seg, mask_offset_h, mask_offset_v
            return points, mask_seg, mask_offset
        else:
            lanes = self.get_bev_lanes(frame)
            points = self.get_laneline_points(lanes)
            mask_seg, mask_offset = self.get_laneline_offset(lanes)
            return points, mask_seg, mask_offset
            # mask_seg, mask_offset_h, mask_offset_v = self.get_laneline_offset(lanes)
            # return points, mask_seg, mask_offset_h, mask_offset_v



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
    engine = Lane3dSegMask(
        width_range=(-19.2, 19.2),
        depth_range=(-30., 66.6),
        width_res=0.1,
        depth_res=0.6)
    helix_db_manager = Ingot(**helper.get_helix_db_creds(use_sec_nacos=False, s3_creds="adprd"))

    engine.tc.add_tag("init end")
    # recodes = engine.query_db(db_manager=helix_db_manager, collections=['Task6647'])
    recodes = engine.query_db(db_manager=helix_db_manager, collections=['Task7541'])
    engine.tc.add_tag("query_db end")
    frame = engine.get_single_frame(recodes, index=100)
    engine.tc.add_tag("get_single_frame end")

    test_function(frame, engine)

    engine.tc.add_tag("final")
    engine.tc.print_tag()
    engine.tc.print_time_cost()
