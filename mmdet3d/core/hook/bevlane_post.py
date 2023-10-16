import math
import numpy as np
import cv2
import copy
import warnings
from .bevlane_af import GenerateAF



class BevLanePost(object):

    warnings.simplefilter('ignore', np.RankWarning)

    def __init__(self, bin_seg=None, width_range=[-19.2, 19.2], depth_range=[-30., 66.],
                 width_res=0.2, depth_res=0.6, use_offset=False, use_off_z=False) -> None:
        if bin_seg is None:
            pass
        else:
            self.bin_seg = bin_seg[0, :]
        self.width_range = width_range
        self.depth_range = depth_range
        self.width_res = width_res
        self.depth_res = depth_res

        self.af_engine = GenerateAF()
        self.color_map = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (100, 255, 0), (100, 0, 255), (255, 100, 0),
            (0, 100, 255), (255, 0, 100), (0, 255, 100),
            (255, 255, 255), (0, 0, 0)
        ]
        self.use_offset = use_offset
        self.use_off_z = use_off_z

    def api(self):
        pass

    def load_bin_seg(self, file_path):
        print('file_path=', file_path)
        self.img = cv2.imread(file_path)
        print('mask shape', self.img.shape)
        self.bin_seg = np.array(self.img[:, :, 0], dtype=np.uint8)

    def get_bin_seg(self):
        return self.bin_seg

    def get_af(self):
        haf, vaf = self.af_engine(self.bin_seg)
        return haf, vaf

    def parse_horizontally(self, binary_map, haf, row, err_thresh=4):
        """
        INPUT:
        this row's bin_seg
            0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0
        get cols by bin_seg
            [4, 5, 6, 7, 12, 13, 14, 15]

        this row's haf
            0 0 0 0 1 1 0 -1 0 0 0 0 1 1 -1 -1 0 0
                    ---> <--         --> <----
                        lane1           lane2
        OUTPUT:
        we need clusters
            [[4, 5, 6, 7], [12, 13, 14, 15]]
        """
        clusters = [[]]
        cols = np.where(binary_map[row, :] > 0)[0] #按行，找到一行中大于0的列的列号
        # print('cols', cols)
        if cols.size > 0:
            prev_col = cols[0] #左到右遍历每一列，prev_col一直记录当前遍历列的前一列
        # print('haf.shape', haf.shape)
        # parse horizontally
        for col in cols:
            # too far away from last point, add new cluster to clusters
            # TODO: distance threshold = err_thresh * width_res = 5 * 0.2 = 1m
            if col - prev_col > err_thresh: #判断当前列与前一列之间，间隔了多少列，设定一个阈值，超过阈值的话就认为不属于同一条车道线
                clusters.append([])  # 如果超过阈值，说明接触到了一条新的车道线实例，是一个新的聚类
                clusters[-1].append(col) # 把这个遇到的新列加入到新类中
                prev_col = col #更新prev_col一直指向当前遍历列的前一列
                continue
            # keep moving to the right, add col to this cluster
            if (haf[row, prev_col] >= 0) and (haf[row, col] >= 0):
                clusters[-1].append(col)
                prev_col = col
                continue
            # found lane center, process vaf, this is center of this lane in the row
            elif (haf[row, prev_col] >= 0) and (haf[row, col] < 0):
                clusters[-1].append(col)
                prev_col = col
                continue
            # found lane end, spawn new lane, add new cluster to clusters
            elif (haf[row, prev_col] < 0) and (haf[row, col] >= 0):
                clusters.append([])
                clusters[-1].append(col)
                prev_col = col
                continue
            # keep moving to the right
            elif (haf[row, prev_col] < 0) and (haf[row, col] < 0):
                clusters[-1].append(col)
                prev_col = col
                continue
        # print('clusters', clusters)
        return clusters

    def get_error_matrix(self, vaf, row, lane_end_pts, clusters):
        # assign existing lanes
        C = np.Inf * np.ones((len(lane_end_pts), len(clusters)), dtype=np.float64)
        # print('vaf.shape', vaf.shape)
        for r, pts in enumerate(lane_end_pts):
            for c, cluster in enumerate(clusters):
                if len(cluster) == 0:
                    continue
                # mean of current cluster
                cluster_mean = np.array([[np.mean(cluster), row]], dtype=np.float32)
                # get vafs from lane end points
                vafs = np.array([vaf[int(round(x[1])), int(round(x[0])), :] for x in pts], dtype=np.float32)
                vafs = vafs / np.linalg.norm(vafs, axis=1, keepdims=True)
                # get predicted cluster center by adding vafs
                pred_points = pts + vafs * np.linalg.norm(pts - cluster_mean, axis=1, keepdims=True)
                # get error between predicted cluster center and actual cluster center
                error = np.mean(np.linalg.norm(pred_points - cluster_mean, axis=1))
                C[r, c] = error
        # print('C', C)
        # print('C', C.shape)
        return C

    def match_and_clustering(self, row, clusters, lane_end_pts, next_lane_id, err_matrix, output, err_thresh):
        """
        INPUT:
            y - 1: clusters
            158: [[4, 5, 6], [12, 13, 14,]]

            y: lane_end_pts
            159: [array[[4, 159], [5, 159], [6, 159]],
                  array[[12, 159], [13, 159], [14, 159]]]
        OUTPUT:
            update_result:
            000...         ...000
            000...111...222...000
        """
        assigned = [False for _ in clusters]
        # print('assigned', assigned)

        # assign cluster to lane (in acsending order of error)
        row_ind, col_ind = np.unravel_index(np.argsort(err_matrix, axis=None), err_matrix.shape)
        for r, c in zip(row_ind, col_ind):
            if err_matrix[r, c] >= err_thresh:
                break
            if assigned[c]:
                continue
            assigned[c] = True
            # update best lane match with current pixel
            output[row, clusters[c]] = r + 1
            lane_end_pts[r] = np.stack((np.array(clusters[c], dtype=np.float32), row * np.ones_like(clusters[c])), axis=1)

        # initialize unassigned clusters to new lanes
        for c, cluster in enumerate(clusters):
            if len(cluster) == 0:
                continue
            if not assigned[c]:
                output[row, cluster] = next_lane_id
                lane_end_pts.append(
                    np.stack((np.array(cluster, dtype=np.float32), row * np.ones_like(cluster)), axis=1))
                next_lane_id += 1

        return output

    def af_cluster_func(self, binary_map, haf, vaf, err_thresh=2, viz=False):
        cluster_result = np.zeros_like(binary_map, dtype=np.uint8)
        lane_end_pts = []
        next_lane_id = 1
        # start decoding from last row to first
        for row in range(binary_map.shape[0] - 1, -1, -1):
            row_clusters = self.parse_horizontally(binary_map, haf, row, err_thresh) # 一行的行聚类
            err_matrix = self.get_error_matrix(vaf, row, lane_end_pts, row_clusters)
            cluster_result = self.match_and_clustering(row, row_clusters, lane_end_pts, next_lane_id, err_matrix, cluster_result, err_thresh)
            # print("cluster_result：", cluster_result.shape)
        if viz:
            im_color = cv2.applyColorMap(40 * cluster_result, cv2.COLORMAP_JET)
            cv2.imwrite(filename="flint/engine/hooks/bevlane/af_cluster.png", img=im_color)

        return cluster_result

    def polyfit2bevlines(self, cluster_result, off, z_off, bev_size=None, sample_pts_num=160, sample_step=1, poly_order=2):
        if bev_size is None:
            bev_size = cluster_result.shape
        h, w = cluster_result.shape
        img_h, img_w = bev_size

        bevlines = []
        cluster_ids = [idx for idx in np.unique(cluster_result) if idx != 0]
        for i in cluster_ids:
            # print("Cluster ID: ", i)
            ys_pred, xs_pred = np.where(cluster_result == i)
            mask = np.bitwise_and(ys_pred > 0, xs_pred >= 0)
            ys_pred = ys_pred[mask]
            xs_pred = xs_pred[mask]
            if (ys_pred.size != xs_pred.size) or (xs_pred.size < 5):#太短的线也不要
                continue

            if self.use_offset:
                off_batch = off[0] #only calculate first image in this batch
                #print(off_batch.shape)
                #off_set_x = off_batch[:, 0, ys_pred, xs_pred]
                #off_set_y = off_batch[:, 1, ys_pred, xs_pred]
                off_set_x = off[0,:, ys_pred, xs_pred][0]
                off_set_y = off[1,:, ys_pred, xs_pred][0]

                off_set_x = off_set_x.detach().cpu().numpy()
                off_set_y = off_set_y.detach().cpu().numpy()
                ys_pred = ys_pred + off_set_y
                xs_pred = xs_pred + off_set_x
                # print("xs_pred:", xs_pred)

            if self.use_off_z:
                z_off_batch = z_off[0][0]  # only calculate first image in this batch
                # print(z_off_batch.shape)
                # print("ys.astype(int), np.floor(xs).astype(int):", xs_filter.astype(int))
                coord_z = z_off_batch[ys_pred.astype(int), xs_pred.astype(int)]
                zs_pred = coord_z

                poly_params_yx = np.polyfit(ys_pred, xs_pred, deg=poly_order)
                poly_params_yz = np.polyfit(ys_pred, zs_pred, deg=poly_order)

                y_min, y_max = np.min(ys_pred), np.max(ys_pred)
                y_min = math.floor(y_min)
                y_max = math.ceil(y_max)
                y_sample = np.array(range(y_min, y_max, sample_step))
                ys_out = np.array(y_sample, dtype=np.float32)

                xs_out = np.polyval(poly_params_yx, ys_out)
                zs_out = np.polyval(poly_params_yz, ys_out)

                xs_out = xs_out[xs_out < w]
                xs_filter = xs_out[xs_out > 0]
                ys_out = ys_out[ys_out < h]
                ys_filter = ys_out[ys_out > 0]

                line = self.generate_3dline(i, xs_filter, ys_filter, zs_out)
            else:
            # NOTE: RankWarning: Polyfit may be poorly conditioned
                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        poly_params = np.polyfit(ys_pred, xs_pred, deg=poly_order)
                    except np.RankWarning:
                        break
                # poly_params = np.polyfit(ys_pred, xs_pred, deg=poly_order)
                y_min, y_max = np.min(ys_pred), np.max(ys_pred)
                # TODO: y threshold is 10, equal to 10 * 0.6m = 6m, the min lane length
                y_threshold = 8
                if y_max - y_min < y_threshold:
                    continue

                y_min = math.floor(y_min)
                # y_max = math.ceil(y_max)
                y_max = math.floor(y_max)
                y_sample = np.array(range(y_min, y_max, sample_step))
                ys = np.array(y_sample, dtype=np.float32)
                xs = np.polyval(poly_params, ys)
                # print("xs_ys:", len(xs), len(ys))
                xs_filter = xs[xs < w]
                xs_filter = xs_filter[xs_filter > 0]
                ys_filter = ys[ys < h]
                ys_filter = ys_filter[ys_filter > 0]

                line = self.generate_line(i, xs_filter, ys_filter)

            # zs = None
            # if self.use_off_z:
            #     z_off_batch = z_off[0][0] #only calculate first image in this batch
            #     # print(z_off_batch.shape)
            #     # print("ys.astype(int), np.floor(xs).astype(int):", xs_filter.astype(int))
            #     coord_z = z_off_batch[ys_filter.astype(int), xs_filter.astype(int)]
            #     zs = coord_z
            #     line = self.generate_3dline(i, xs_filter, ys_filter, zs)
            # else:
            #     line = self.generate_line(i, xs_filter, ys_filter)
            bevlines.append(line)
            
        return bevlines

    def generate_line(self, id, xs, ys):
        left_offset = self.width_range[0]
        front_offset = self.depth_range[1]
        width_res = self.width_res
        depth_res = self.depth_res

        line = {
            'line_id': 0,
            'bevindexs': [],
            'coordinates': [],
        }
        line['line_id'] = id

        if len(xs) == len(ys):
            for n in range(len(ys)):
                x_bev = xs[n]
                y_bev = ys[n]
                # print("x_bev:", x_bev, y_bev)
                line['bevindexs'].append([int(x_bev), int(y_bev)])
                # x_imu_offset = off[:,0,x]
                # y_imu_offset = off[]
                x_imu = left_offset + (x_bev * width_res)
                y_imu = front_offset - (y_bev * depth_res)
                line['coordinates'].append([x_imu, y_imu])
        return line

    def generate_3dline(self, id, xs, ys, zs):
        left_offset = self.width_range[0]
        front_offset = self.depth_range[1]
        width_res = self.width_res
        depth_res = self.depth_res

        line = {
            'line_id': 0,
            'bevindexs': [],
            'coordinates': [],
        }
        line['line_id'] = id
        if len(xs) == len(ys):
            for n in range(len(ys)):
                x_bev = xs[n]
                y_bev = ys[n]
                z_bev = zs[n]
                # print("x_bev:", x_bev, y_bev, z_bev)
                line['bevindexs'].append([int(x_bev), int(y_bev), z_bev])
                x_imu = left_offset + (x_bev * width_res)
                y_imu = front_offset - (y_bev * depth_res)
                z_imu = z_bev
                line['coordinates'].append([x_imu, y_imu, z_imu])
        return line

    def debug_show(self, bevlines):
        img = copy.deepcopy(self.img)
        for line_id, line in enumerate(bevlines):
            color = self.color_map[line_id % len(self.color_map)]
            for point in line['bevindexs']:
                u = point[0]
                v = point[1]
                cv2.circle(img, (u, v), 1, color)
        cv2.imshow("debug_show", img)
        cv2.waitKey(3000)
        cv2.imwrite("flint/engine/hooks/bevlane/pred_bevlines.png", img)

    def __call__(self, bin_seg, haf, vaf, off, z_off):
        # print("visu post: self.width_res", self.width_res)
        cluster_result = self.af_cluster_func(bin_seg, haf, vaf)
        bevlines = self.polyfit2bevlines(cluster_result, off, z_off)
        return bevlines