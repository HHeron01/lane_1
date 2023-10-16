import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import torch
import os

import torchvision
from matplotlib.pyplot import MultipleLocator
# from mmdet3d.models.multiview.head.anchor_bevlane_head import LaneATTHead


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


# @HOOKS.register_module()
class Anchor_LaneVisFunc(object):

    def __init__(self, use_offset=True, use_off_z=True) -> None:
        self.cam_names = ["fl", "fw", "fr", "bl", "bk", "br"]
        self.color_map = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (100, 255, 0), (100, 0, 255), (255, 100, 0),
            (0, 100, 255), (255, 0, 100), (0, 255, 100),
            (255, 255, 255), (0, 0, 0), (0, 100, 100)
        ]

        width_range = [-19.2, 19.2]
        depth_range = [0.0, 96.0]
        self.width_res = 0.2
        self.depth_res = 0.3

        self.width_range = width_range
        self.depth_range = depth_range
        self.use_offset = use_offset
        self.use_off_z = use_off_z

        # NOTE: postprocessor need Init
        # self.post_engine = BevLanePost(None, width_range, depth_range, self.width_res, self.depth_res, use_offset=self.use_offset, use_off_z=self.use_off_z)
        self.pred_z = -0.3

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
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img



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


    def draw_pic_2d(self, image, gt_lanes, intrins, extrinsics,
                    post_rots, post_trans, pred=True):
        calib = np.matmul(intrins, extrinsics)
        for line_id, gt_lane in enumerate(gt_lanes):
            color = self.color_map[line_id % len(self.color_map)]
            if pred:
                cv2.putText(image, "PRED", (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                gt_lane = gt_lane['coordinates']
                if len(gt_lane) < 2:
                    continue
                gt_lane = np.array(gt_lane)
                # a = np.ones([gt_lane.shape[0], 1])
                if not self.use_off_z:
                    a = np.full((gt_lane.shape[0], 1), fill_value=self.pred_z)
                    gt_lane = np.concatenate((gt_lane, a), axis=1)
            else:
                if len(gt_lane) < 2:
                    continue
                cv2.putText(image, "GT", (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
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
            for point in line['bevindexs']:
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

        cv2.putText(img, "GT", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)

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

    def get_bevline_points(self, decodeds):
        bevlines = []
        for i, decoded in enumerate(decodeds[0]):
            points = decoded.points.tolist()
            line = {
                'line_id': 0,
                'coordinates': [],
            }
            line['line_id'] = i
            for point in points:
                    x_bev = round(point[0] + self.width_range[0], 3)
                    # x_bev = round(point[0], 3)
                    y_bev = round(point[1], 3)
                    z_bev = round(point[2], 4)
                    # print("x_bev:", x_bev, y_bev, z_bev)
                    line['coordinates'].append([x_bev, y_bev, z_bev])

            bevlines.append(line)
        return bevlines

    def get_gen_anchor_bevline_points(self, decodeds):
        bevlines = []
        for i, decoded in enumerate(decodeds):
            points = decoded.tolist()
            line = {
                'line_id': 0,
                'coordinates': [],
            }
            line['line_id'] = i
            for point in points:
                x_bev = round(point[0], 3)
                y_bev = round(point[1], 3)
                z_bev = round(point[2], 4)
                # print("x_bev:", x_bev, y_bev, z_bev)
                line['coordinates'].append([x_bev, y_bev, z_bev])

            bevlines.append(line)
        return bevlines



    def get_lane_imu_img_2D(self, points, bevlines):
        filepath = "mmdet3d/core/hook/imu_2d_compare.png"
        # print("filepath: " + filepath)
        fig_2d = plt.figure(figsize=(6.4, 6.4))
        plt.grid(linestyle='--', color='y', linewidth=0.5)
        x_major_locator = MultipleLocator(1)
        y_major_locator = MultipleLocator(4)
        ax = plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)
        ax.yaxis.set_major_locator(y_major_locator)

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

    def get_lane_imu_img_3D(self, points, bevlines):
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
        for gt_lane in points:
            if len(gt_lane.shape) < 2:
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
            pred_lane = pred_lane['coordinates']
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
        plt.legend([line1, line2], ['gt', 'pred'], loc=(0.75, 0.7), fontsize=15)
        plt.tick_params(pad=0)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0, 0)
        plt.savefig(filepath)
        plt.cla()
        plt.close()
        return filepath

    def get_disp_img(self, pred_img, gt_img, imu_2d_compare, imu_3d_compare):#, bev_add_xy, bev_add_xyz):
        h, w, c = gt_img.shape

        compare_h, compare_w, _ = imu_2d_compare.shape
        compare_new_shape = (int((h / compare_h)) * compare_w, h)


        imu_2d_compare = cv2.resize(imu_2d_compare, compare_new_shape)
        imu_3d_compare = cv2.resize(imu_3d_compare, compare_new_shape)

        disp_pred = np.concatenate((pred_img, imu_2d_compare), axis=1)
        disp_gt = np.concatenate((gt_img, imu_3d_compare), axis=1)

        disp_img_plus = np.concatenate((disp_pred, disp_gt), axis=0)

        return disp_img_plus


    def __call__(self, runner):

        root_path = runner.work_dir
        # visu_path = os.path.join(root_path, 'vis_pic/')
        visu_path = root_path + '/vis_pic/'
        if not os.path.exists(visu_path):
            os.makedirs(visu_path)
        # get ground truth
        img_metas = runner.data_batch.get('img_metas')

        file_path = img_metas.data[0][0].get('file_path')
        gt_lanes = img_metas.data[0][0].get('origin_lanes')
        img_input = runner.data_batch.get("img_inputs")
        imgs, intrins, extrinsics, post_rots, post_trans, undists, bda_rot, rots, trans, grid, drop_idx = img_input
        epoch = runner.epoch

        net_out = runner.outputs.get('net_out')

        proposals_list, topdown = net_out
        img = self.get_cam_imgs(imgs)
        img_draw_pred = img.copy()
        img_draw_gt = img.copy()

        # pred
        if runner.model.module.bev_lane_head.head_name == 'gen_anchor_head':
            Gen_BevLaneHead = runner.model.module.bev_lane_head
            decodeds = Gen_BevLaneHead.decode(proposals_list[0])
            bevlines = self.get_gen_anchor_bevline_points(decodeds)

        if runner.model.module.bev_lane_head.head_name == 'att_anchor_head':
            LaneATTHead = runner.model.module.bev_lane_head
            # anchor_show = LaneATTHead.draw_anchors(int(LaneATTHead.fmap_w), int(LaneATTHead.fmap_h))
            anchor_show = LaneATTHead.draw_anchors(int(LaneATTHead.img_w), int(LaneATTHead.img_h))
            decodeds = LaneATTHead.decode(proposals_list)
            bevlines = self.get_bevline_points(decodeds)

        heat_img = self.get_pred_bev_feat(topdown)
        img_draw_pred = self.draw_pic_2d(img_draw_pred, bevlines, intrins[0], extrinsics[0], post_rots[0],
                                             post_trans[0], pred=True)
        img_draw_gt = self.draw_pic_2d(img_draw_gt, gt_lanes, intrins[0], extrinsics[0], post_rots[0],
                                           post_trans[0], pred=False)

        filepath_2d = self.get_lane_imu_img_2D(gt_lanes, bevlines)
        imu_2d_compare = cv2.imread(filepath_2d)

        filepath_3d = self.get_lane_imu_img_3D(gt_lanes, bevlines)
        imu_3d_compare = cv2.imread(filepath_3d)

        disp_img = self.get_disp_img(img_draw_pred, img_draw_gt, imu_2d_compare, imu_3d_compare)
        #
        cv2.imwrite(visu_path + str(epoch) + '_disp_img_plus.jpg', disp_img)
        #
        cv2.imwrite(visu_path + str(epoch) + "_val_heat.png", heat_img)

        # cv2.imwrite(visu_path + str(epoch) + "_anchor_show.png", anchor_show)
        

    def vis_2d_result(self, runner):
        root_path = runner.work_dir
        # visu_path = os.path.join(root_path, 'vis_pic/')
        visu_path = root_path + '/vis_pic/'
        if not os.path.exists(visu_path):
            os.makedirs(visu_path)
        # get ground truth
        img_metas = runner.data_batch.get('img_metas')

        file_path = img_metas.data[0][0].get('file_path')
        gt_lanes = img_metas.data[0][0].get('gt_lanes_2d')
        targets = runner.data_batch.get('label_2d')
        img_input = runner.data_batch.get("img_inputs")
        imgs, intrins, extrinsics, post_rots, post_trans, undists, bda_rot, rots, trans, grid, drop_idx = img_input
        epoch = runner.epoch

        net_out = runner.outputs.get('net_out')

        _, _ , topdown = net_out
        img = self.get_cam_imgs(imgs)
        img_draw_pred = img.copy()
        img_draw_target = img.copy()
        img_draw_gt = img.copy()

        LaneATTHead = runner.model.module.lane_head
        anchor_show = LaneATTHead.draw_anchors(int(LaneATTHead.img_w), int(LaneATTHead.img_h))
        decodeds = LaneATTHead.get_decodes(net_out, nms_thres=45, nms_topk=3000, conf_threshold=0.5)
        decodeds_target = LaneATTHead.get_targets_decodes(targets)
        lanes_pred_2d = self.get_lane2d_points(decodeds)
        lanes_targets =self.get_lane2d_points(decodeds_target)

        heat_img = self.get_pred_bev_feat(topdown)
        img_draw_pred = self.draw_pic_2d(img_draw_pred, lanes_pred_2d, pred=True)
        img_draw_target = self.draw_pic_2d(img_draw_target, lanes_targets, target=True)
        img_draw_gt = self.draw_pic_2d(img_draw_gt, gt_lanes, pred=False)

        cv2.imwrite(visu_path + str(epoch) + '_img_pred.jpg', img_draw_pred)
        cv2.imwrite(visu_path + str(epoch) + "_vis_heat.png", heat_img)

        cv2.imwrite(visu_path + str(epoch) + '_anchors.jpg', anchor_show)
        cv2.imwrite(visu_path + str(epoch) + '_img_gt.jpg', img_draw_gt)
        cv2.imwrite(visu_path + str(epoch) + '_img_target.jpg', img_draw_target)
























