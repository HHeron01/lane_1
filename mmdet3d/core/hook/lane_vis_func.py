import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import torch
from .bevlane_post import BevLanePost



def torch_inputs_to_imgs(inputs, mean=[0.48, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    inputs_cpu = np.transpose(inputs.data.cpu().numpy(), [0, 2, 3, 1])
    img = ((inputs_cpu * std * mean) * 255).astype(np.uint8).copy()
    return img

class LaneVisFunc(object):

    def __init__(self, grid_cfg=None, use_offset=False, bev_upsample=1) -> None:
        self.cam_names = ["fl", "fw", "fr", "bl", "bk", "br"]
        self.color_map = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (100, 255, 0), (100, 0, 255), (255, 100, 0),
            (0, 100, 255), (255, 0, 100), (0, 255, 100),
            (255, 255, 255), (0, 0, 0)
        ]
        width_range = [-19.2, 19.2]
        depth_range = [-30., 66.]
        width_res = 0.2
        depth_res = 0.2
        if grid_cfg is not None:
            width_res = grid_cfg.grid_res[0]
            depth_res = grid_cfg.grid_res[1]
            width_range = (grid_cfg.offset[0], grid_cfg.grid_size[0] + grid_cfg.offset[0] - width_res)
            depth_range = (grid_cfg.offset[1], grid_cfg.grid_size[1] + grid_cfg.offset[1] - depth_res)

        width_res = width_res / bev_upsample
        depth_res = depth_res / bev_upsample
        self.bev_upsample = bev_upsample

        self.width_res = width_res
        self.depth_res = depth_res
        self.width_range = width_range
        self.depth_range = depth_range
        self.use_offset = use_offset

        # NOTE: postprocessor need Init
        self.post_engine = BevLanePost(None, width_range, depth_range, width_res, depth_res, use_offset=self.use_offset)
        self.pred_z = -0.3

    # @torchsnooper.snoop()
    def get_cam_imgs(self, inputs):
        """
        inputs is from dataloader single frame, get images from "imgs"
        """
        # NOTE: inputs["imgs"]
        # NOTE: torch size is [batch=8, N=6, T=1, C=3, H=256, W=480]
        # print('get_cam_imgs inputs["imgs"]', inputs["imgs"].shape)

        images = inputs["imgs"]
        imgs_last_time = images[0, :, -1, :, :, :]
        imgs_last_time = torch.squeeze(imgs_last_time)
        imgs_np = torch_inputs_to_imgs(imgs_last_time)

        cam_imgs = {}
        # self.down_sample = 4
        for id, cam in enumerate(self.cam_names):
            img = imgs_np[id]
            # depth = img.shape[0] * self.down_sample
            # width = img.shape[1] * self.down_sample
            # img = cv2.resize(img, (width, depth))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cam_imgs[cam] = img
            # print("cam {} shape {}".format(cam, img.shape))
        return cam_imgs

    def ego_to_cam(self, points, rot, trans, intrins):
        """Transform points (3 x N) from ego frame into a pinhole camera
        """
        points = points - trans.unsqueeze(1)
        points = rot.permute(1, 0).matmul(points)

        points = intrins.matmul(points)
        points[:2] /= points[2:3]

        return points

    def cam_to_ego(self, points, rot, trans, intrins):
        """Transform points (3 x N) from pinhole camera with depth
        to the ego frame
        """
        points = torch.cat((points[:2] * points[2:3], points[2:3]))
        points = intrins.inverse().matmul(points)

        points = rot.matmul(points)
        points += trans.unsqueeze(1)

        return points

    # @torchsnooper.snoop()
    def get_calib(self, inputs):
        k = inputs['k']
        imu2c = inputs['imu2c']
        undist = inputs['undist']
        # NOTE: k torch.Size([batch=8, N=6, T=1, 3, 4])
        # NOTE: imu2c torch.Size([B=8, N=6, T=1, 4, 4])
        # NOTE: undist torch.Size([B=8, N=6, 7])

        cam2pix = {}
        imu2cam = {}
        undistort = {}
        for id, cam in enumerate(self.cam_names):
            K, I2C, D = self.decode_params(id, k, imu2c, undist)
            cam2pix[cam] = K
            imu2cam[cam] = I2C
            undistort[cam] = D
            # print('cam2pix[{}] is {}'.format(cam, K.shape))
            # print('imu2cam[{}] is {}'.format(cam, I2C.shape))
            # print('undistort[{}] is {}'.format(cam, D.shape))
        return cam2pix, imu2cam, undistort

    def decode_params(self, img_id, k, imu2c, undist):
        k = k.numpy()
        K = k[0, img_id, -1, :, :]
        K = K.squeeze()

        imu2c = imu2c.numpy()
        I2C = imu2c[0, img_id, -1, :, :]
        I2C = I2C.squeeze()

        undist = undist.numpy()
        D = undist[0, img_id, :]
        D = D.squeeze()
        return K, I2C, D

    def decode_calib(self, img_id, calibs, k, cam2imu, undist):
        imu2pixel = calibs.numpy()
        imu2pixel = imu2pixel.squeeze(axis=0)
        imu2pixel = imu2pixel[img_id]
        # temp = torch.zeros((3, 1))
        # exp_k = torch.cat((k, temp), dim=1)
        # exp_k.numpy()
        k = k.numpy()
        k = k.squeeze()
        k = k[img_id]
        exp_k = np.zeros_like(imu2pixel)
        exp_k[0, 0] = 1
        exp_k[1, 1] = 1
        exp_k[2, 2] = 1
        exp_k = k @ exp_k
        cam2imu = cam2imu.numpy()
        cam2imu = cam2imu.squeeze()
        cam2imu = cam2imu[img_id]
        imu2c = np.linalg.inv(cam2imu)
        undist = undist.numpy()
        undist = undist.squeeze(axis=0)
        D = undist[img_id]
        return exp_k, imu2c, D

    def get_gt_seg_mask_img(self, inputs):
        target_tensor = inputs['seg_mask']
        target_tensor = target_tensor.unsqueeze(1).repeat(1, 3, 1, 1)
        target = target_tensor.numpy()
        target = np.transpose(target[0], (1, 2, 0))
        target[np.where(target > 0)] = 255
        # print('target_img', target_img[10, :, 0])
        target_img = np.ascontiguousarray(target)
        return target_img

    def get_gt_af_img(self, inputs):
        haf_label_tensor = inputs['haf_mask']
        haf_label_tensor = haf_label_tensor.repeat(3, 1, 1)
        haf_label = haf_label_tensor.numpy()

        vaf_label_tensor = inputs['vaf_mask']
        vaf_label_tensor = vaf_label_tensor[0, 0, :, :].repeat(1, 3, 1, 1)
        vaf_label = vaf_label_tensor.numpy()
        vaf_label = vaf_label.squeeze()

        # haf_label_img = img_trans(haf_label)
        haf_label_img = np.transpose(haf_label, (1, 2, 0))
        # vaf_label_img = img_trans(vaf_label)
        vaf_label_img = np.transpose(vaf_label, (1, 2, 0))
        haf_label_img = np.ascontiguousarray(haf_label_img)
        vaf_label_img = np.ascontiguousarray(vaf_label_img)

        return haf_label_img, vaf_label_img

    def get_gt_offset_hv_img(self, inputs):
        # TODO: this label need to be updated, tentor to numpy
        mask_h = inputs['hoff_mask']
        mask_v = inputs['voff_mask']
        return mask_h, mask_v

    def get_gt_points(self, inputs):
        points = inputs['lines_imu']
        # NOTE: size is [batch_num, lines_num, lines_points_num, 3]
        # NOTE: points[0] mean the first batch
        new_points = []
        for lane_id, coord in enumerate(points[0]):
            new_coord = []
            for poi in coord:
                x = poi[0].item()
                y = poi[1].item()
                z = poi[2].item()
                # TODO: use numpy filter function
                if x > 20 or x < -20 or y > 66 or y < -30 or z < -10 or z > 10:
                    continue
                new_coord.append((x, y, z))
            new_points.append(new_coord)
        return new_points

    def get_pred_seg_mask(self, results, inputs):
        # NOTE: inputs = {'imgs', ... 'seg_mask', ...}
        target_tensor = inputs['seg_mask']
        # NOTE: results = [odhead_preds, lanehead_preds]
        # NOTE: lanehead_pred = [(bin_seg, embedding, haf, vaf, hoff, voff), bev_feat]
        # print("results: ", len(results))
        # NOTE: results: 2, mean odhead_preds and lanehead_pred
        # print("prediction: ", len(results))
        # NOTE: prediction: 2, mean lane pred and bevfeat

        (bin_seg, embedding, haf, vaf, off), bev_feat = results
        pred_hots = torch.argmax(bin_seg, dim=1)
        pred_tensor = torch.zeros_like(target_tensor)
        print("lll:", pred_tensor.shape, pred_hots.shape)
        pred_tensor[(pred_hots == 1)] = 100
        pred_tensor = pred_tensor.unsqueeze(1).repeat(1, 3, 1, 1)
        pred = pred_tensor.numpy()
        pred_img = np.transpose(pred[0], (1, 2, 0))
        # print('pred_img', pred_img[10, :, 0])
        pred_img = np.ascontiguousarray(pred_img)

        return pred_img

    def get_pred_bev_feat(self, results):
        # NOTE: results = [odhead_preds, lanehead_preds]
        # NOTE: lanehead_pred = [(bin_seg, embedding, haf, vaf, hoff, voff), bev_feat]

        (bin_seg, embedding, haf, vaf, off), bev_feat = results
        intermidate = bev_feat
        # intermidate = torch.sum(intermidate, dim=1)
        intermidate = torch.max(intermidate, dim=1)[0]
        intermidate = intermidate.detach().cpu().numpy()
        # print("intermidate:", intermidate.shape)
        heat_img = self.img_trans(intermidate[0])
        # print("heat_img:", heat_img.shape)
        heat_img = np.ascontiguousarray(heat_img)
        return heat_img

    def get_pred_embedding(self, results):
        # NOTE: results = [odhead_preds, lanehead_preds]
        # NOTE: lanehead_pred = [(bin_seg, embedding, haf, vaf, hoff, voff), bev_feat]

        (bin_seg, embedding, haf, vaf, off), bev_feat = results
        # NOTE: embedding torch.Size is ([batch, C=4, H=160, W=64])
        batch_num = 0
        embedding = embedding[batch_num]
        embed_img = torch.max(embedding, dim=0)[0]
        embed_img = embed_img.detach().cpu().numpy()
        # NOTE: embed_img = np.transpose(embed_img, (1, 2, 0))
        embed_img = self.img_trans(embed_img)
        embed_img = np.ascontiguousarray(embed_img)
        return embed_img

    def img_trans(self, features):
        cv2.normalize(features, features, 0., 1., cv2.NORM_MINMAX)
        norm_img = np.asarray(features * 255, dtype=np.uint8)
        out_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)
        return out_img

    def get_resize_img(self, img, new_shape, name="img"):
        # img = np.ascontiguousarray(img)
        img = cv2.resize(img, new_shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.putText(img, name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        return img

    def get_concatenate_bev_img(self, gt_seg_img, points_img, heat_img, embed_img, pred_img, bevlines_img):
        new_shape = (gt_seg_img.shape[1] * 4, gt_seg_img.shape[0] * 4)
        gt_seg_img = self.get_resize_img(gt_seg_img, new_shape, "gt_seg")
        points_img = self.get_resize_img(points_img, new_shape, "points_img")
        heat_img = self.get_resize_img(heat_img, new_shape, "bev_feat")
        embed_img = self.get_resize_img(embed_img, new_shape, "embedding")
        pred_img = self.get_resize_img(pred_img, new_shape, "pred_seg")
        bevlines_img = self.get_resize_img(bevlines_img, new_shape, "bevlines_img")

        disp_bev_pred = np.concatenate((pred_img, heat_img, bevlines_img), axis=1)
        disp_bev_gt = np.concatenate((gt_seg_img, embed_img, points_img), axis=1)

        return disp_bev_pred, disp_bev_gt

    def get_concatenate_bev_img_bak(self, gt_seg_img, points_img, pred_img, bevlines_img):
        new_shape = (gt_seg_img.shape[1] * 4, gt_seg_img.shape[0] * 4)
        gt_seg_img = self.get_resize_img(gt_seg_img, new_shape, "gt_seg")
        points_img = self.get_resize_img(points_img, new_shape, "points_img")
        pred_img = self.get_resize_img(pred_img, new_shape, "pred_seg")
        bevlines_img = self.get_resize_img(bevlines_img, new_shape, "bevlines_img")

        disp_bev_pred = np.concatenate((pred_img, bevlines_img), axis=1)
        disp_bev_gt = np.concatenate((gt_seg_img, points_img), axis=1)

        return disp_bev_pred, disp_bev_gt

    def get_concatenate_cams_img(self, cam_imgs):
        # NOTE: the camera order need check self.cam_names
        img_fl = cam_imgs['fl']
        img_fr = cam_imgs['fr']
        img_fw = cam_imgs['fw']
        img_bk = cam_imgs['bk']
        img_bl = cam_imgs['bl']
        img_br = cam_imgs['br']

        # img_no = np.zeros_like(img_fl)

        def put_text_to_img(img, name="img", scale=1):
            cv2.putText(img, name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0), 1)

        put_text_to_img(img_fl, "FRONT-LEFT")
        put_text_to_img(img_fr, "FRONT-RIGHT")
        put_text_to_img(img_fw, "FRONT-WIDE")
        put_text_to_img(img_bk, "BACK-0922")
        put_text_to_img(img_bl, "BACK-LEFT")
        put_text_to_img(img_br, "BACK-RIGHT")

        disp_line1 = np.concatenate((img_fl, img_fw, img_fr), axis=1)
        disp_line2 = np.concatenate((img_bl, img_bk, img_br), axis=1)
        disp_img = np.concatenate((disp_line1, disp_line2), axis=0)
        return disp_img

    # @torchsnooper.snoop()
    def get_disp_img(self, pred_cams_img, gt_cams_img, disp_bev_pred, disp_bev_gt, bev_eval_img):
        depth = gt_cams_img.shape[0] * 1
        width = gt_cams_img.shape[0] * 2
        disp_bev_pred = cv2.resize(disp_bev_pred, (width, depth))
        disp_bev_gt = cv2.resize(disp_bev_gt, (width, depth))
        disp_bev_eval = cv2.resize(bev_eval_img, (width, depth * 2))

        disp_line1 = np.concatenate((pred_cams_img, disp_bev_pred), axis=1)
        disp_line2 = np.concatenate((gt_cams_img, disp_bev_gt), axis=1)
        disp_img = np.concatenate((disp_line1, disp_line2), axis=0)
        disp_img_plus = np.concatenate((disp_img, disp_bev_eval), axis=1)

        return disp_img_plus

    # @torchsnooper.snoop()
    def gt_projection_img(self, cam_imgs, cam2pix, imu2cam, undistort, gt_points):

        for id, cam in enumerate(self.cam_names):
            c2p = cam2pix[cam]
            i2c = imu2cam[cam]
            D = undistort[cam]
            img = cam_imgs[cam]

            for lane_id, coord in enumerate(gt_points):
                for poi in coord:
                    x = poi[0]
                    y = poi[1]
                    z = poi[2]
                    # fliter poi@imu
                    if x > 20 or x < -20 or y > 66 or y < -30 or z < -10 or z > 10:
                        continue
                    poi_imu = np.array([x, y, z, 1.], dtype=np.float32).reshape(4, 1)
                    poi_cam = i2c @ poi_imu
                    if poi_cam[2] <= 0:
                        continue
                    poi_img = c2p @ poi_cam
                    poi_img = poi_img.squeeze()
                    poi_img /= poi_img[-1]
                    u, v = self.pinhole_distort_point2d(poi_img[0], poi_img[1], c2p, D)
                    u = int(u)
                    v = int(v)
                    if u < 0 or v < 0 or u > img.shape[1] or v > img.shape[0]:
                        continue
                    color_id = lane_id % len(self.color_map)
                    color = self.color_map[color_id]
                    cv2.circle(img, (u, v), 4, color, -1)
        return cam_imgs

    def pred_projection_img(self, cam_imgs, cam2pix, imu2cam, undistort, pred_points):

        for id, cam in enumerate(self.cam_names):
            c2p = cam2pix[cam]
            i2c = imu2cam[cam]
            D = undistort[cam]
            img = cam_imgs[cam]

            for line_id, line in enumerate(pred_points):
                for poi in line['coordinates']:
                    x = poi[0]
                    y = poi[1]
                    z = -0.5
                    poi_imu = np.array([x, y, z, 1.], dtype=np.float32).reshape(4, 1)
                    poi_cam = i2c @ poi_imu
                    if poi_cam[2] <= 0:
                        continue
                    poi_img = c2p @ poi_cam
                    poi_img = poi_img.squeeze()
                    poi_img /= poi_img[-1]
                    u, v = self.pinhole_distort_point2d(poi_img[0], poi_img[1], c2p, D)
                    u = int(u)
                    v = int(v)
                    if u < 0 or v < 0 or u > img.shape[1] or v > img.shape[0]:
                        continue
                    color_id = line_id % len(self.color_map)
                    color = self.color_map[color_id]
                    cv2.circle(img, (u, v), 8, color, -1)
        return cam_imgs

    def points_projection_img(self, cam_imgs, cam2pix, imu2cam, undistort, points,
                              aug_rot, aug_tran, type="gt"):
        # TODO: use numpy function
        for id, cam in enumerate(self.cam_names):
            c2p = cam2pix[cam]
            i2c = imu2cam[cam]
            D = undistort[cam]
            img = cam_imgs[cam]
            rot = aug_rot[cam]
            tran = aug_tran[cam]
            for lane_id, line in enumerate(points):
                if type == "pred":
                    coord = line['coordinates']
                else:
                    coord = line
                for poi in coord:
                    x = poi[0]
                    y = poi[1]
                    z = poi[2] if type == "gt" else self.pred_z
                    # fliter poi@imu
                    if x > 20 or x < -20 or y > 100 or y < -50 or z < -10 or z > 10:
                        continue
                    poi_imu = np.array([x, y, z, 1.], dtype=np.float32).reshape(4, 1)
                    poi_cam = i2c @ poi_imu
                    if poi_cam[2] <= 0:
                        continue
                    poi_img = c2p @ poi_cam
                    # poi_img (3, 1)
                    poi_img = poi_img.squeeze()
                    poi_img /= poi_img[-1]
                    new_poi_img = poi_img
                    cam_model = "pinhole" if D[-1] == 0 else "fisheye"
                    if cam_model == "pinhole":
                        u, v = self.pinhole_distort_point2d(poi_img[0], poi_img[1], c2p, D)
                    else:
                        u, v = self.fisheye_distort_point2d(poi_img[0], poi_img[1], c2p, D)
                    new_poi_img = [u, v, 1]
                    new_poi_img = rot @ new_poi_img
                    new_poi_img += tran
                    u = int(new_poi_img[0])
                    v = int(new_poi_img[1])
                    if u < 0 or v < 0 or u > img.shape[1] or v > img.shape[0]:
                        continue
                    color_id = lane_id % len(self.color_map)
                    color = self.color_map[color_id]
                    cv2.circle(img, (u, v), 2, color, -1)
        return cam_imgs

    def pinhole_distort_point2d(self, u, v, k, D):
        """
        k: intrinsic matirx
        D: undistort coefficient
        """
        cx = k[0, 2]
        cy = k[1, 2]
        fx = k[0, 0]
        fy = k[1, 1]
        # image 2 camera coordinates
        x = (u - cx) / fx
        y = (v - cy) / fy
        r2 = x * x + y * y

        if len(D) == 6:
            k1, k2, p1, p2, k3, _ = D[:]
        elif len(D) == 7:
            k1, k2, k3, p1, p2, k4, _ = D[:]

        # Radial distorsion
        x_dist = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)
        y_dist = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)
        # Tangential distorsion
        x_dist = x_dist + (2 * p1 * x * y + p2 * (r2 + 2 * x * x))
        y_dist = y_dist + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)

        # Back to absolute coordinates.
        x_dist = x_dist * fx + cx
        y_dist = y_dist * fy + cy
        return x_dist, y_dist

    def fisheye_distort_point2d(self, u, v, k, D):
        """
        k: intrinsic matirx
        D: undistort coefficient
        """
        cx = k[0, 2]
        cy = k[1, 2]
        fx = k[0, 0]
        fy = k[1, 1]
        x = (u - cx) / fx
        y = (v - cy) / fy
        r = math.sqrt(x * x + y * y)
        theta = np.arctan(r)
        theta2 = theta ** 2
        if len(D) == 6:
            k1, k2, p1, p2, k3, _ = D[:]
        elif len(D) == 7:
            k1, k2, k3, p1, p2, k4, _ = D[:]

        # radial distortion
        rad_poly = theta * (1 + k1 * theta2 + k2 * theta2 ** 2 +
                            k3 * theta2 ** 3 + k4 * theta2 ** 4) / r
        # distort point coords
        x_dist = x * rad_poly
        y_dist = y * rad_poly
        # distort pixel coords
        x_dist = x_dist * fx + cx
        y_dist = y_dist * fy + cy
        return x_dist, y_dist

    def post_process(self, results):
        # NOTE: results = [odhead_preds, lanehead_preds]
        # NOTE: lanehead_pred = [(bin_seg, embedding, haf, vaf, hoff, voff), bev_feat]

        (bin_seg, embedding, haf, vaf, off), bev_feat = results
        # NOTE: size = [batch_size, C, H, W]
        # NOTE: binary_seg:  torch.Size([1, 2, 160, 64])
        # NOTE: embedding:  torch.Size([1, 4, 160, 64])
        # NOTE: haf:  torch.Size([1, 1, 160, 64])
        # NOTE: vaf:  torch.Size([1, 2, 160, 64])
        binary_seg = bin_seg
        batch_size = binary_seg.shape[0]
        binary_seg = binary_seg.detach().cpu().numpy()
        embedding = embedding.detach().cpu().numpy()
        haf = haf.detach().cpu().numpy()
        vaf = vaf.detach().cpu().numpy()
        # TODO: closed, because techday demo, need to reopen
        # hoff = hoff.detach().cpu().numpy()
        # voff = voff.detach().cpu().numpy()

        for batchi in range(batch_size):
            binary_seg = np.argmax(binary_seg[batchi], axis=0)
            # embedding = embedding[batchi]
            # embedding = np.transpose(embedding[batchi], (1, 2, 0))
            haf = haf[batchi]
            vaf = vaf[batchi]
            haf[0, ...] = binary_seg * haf[0, ...]
            vaf[0, ...] = binary_seg * vaf[0, ...]
            vaf[1, ...] = binary_seg * vaf[1, ...]
            haf = haf[0]
            vaf = vaf.transpose((1, 2, 0))
            # TODO: closed, because techday demo, need to reopen
            # hoff = hoff.detach().cpu().numpy()[batchi]
            # voff = voff.detach().cpu().numpy()[batchi]
            # hoff[0, ...] = binary_seg * hoff[0, ...]
            # voff[0, ...] = binary_seg * voff[0, ...]
            # NOTE: choose init dbscan method
            cluster_method = 'affinity_field'
            bevlines = self.post_engine(binary_seg, haf, vaf, off)
            break
        return bevlines

    def get_points_img(self, gt_img, points):
        img = copy.deepcopy(gt_img)
        width_res = self.width_res
        depth_res = self.depth_res
        width_range = self.width_range
        depth_range = self.depth_range

        for line_id, line in enumerate(points):
            color = self.color_map[line_id % len(self.color_map)]
            for point_imu in line:
                x = point_imu[0] * self.bev_upsample
                y = point_imu[1] * self.bev_upsample
                # TODO: use self param
                u = int((x - width_range[0]) / width_res)
                v = int((depth_range[1] - y) / depth_res)
                # NOTE: color one grid-cell
                cv2.circle(img, (u, v), 1, color)
        return img

    def get_bevlines_img(self, pred_img, bevlines):
        img = copy.deepcopy(pred_img)

        for line_id, line in enumerate(bevlines):
            color = self.color_map[line_id % len(self.color_map)]
            for point in line['bevindexs']:
                u = point[0]
                v = point[1]
                # NOTE: color one grid-cell
                cv2.circle(img, (u, v), 1, color)
        return img

    def get_lane_imu_img(self, points, bevlines):
        filepath = "flint/engine/hooks/bevlane/lane_imu_img.png"
        # print("filepath: " + filepath)
        plt.grid(linestyle='--', color='y', linewidth=0.5)

        wr = self.width_range
        dr = self.depth_range
        corner_x = [wr[0], wr[0], wr[1], wr[1]]
        corner_y = [dr[0], dr[1], dr[0], dr[1]]
        # plt.plot(corner_x, corner_y, 'ro-', color='k')

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
            for poi in line['coordinates']:
                x_data.append(poi[0])
                y_data.append(poi[1])
            plt.plot(x_data, y_data, linestyle=':', color='g', linewidth=1)

        plt.xlabel('X: offset, res=0.6')
        plt.ylabel('Y: distance')
        plt.title("GroundTruth: Blue __;   Prediction: Green ..")
        plt.savefig(filepath)
        plt.cla()
        plt.close()
        return filepath

    def debug_concatenate_bev_img(self, gt_seg_img, points_img, heat_img, embed_img, pred_img):
        new_shape = (gt_seg_img.shape[1] * 4, gt_seg_img.shape[0] * 4)
        gt_seg_img = self.get_resize_img(gt_seg_img, new_shape, "gt_seg")
        points_img = self.get_resize_img(points_img, new_shape, "points_img")
        heat_img = self.get_resize_img(heat_img, new_shape, "bev_feat")
        embed_img = self.get_resize_img(embed_img, new_shape, "embedding")
        pred_img = self.get_resize_img(pred_img, new_shape, "pred_seg")
        disp_bev = np.concatenate((gt_seg_img, points_img, heat_img, embed_img, pred_img), axis=1)
        return disp_bev

    def get_post_aug(self, inputs):
        # NOTE: this function is for getting image augmentation param
        """
        for example:
        # NOTE: post_rot torch.Size([batch=4, N=6, 3, 3])
        post_rot[0] = [[[0.25000, 0.00000, 0.00000],
                        [0.00000, 0.25000, 0.00000],
                        [0.00000, 0.00000, 1.00000]],

                        [[0.50000, 0.00000, 0.00000],
                        [0.00000, 0.50000, 0.00000],
                        [0.00000, 0.00000, 1.00000]],

                        [[0.25000, 0.00000, 0.00000],
                        [0.00000, 0.25000, 0.00000],
                        [0.00000, 0.00000, 1.00000]],

                        [[0.25000, 0.00000, 0.00000],
                        [0.00000, 0.25000, 0.00000],
                        [0.00000, 0.00000, 1.00000]],

                        [[0.25000, 0.00000, 0.00000],
                        [0.00000, 0.25000, 0.00000],
                        [0.00000, 0.00000, 1.00000]],

                        [[0.25000, 0.00000, 0.00000],
                        [0.00000, 0.25000, 0.00000],
                        [0.00000, 0.00000, 1.00000]]]
        # NOTE: post_tran torch.Size([batc=4, N=6, 3])
        post_tran[0] = [[   0.,  -14.,    0.],
                        [-240., -136.,    0.],
                        [   0.,  -14.,    0.],
                        [   0.,  -14.,    0.],
                        [   0.,  -14.,    0.],
                        [   0.,  -14.,    0.]]
        """
        aug_rot = {}
        aug_tran = {}
        for id, cam in enumerate(self.cam_names):
            aug_rot[cam] = inputs['post_rot'][0][id].numpy()
            aug_tran[cam] = inputs['post_tran'][0][id].numpy()
        return aug_rot, aug_tran

    def __call__(self, inputs, results):
        # get ground truth
        target_img = self.get_gt_seg_mask_img(inputs)
        points = self.get_gt_points(inputs)
        points_img = self.get_points_img(target_img, points)

        # get prediction
        heat_img = self.get_pred_bev_feat(results)
        embed_img = self.get_pred_embedding(results)
        pred_img = self.get_pred_seg_mask(results, inputs)

        # NOTE: target_img (160, 64, 3)
        # NOTE: heat_img (160, 64, 3)
        # NOTE: pred_img (160, 64, 3)
        # NOTE: disp_bev = self.debug_concatenate_bev_img(target_img, points_img, heat_img, embed_img, pred_img)

        # postprocessing
        bevlines = self.post_process(results)
        bevlines_img = self.get_bevlines_img(pred_img, bevlines)

        filepath = self.get_lane_imu_img(points, bevlines)
        bev_eval_img = cv2.imread(filepath)

        disp_bev_pred, disp_bev_gt = self.get_concatenate_bev_img(
            target_img, points_img,
            heat_img, embed_img,
            pred_img, bevlines_img)

        # get proj
        cam2pix, imu2cam, undistort = self.get_calib(inputs)
        post_rot, post_tran = self.get_post_aug(inputs)


        # get cam images
        cam_imgs = self.get_cam_imgs(inputs)
        # cams_img_with_gt = self.gt_projection_img(cam_imgs, cam2pix, imu2cam, undistort, points)
        cams_img_with_gt = self.points_projection_img(
            cam_imgs, cam2pix, imu2cam, undistort,
            points, post_rot, post_tran, type="gt")
        cams_img_with_gt = self.get_concatenate_cams_img(cams_img_with_gt)

        cam_imgs2 = self.get_cam_imgs(inputs)
        # cams_img_with_pred = self.pred_projection_img(cam_imgs2, cam2pix, imu2cam, undistort, bevlines)
        cams_img_with_pred = self.points_projection_img(
            cam_imgs2, cam2pix, imu2cam, undistort,
            bevlines, post_rot, post_tran, type="pred")
        cams_img_with_pred = self.get_concatenate_cams_img(cams_img_with_pred)

        # final display image
        disp_img = self.get_disp_img(cams_img_with_pred, cams_img_with_gt, disp_bev_pred, disp_bev_gt, bev_eval_img)

        disp_img = np.ascontiguousarray(disp_img)
        disp_img = cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB)

        return disp_img











