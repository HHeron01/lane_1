import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append('/workspace/lane')
from mmdet3d.models.builder import NECKS


EPSILON = 1e-6

def integral_image(features):
    return torch.cumsum(torch.cumsum(features, dim=-1), dim=-2)

def fisheye_distort_points(undist_points, k, D, flag):
    """Dortion of a set of 2D points based on the lens distort model of fisheye
    camera model.
    Args:
        points ([np.ndarray]): 2D points with shape :math:`(N,2)` or :math:`(B,N,2)`.
        camera_matrix ([np.ndarray]): Intrinsic with shape :math:`(3,3)` or :math:`(B,3,3)`.
        dist_coeffs ([np.ndarray]): distort coefficients with :math:`(k_1,k_2,k_3,k_4])`. \
            This is a vector with 4 elements with shape :math:`(N)` or :math:`(B,N)` .
        lib (module): operator lib, one of numpy and torch, default numpy.
    Returns:
        np.ndarray: distort 2D points with shape :math:`(N,2)` or :math:`(B,N,2)`.
    """
    cx = k[..., 0, 2]
    cy = k[..., 1, 2]
    fx = k[..., 0, 0]
    fy = k[..., 1, 1]
    x = (undist_points[..., 0] - cx) / fx
    y = (undist_points[..., 1] - cy) / fy
    # Dort points
    r = torch.sqrt(x * x + y * y)
    # if r > 2.22045e-16:
    theta = torch.arctan(r)
    theta2 = theta**2
    # radial distortion
    rad_poly = theta * (1 + D[..., 0:1] * theta2 + D[..., 1:2] * theta2 ** 2 +
                        D[..., 2:3] * theta2 ** 3 + D[..., 5:6] * theta2 ** 4) / r
    # distort point coords
    x_dist = x * rad_poly
    y_dist = y * rad_poly
    # distort pixel coords
    x_dist = x_dist * fx + cx
    y_dist = y_dist * fy + cy
    dist_points = torch.stack([x_dist, y_dist], dim=3)
    dist_points = dist_points * flag[..., :-1]
    return dist_points


def pinhole_distort_points(undist_points, k, D, flag):
    """
    k: intrinsic matirx
    D: undistort coefficient k1, k2, k3, p1, p2, k4
    """
    cx = k[..., 0, 2]
    cy = k[..., 1, 2]
    fx = k[..., 0, 0]
    fy = k[..., 1, 1]
    x = (undist_points[..., 0] - cx) / fx
    y = (undist_points[..., 1] - cy) / fy
    k1, k2, k3, p1, p2, k4 = D[..., :-1]
    r2 = x * x + y * y
    # Radial distorsion
    x_dist = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)
    y_dist = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)

    # Tangential distorsion
    x_dist = x_dist + (2 * p1 * x * y + p2 * (r2 + 2 * x * x))
    y_dist = y_dist + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)

    # Back to absolute coordinates.
    x_dist = x_dist * fx + cx
    y_dist = y_dist * fy + cy
    dist_points = torch.stack([x_dist, y_dist], dim=3)
    dist_points = dist_points * flag[..., :-1]
    return dist_points
#这是把三维投影到二维的
def perspective(matrix, vector):
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

@NECKS.register_module()
class OFT(nn.Module):
    def __init__(self, grid_res, grid_height, scale=1, add_coord=True, debug=False, **kwargs):
        super().__init__()

        y_corners = -torch.arange(0, grid_height, grid_res[2]) + grid_height / 2.
        y_corners = F.pad(y_corners.view(-1, 1, 1, 1), [1, 1])

        self.register_buffer('y_corners', y_corners)
        self.height_channel = len(y_corners) - 1
        self.scale = scale
        self.debug = debug
        self.add_coord = add_coord

    def forward(self, features, ks, imu2cs, post_rots, post_trans, undists, **kwargs):
        orthos = []
        grid = kwargs['grid']
        drop_idx = kwargs['drop_idx']
        neck = kwargs['neck']

        corners = grid.unsqueeze(1) + self.y_corners.view(-1, 1, 1, 3)
        for feature, k, imu2c, post_rot, post_tran, undist, corner in zip(features, ks, imu2cs, post_rots, post_trans,
                                                                          undists, corners):
            voxel_feats = []
            calib = torch.matmul(k, imu2c)
            num_img, channel, img_height, img_width = feature.size()
            for i in range(num_img):
                if i in drop_idx:
                    continue
                img_corners, flag = perspective(calib[i, :, :], corner)
                if undist is not None:
                    if undist[i, -1] == 1:
                        img_corners = fisheye_distort_points(img_corners, k[i, :, :], undist[i, :], flag)
                    else:
                        img_corners = pinhole_distort_points(img_corners, k[i, :, :], undist[i, :], flag)

                # Normalize to [-1, 1]
                img_sacle = corners.new([self.scale, self.scale])
                img_corners = torch.matmul(post_rot[i, :2, :2], img_corners.unsqueeze(-1)
                                           ) + post_tran[i, :2].unsqueeze(1)
                norm_corners = img_corners.squeeze(-1) / img_sacle
                norm_corners = (2 * norm_corners - 1).clamp(-1, 1)
                # Get top-left and bottom-right coordinates of voxel bounding boxes
                bbox_corners = torch.cat([
                    torch.min(norm_corners[:-1, :-1, :-1],
                              norm_corners[:-1, 1:, :-1]),
                    torch.max(norm_corners[1:, 1:, 1:],
                              norm_corners[1:, :-1, 1:])
                ], dim=-1)
                norm_corners = norm_corners.unsqueeze(-1)
                hori = torch.cat(
                    [norm_corners[:-1, :-1, :-1, 0], norm_corners[:-1, :-1, 1:, 0], norm_corners[:-1, 1:, :-1, 0],
                     norm_corners[:-1, 1:, 1:, 0],
                     norm_corners[1:, :-1, :-1, 0], norm_corners[1:, :-1, 1:, 0], norm_corners[1:, 1:, :-1, 0],
                     norm_corners[1:, 1:, 1:, 0]], dim=-1)
                vert = torch.cat(
                    [norm_corners[:-1, :-1, :-1, 1], norm_corners[:-1, :-1, 1:, 1], norm_corners[:-1, 1:, :-1, 1],
                     norm_corners[:-1, 1:, 1:, 1],
                     norm_corners[1:, :-1, :-1, 1], norm_corners[1:, :-1, 1:, 1], norm_corners[1:, 1:, :-1, 1],
                     norm_corners[1:, 1:, 1:, 1]], dim=-1)
                left = torch.min(hori, dim=-1, keepdim=True)[0]
                right = torch.max(hori, dim=-1, keepdim=True)[0]
                top = torch.min(vert, dim=-1, keepdim=True)[0]
                bot = torch.max(vert, dim=-1, keepdim=True)[0]
                bbox_corners = torch.cat([left, top, right, bot], dim=-1)
                _, depth, width, _ = bbox_corners.size()
                bbox_corners = bbox_corners.flatten(1, 2)
                if self.debug:
                    bbox_corners_c = bbox_corners.cpu().numpy()
                    np.save('./bbox_corners_' + str(1 / self.scale) + '_' + str(i) + '.npy', bbox_corners_c)

                # Compute the area of each bounding box
                area = ((bbox_corners[..., 2:] - bbox_corners[..., :2]).prod(dim=-1) *
                        img_height * img_width * 0.25 + EPSILON).unsqueeze(0)
                visible = (area > EPSILON)

                # Sample integral image at bounding box locations
                integral_img = integral_image(feature[i, :, :, :])
                if self.debug:
                    integral_img_c = integral_img.cpu().numpy()
                    np.save('./integral_img_' + str(i) + '.npy', integral_img_c)
                top_left = F.grid_sample(integral_img.unsqueeze(0), bbox_corners[..., [0, 1]].unsqueeze(0),
                                         padding_mode="border", align_corners=True)
                btm_right = F.grid_sample(integral_img.unsqueeze(0), bbox_corners[..., [2, 3]].unsqueeze(0),
                                          padding_mode="border", align_corners=True)
                top_right = F.grid_sample(integral_img.unsqueeze(0), bbox_corners[..., [2, 1]].unsqueeze(0),
                                          padding_mode="border", align_corners=True)
                btm_left = F.grid_sample(integral_img.unsqueeze(0), bbox_corners[..., [0, 3]].unsqueeze(0),
                                         padding_mode="border", align_corners=True)

                # Compute voxel features (ignore features which are not visible)
                vox_feat = (top_left + btm_right - top_right - btm_left)
                vox_feat = vox_feat / area
                vox_feat = vox_feat.squeeze(0) * visible.float()
                if neck == "3d":
                    vox_feat = vox_feat.view(channel, self.height_channel, depth, width).permute(0, 2, 3, 1)
                else:
                    vox_feat = vox_feat.view(channel * self.height_channel, depth, width)
                voxel_feats.append(vox_feat)
            ortho = torch.stack(voxel_feats)
            ortho = torch.max(ortho, 0)[0]
            orthos.append(ortho)
        orthos = torch.stack(orthos)
        return orthos


@NECKS.register_module()
class OFTV2(nn.Module):

    def __init__(self, grid_res, grid_height, scale=1, add_coord=True, debug=False, **kwargs):
        super().__init__()

        y_corners = -torch.arange(0, grid_height, grid_res[2]) + grid_height / 2.
        y_corners = F.pad(y_corners.view(-1, 1, 1, 1), [1, 1])

        self.register_buffer('y_corners', y_corners)
        self.height_channel = len(y_corners) - 1
        self.scale = scale
        self.debug = debug
        self.add_coord = add_coord

    def forward(self, features, ks, imu2cs, post_rots, post_trans, undists, **kwargs):
        orthos = []
        grid = kwargs['grid']
        drop_idx = kwargs['drop_idx']
        neck = kwargs['neck']
        corners = grid.unsqueeze(1)[:, :, :-1, :-1] + self.y_corners[:-1].view(-1, 1, 1, 3)
        for feature, k, imu2c, post_rot, post_tran, undist, corner in zip(features, ks, imu2cs, post_rots, post_trans,
                                                                          undists, corners):
            voxel_feats = []
            calib = torch.matmul(k, imu2c)
            num_img, channel, img_height, img_width = feature.size()
            for i in range(num_img):
                if i in drop_idx:
                    continue
                img_corners, flag = perspective(calib[i, :, :], corner)
                if undist is not None:
                    if undist[i, -1] == 1:
                        img_corners = fisheye_distort_points(img_corners, k[i, :, :], undist[i, :], flag)
                    else:
                        img_corners = pinhole_distort_points(img_corners, k[i, :, :], undist[i, :], flag)
                # Normalize to [-1, 1]
                img_sacle = corners.new([self.scale, self.scale])
                img_corners = torch.matmul(post_rot[i, :2, :2], img_corners.unsqueeze(-1)
                                           ) + post_tran[i, :2].unsqueeze(1)
                norm_corners = img_corners.squeeze(-1) / img_sacle
                norm_corners = (2 * norm_corners - 1).clamp(-1, 1)
                # Get top-left and bottom-right coordinates of voxel bounding boxes
                _, depth, width, _ = norm_corners.size()
                bbox_corners = norm_corners.flatten(1, 2)
                visible = (bbox_corners[..., 0] > -1) & (bbox_corners[..., 0] < 1) & (bbox_corners[..., 1] < 1) & (
                            bbox_corners[..., 1] > -1)
                visible = visible.unsqueeze(0)

                if self.debug:
                    bbox_corners_c = bbox_corners.view(self.height_channel, depth, width, 2).cpu().numpy()
                    np.save('./bbox_corners_' + str(1 / self.scale) + '_' + str(i) + '.npy', bbox_corners_c)
                    img_corners_c = img_corners.view(self.height_channel, depth, width, 2).cpu().numpy()
                    np.save('./img_corners_' + str(1 / self.scale) + '_' + str(i) + '.npy', img_corners_c)
                    visible_c = visible.view(self.height_channel, depth, width).cpu().numpy()
                    np.save('./visible_' + str(i) + '.npy', visible_c)
                    # Compute the area of each bounding box
                vox_feat = F.grid_sample(feature[i, :, :, :].unsqueeze(
                    0), bbox_corners.unsqueeze(0), padding_mode="border", align_corners=True)
                vox_feat = vox_feat.squeeze(0) * visible.float()
                if neck == "3d":
                    vox_feat = vox_feat.view(channel, self.height_channel, depth, width).permute(0, 2, 3, 1)
                else:
                    vox_feat = vox_feat.view(channel * self.height_channel, depth, width)
                voxel_feats.append(vox_feat)
            ortho = torch.stack(voxel_feats)
            ortho = torch.max(ortho, 0)[0]
            orthos.append(ortho)
        orthos = torch.stack(orthos)
        return orthos

@NECKS.register_module()
class OFTV3(nn.Module):
    def __init__(self, grid_res, grid_height, scale=1, add_coord=False, debug=False, **kwargs):
        super().__init__()

        y_corners = -torch.arange(0, grid_height, grid_res[2]) + grid_height / 2.
        y_corners = F.pad(y_corners.view(-1, 1, 1, 1), [2, 0])


        self.register_buffer('y_corners', y_corners)
        self.height_channel = len(y_corners) - 1
        self.scale = scale
        self.debug = debug
        self.z_corners = y_corners.cuda()
        self.add_coord = add_coord
        # print("self.add_coord:", self.add_coord)

    def forward(self, features, ks, imu2cs, post_rots, post_trans, undists, **kwargs):
    # def forward(self, input):
        orthos = []
        grid = kwargs['grid']
        drop_idx = kwargs['drop_idx']
        neck = kwargs['neck']
        corners = grid.unsqueeze(1)[:, :, :-1, :-1] + self.z_corners[:-1].view(-1, 1, 1, 3)
        for feature, k, undist, imu2c, post_rot, post_tran, corner in zip(features, ks, undists, imu2cs, post_rots, post_trans, corners):
            voxel_feats = []
            calib = torch.matmul(k, imu2c)  # 校准矩阵
            num_img, channel, img_height, img_width = feature.size() 
            for i in range(num_img):
                if i in drop_idx:
                    continue
                img_corners, flag = perspective(calib[i, :, :], corner) # 将三维坐标变换成二维坐标
                if undist is not None:# 这里的畸变参数
                    if undist[i, -1] == 1:
                        img_corners = fisheye_distort_points(img_corners, k[i, :, :], undist[i, :], flag)
                    else:
                        img_corners = pinhole_distort_points(img_corners, k[i, :, :], undist[i, :], flag)
                img_sacle = corners.new([self.scale, self.scale])
                img_corners = torch.matmul(post_rot[i, :2, :2], img_corners.unsqueeze(-1)
                                           ) + post_tran[i, :2].unsqueeze(1)
                norm_corners = img_corners.squeeze(-1) / img_sacle   # 这里是平移，旋转矩阵
                height, depth, width, _ = norm_corners.size()   # 
                bbox_corners = norm_corners.view(-1, 2).long()
                visible = (bbox_corners[..., 0] > 0) & (bbox_corners[..., 0]
                                                        < img_width) & (bbox_corners[..., 1] < img_height) & (
                                      bbox_corners[..., 1] > 0)

                if self.debug:
                    # bbox_corners_c = norm_corners.view(self.height_channel, dep2th, width, 2).cpu().numpy()
                    bbox_corners_c = bbox_corners.float().cpu().numpy()
                    np.save('./bbox_corners_' + str(1 / self.scale) + '_' + str(i) + '.npy', bbox_corners_c)
                    img_corners_c = img_corners.view(self.height_channel, depth, width, 2).cpu().numpy()
                    np.save('./img_corners_' + str(1 / self.scale) + '_' + str(i) + '.npy', img_corners_c)
                    visible_c = visible.view(self.height_channel, depth, width).cpu().numpy()
                    np.save('./visible_' + str(i) + '.npy', visible_c)
                    # Compute the area of each bounding box
                vox_feat = torch.zeros((channel, height * depth * width), device=feature.device)
                vox_feat[:, visible] = feature[i, :, bbox_corners[visible, 1], bbox_corners[visible, 0]]
                # Flatten to orthographic feature map
                if neck == "3d":
                    if self.add_coord:
                        corners_t = corner.permute(3, 0, 1, 2)
                        vox_feat = vox_feat.view(channel, self.height_channel, depth, width)
                        vox_feat = torch.cat((vox_feat, corners_t), 0)
                        vox_feat = vox_feat.permute(0, 2, 3, 1)
                    else:
                        vox_feat = vox_feat.view(channel, self.height_channel, depth, width).permute(0, 2, 3, 1)
                else:
                    if self.add_coord:
                        corners_t = corner.permute(3, 0, 1, 2)
                        vox_feat = vox_feat.view(channel, self.height_channel, depth, width)
                        vox_feat = torch.cat((vox_feat, corners_t), 0)
                        vox_feat = vox_feat.view((channel + 3) * self.height_channel, depth, width)
                    else:
                        vox_feat = vox_feat.view(channel * self.height_channel, depth, width)
                voxel_feats.append(vox_feat)
            ortho = torch.stack(voxel_feats)
            ortho = torch.max(ortho, 0)[0]  # 获取概率最大的
            orthos.append(ortho)
        orthos = torch.stack(orthos)
        return orthos