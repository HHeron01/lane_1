import numpy as np
import os
import string
import json
import math
import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

# camera_model_param:
# cx: 959.3435349245617  # 相机主点cx
# cy: 548.5132971653981  # 相机主点cy
# fx: 932.1969518276043  # 相机焦距fx
# fy: 932.1969518276043  # 相机焦距fy
# k1: 0.2998493816933048  # 径向畸变系数k1 / 等距模型k1
# k2: -0.0198318282342113  # 径向畸变系数k2 / 等距模型k2
# k3: 0.00108686964197308  # 径向畸变系数k3 / 等距模型k3
# k4: 0.6510364285533242  # 径向畸变系数k4 / 等距模型k4
# p1: 0.000339452904797898  # 切向畸变系数p1 (if model_type == radtan)
# p2: -3.395515874104971e-05  # 切向畸变系数p2 (if model_type == radtan)


"""
径向畸变（Radial Distortion）：径向畸变是由于镜头的形状或物理特性引起的。它分为桶形畸变（barrel distortion）和枕形畸变（pincushion distortion）两种类型。
桶形畸变导致图像中心附近的直线向外弯曲，使物体看起来变短。这种畸变的径向畸变系数为正。
枕形畸变导致图像中心附近的直线向内弯曲，使物体看起来变长。这种畸变的径向畸变系数为负。
径向畸变的校正公式一般可以由多项式函数表示，其中包括径向畸变系数和径向距离的幂次。去畸变操作通过应用这些校正公式，在图像上对每个像素进行校正，以恢复直线的直线性质。
切向畸变（Tangential Distortion）：切向畸变是由于镜头与图像平面不完全平行引起的。它会导致图像中的线条在沿着水平或垂直方向产生弯曲。这种畸变由切向畸变系数来描述。
切向畸变的校正公式通常包括对应于x和y方向的切向畸变系数，以及与x和y坐标相关的径向畸变项。去畸变操作通过这些校正公式，对图像上的像素进行相应的校正。
"""
def pinhole_distort_points(undist_points):
    """
    k: 内参矩阵
    D: 去畸变系数 k1, k2, k3, p1, p2, k4
    """
    cx = 953.6932983398438  # 相机光心在图像上的x坐标
    cy = 551.2189331054688  # 相机光心在图像上的y坐标
    fx = 931.8326416015625  # 相机的焦距x
    fy = 931.8326416015625  # 相机的焦距y

    undist_points = undist_points.T  # 将输入的点云坐标转置（每个点的坐标按列排列）
    x = (undist_points[:, 0] - cx) / fx  # 计算x坐标的归一化坐标值
    y = (undist_points[:, 1] - cy) / fy  # 计算y坐标的归一化坐标值

    # 去畸变系数 k1, k2, k3, p1, p2, k4
    k1, k2, k3, p1, p2, k4 = 0.30465176701545715, -0.02011118270456791, 0.001044161501340568, 0.0, 0.0, 0.6589123010635376

    r2 = x * x + y * y  # 点的径向距离的平方

    # 利用径向畸变对点进行校正
    x_dist = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2 + k4 * r2 * r2 * r2 * r2)
    y_dist = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2 + k4 * r2 * r2 * r2 * r2)

    # 利用切向畸变对点进行校正
    x_dist = x_dist + (2 * p1 * x * y + p2 * (r2 + 2 * x * x))
    y_dist = y_dist + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)

    # 转换回绝对坐标
    x_dist = x_dist * fx + cx
    y_dist = y_dist * fy + cy

    # 构建畸变后的点云坐标
    dist_points = np.stack([x_dist, y_dist])

    return dist_points


# def ego2image(ego_points, camera_intrinsic, camera2ego_matrix):
#     ego2camera_matrix = np.linalg.inv(camera2ego_matrix)
#     camera_points = ego2camera_matrix[:3, :3] @ ego_points.T + ego2camera_matrix[:3, 3].reshape(3, 1)
#     image_points_ = camera_intrinsic[:3, :3] @ camera_points
#     image_points = image_points_[:2, :] / image_points_[2]
#     image_points = pinhole_distort_points(image_points)
#     return image_points

"""
该函数接受车辆坐标系下的点的坐标、相机的内参矩阵和相机坐标系到车辆坐标系的转换矩阵作为输入参数。首先，通过计算相机坐标系到车辆坐标系的转换矩阵的逆矩阵，得到从车辆坐标系到相机坐标系的转换矩阵。然后，通过矩阵运算，将车辆坐标系下的点转换为相机坐标系下的点。接下来，使用相机的内参矩阵，将相机坐标系下的点转换为图像坐标系下的点。
"""
def ego2image(ego_points, camera_intrinsic, camera2ego_matrix):
    ego2camera_matrix = np.linalg.inv(camera2ego_matrix)  # 计算相机坐标系到车辆坐标系的转换矩阵的逆矩阵
    camera_points = ego2camera_matrix[:3, :3] @ ego_points.T + ego2camera_matrix[:3, 3].reshape(3, 1)  # 进行坐标转换运算，将车辆坐标系下的点转换为相机坐标系下的点
    image_points_ = camera_intrinsic[:3, :3] @ camera_points  # 使用相机的内参矩阵将相机坐标系下的点转换为图像坐标系下的点
    image_points = image_points_[:2, :] / image_points_[2]  # 归一化坐标处理，将齐次坐标(3D)转换为2D图像坐标
    # image_points = pinhole_distort_points(image_points)  # 可选的步骤，对图像坐标进行去畸变操作
    return image_points

    
def perspective_fit(vectors, extrinsics, intrinsic, distortion_params, img_name, data_raw):
    key_frame_name = img_name
    img_path = data_raw + '/' + str(key_frame_name) + '/' + 'front' + '/' + str(key_frame_name) + '_' + str(key_frame_name) +'.jpg'
    # 通过拼接路径，获取图像的路径

    image = cv2.imread(img_path)  # 读取图像
    color_map = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (100, 255, 0), (100, 0, 255), (255, 100, 0),
        (0, 100, 255), (255, 0, 100), (0, 255, 100),
        (255, 255, 255), (0, 0, 0), (0, 100, 100)
    ]
    # 定义颜色映射表，用于不同车道线的颜色标识

    img_lines = []  # 创建一个空列表，用于保存图像坐标系下的线段
    for i, vector in enumerate(vectors):  # 遍历车道线向量列表
        img_line = []  # 创建一个空列表，用于保存当前车道线向量中的线段
            
        for line in vector:  # 遍历当前车道线向量中的线段
            line = np.array(line)  # 将线段转换为NumPy数组类型
            img_points = ego2image(line, intrinsic, extrinsics)  # 将线段坐标进行坐标变换，转换为图像坐标系下的点
            img_points = img_points.T  # 转置操作，将每一列代表一个坐标点的x和y值
            img_line.append(img_points.tolist())  # 将转换后的坐标点列表添加到img_line中
        img_lines.append(img_line)  # 将处理后的img_line添加到img_lines列表中


    lines = []
    for i in range(len(img_lines)):
        lines += img_lines[i]
    # 将img_lines中的坐标转换为一维的lines列表，方便后续处理

    lens = len(lines) // 2

    lens = len(lines) // 2  # 计算车道线数量，并取整除以2，得到一半的数量

    cat_lines = []  # 创建一个空列表，用于保存拼接后的车道线

    for i in range(lens):  # 遍历一半的车道线数量
        cat_lane = lines[i] + lines[i + lens]  # 将对应位置的两条车道线进行拼接
        cat_lines.append(lines[i] + lines[i + lens])  # 将拼接后的车道线添加到cat_lines列表中


    fit_lanes = []  # 创建一个空列表，用于保存拟合后的车道线

    for list_line in cat_lines:  # 遍历拼接后的车道线列表
        np_lane = np.array(list_line)  # 将车道线转换为NumPy数组类型

        arrSortIndex = np.lexsort([np_lane[:, 1]])  # 根据车道线的y坐标进行排序的索引
        np_lane = np_lane[arrSortIndex, :]  # 根据索引对车道线重新排序
        xs_gt = np_lane[:, 0]  # 提取重新排序后的车道线的x坐标
        ys_gt = np_lane[:, 1]  # 提取重新排序后的车道线的y坐标

        """
        对点 (x, y) 进行 deg 次多项式拟合，拟合函数形式为 p(x) = p[0] * x**deg + ... + p[deg]。返回使得在阶次 deg, deg-1, ... 0 中平方误差最小的系数向量 p。
        """
        poly_params_yx = np.polyfit(ys_gt, xs_gt, deg=3)
        # 对车道线进行多项式拟合，得到拟合多项式的系数

        y_min, y_max = np.min(ys_gt), np.max(ys_gt)  # 获取ys_gt中的最小值和最大值，并分别赋值给y_min和y_max
        y_min = math.floor(y_min)  # 向下取整，得到y_min的整数值
        y_max = math.ceil(y_max)  # 向上取整，得到y_max的整数值
        y_sample = np.array(range(y_min, y_max, 5))  # 在y_min和y_max之间以5为步长进行等间隔采样，生成一个NumPy数组y_sample
        ys_out = np.array(y_sample, dtype=np.float32)  # 将y_sample转换为NumPy数组，并指定数据类型为float32，赋值给ys_out
        xs_out = np.polyval(poly_params_yx, ys_out)  # 使用多项式参数poly_params_yx对ys_out进行求值，得到对应的x坐标


        fit_lane = np.zeros((len(xs_out), 2))
        fit_lane[:, 0] = xs_out
        fit_lane[:, 1] = ys_out
        # 构造拟合的车道线坐标
        
        # mask_idex = fit_lane[:, 0] > 0
        # if not any(mask_idex):
        #     continue
        #
        # fit_lane = fit_lane[mask_idex]

        fit_lanes.append(fit_lane)
    # 将拟合的车道线添加到fit_lanes列表中

    for k in range(len(fit_lanes)):
        image_lane = fit_lanes[k - 1]
        for i in range(1, image_lane.shape[0]):
            image = cv2.line(image, (int(image_lane[i - 1][0]), int(image_lane[i - 1][1])),
                            (int(image_lane[i][0]), int(image_lane[i][1])), color_map[k], 1)
    # 将拟合的车道线在图像上进行可视化显示，使用不同颜色进行标识

    cv2.imwrite('./align_time.jpg', image)
    # 将处理后的图像保存到本地
    
    # cv2.imshow("frame", image)
    # cv2.waitKey(0)


def perspective(vectors, extrinsics, intrinsic, distortion_params, img_name, data_raw):
    key_frame_name = img_name  # 将img_name赋值给key_frame_name
    img_path = data_raw + '/' + str(key_frame_name) + '/' + 'front' + '/' + str(key_frame_name) + '_' + str(key_frame_name) +'.jpg'  # 构建图像路径
    # image = cv2.imread('/home/slj/Documents/workspace/ThomasVision/data/smart_lane/data_raw/20230621134244513/front/20230621134244513_20230621134244513.jpg')  # 读取图像
    image = cv2.imread(img_path)  # 读取图像
    color_map = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),  # 颜色映射表
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (100, 255, 0), (100, 0, 255), (255, 100, 0),
        (0, 100, 255), (255, 0, 100), (0, 255, 100),
        (255, 255, 255), (0, 0, 0), (0, 100, 100)
    ]

    for i, vector in enumerate(vectors):  # 遍历vectors中的向量
        for line in vector:  # 遍历向量中的每条线
            line = np.array(line)  # 将线转换为NumPy数组类型

            img_points = ego2image(line, intrinsic, extrinsics)  # 将车辆坐标系的线投影到图像坐标系
            img_points = img_points.T  # 转置图像点坐标矩阵
            for k in range(1, img_points.shape[0]):  # 遍历图像点坐标矩阵中的每个点
                image = cv2.line(image, (int(img_points[k - 1][0]), int(img_points[k - 1][1])),  # 在图像上绘制线段
                                 (int(img_points[k][0]), int(img_points[k][1])), color_map[i], 1)
    cv2.imwrite('./align_time.jpg', image)  # 将处理后的图像保存为文件

"""
给定的车辆位姿（x、y位置和偏航角），计算相机相对于车辆的外部参数，用于后续的坐标转换和姿态对齐操作。
"""
def pose_align(ego_pose):
    x_position, y_position, key_yaw = ego_pose  # 将ego_pose解包为x_position、y_position和key_yaw
    R_OB = [math.cos(yaw), -math.sin(yaw), 0, math.sin(yaw), math.cos(yaw), 0, 0, 0, 1]  # 构建旋转矩阵R_OB
    R_OB = np.array(R_OB).reshape(3, 3)  # 将R_OB转换为NumPy数组并reshape为3x3的矩阵
    T_OB = np.array([x_position, y_position, 0])  # 构建平移矩阵T_OB

    extrinsics = np.zeros((4, 4))  # 构建4x4的零矩阵extrinsics
    extrinsics[:3, :3] = R_OB  # 将旋转矩阵R_OB赋值给extrinsics的前3行3列
    extrinsics[:3, 3] = T_OB  # 将平移矩阵T_OB赋值给extrinsics的前3行第4列
    extrinsics[3, 3] = 1.0  # 将extrinsics的第4行第4列赋值为1.0，用于保持矩阵的齐次性

    return extrinsics  # 返回extrinsics作为结果

"""
根据车辆位姿和所有点线信息，计算对齐后的线段，并存储在列表vectors中
"""
def motion_unalign(all_point_lines, all_poses):
    vectors = []  # 用于存储结果的列表

    for i, all_point_info in enumerate(zip(all_point_lines, all_poses)):  # 遍历all_point_lines和all_poses的组合
        all_point_line, all_pose = all_point_info  # 解包all_point_info，得到all_point_line和all_pose
        if i == 0:  # 如果是第一帧
            key_pose = all_pose  # 将当前帧的位姿作为关键位姿
            extrinsics_OA = pose_align(key_pose)  # 根据关键位姿计算外部参数extrinsics_OA
            local_vector = []  # 用于存储当前帧的局部向量
            for line in all_point_line:  # 遍历当前帧的线段
                line = np.array(line)  # 将线段转换为NumPy数组类型
                local_line = line @ extrinsics_OA[:3, :3] + extrinsics_OA[:3, 3]  # 计算局部线段
                local_vector.append(local_line)  # 将局部线段添加到局部向量中
            vectors.append(local_vector)  # 将局部向量添加到结果列表中
        else:  # 对于不是第一帧的情况
            pose = all_pose  # 获取当前帧的位姿
            extrinsics_OB = pose_align(pose)  # 根据当前帧的位姿计算外部参数extrinsics_OB

            align_lines = []  # 用于存储对齐后的线段
            for line in all_point_line:  # 遍历当前帧的线段
                line = np.array(line)  # 将线段转换为NumPy数组类型

                align_line = line @ extrinsics_OB[:3, :3] + extrinsics_OB[:3, 3]  # 计算对齐后的线段
                align_lines.append(align_line)  # 将对齐后的线段添加到列表中

            vectors.append(align_lines)  # 将对齐后的线段列表添加到结果列表中

    return vectors  # 返回结果列表

def motion_align(all_point_lines, all_poses):
    vectors = []  # 用于存储结果的列表

    # key_pose = all_poses[1]
    # extrinsics_OA = pose_align(key_pose)
    # local_vector = []
    # for line in all_point_lines[1]:
    #     line = np.array(line)
    #     line = line[:, ]
    #     local_line = line @ np.linalg.inv(extrinsics_OA[:3, :3]) + extrinsics_OA[:3, 3]
    #     # local_line = line + extrinsics_OA[:3, 3]
    #     local_vector.append(local_line)
    # # vectors.append(local_vector)
    # vectors.append(all_point_lines[1])

    for i, all_point_info in enumerate(zip(all_point_lines, all_poses)):
        all_point_line, all_pose = all_point_info
        if i == 0:  # 对于第一帧的情况
            # continue
            key_pose = all_pose  # 将当前帧的位姿作为关键位姿
            extrinsics_OA = pose_align(key_pose)  # 根据关键位姿计算外部参数extrinsics_OA
            local_vector = []  # 用于存储当前帧的局部向量
            for line in all_point_line:  # 遍历当前帧的线段
                # if line == []:
                #     continue
                line = np.array(line)  # 将线段转换为NumPy数组类型
                line = line[:, ]  # 切片操作（可能省略了一些代码）
                # try:
                local_line = line @ np.linalg.inv(extrinsics_OA[:3, :3]) + extrinsics_OA[:3, 3]  # 计算局部线段
                # except:
                #     pass
                # local_line = line + extrinsics_OA[:3, 3]
                local_vector.append(local_line)  # 将局部线段添加到局部向量中
            # vectors.append(local_vector)
            vectors.append(all_point_line)  # 将当前帧的线段添加到结果列表中

        else:  # 对于不是第一帧的情况
            pose = all_pose  # 获取当前帧的位姿
            # x_position, y_position, yaw = pose
            extrinsics_OB = pose_align(pose)  # 根据当前帧的位姿计算外部参数extrinsics_OB

            extrinsics_AB = np.linalg.inv(extrinsics_OA) @ extrinsics_OB  # 计算外部参数extrinsics_AB
            extrinsics_BA = np.linalg.inv(extrinsics_AB)  # 计算外部参数extrinsics_BA（可能省略了一些代码）
            # extrinsics_BA = extrinsics_AB

            align_lines = []  # 用于存储对齐后的线段
            for line in all_point_line:  # 遍历当前帧的线段
                # if line == []:
                #     continue
                line = np.array(line)  # 将线段转换为NumPy数组类型
                align_line = line @ extrinsics_BA[:3, :3] - extrinsics_BA[:3, 3]  # 计算对齐后的线段
                # align_line = line @ np.linalg.inv(extrinsics_OB[:3, :3]) + extrinsics_OB[:3, 3]
                # align_line = line + extrinsics_OB[:3, 3]
                align_lines.append(align_line.tolist())  # 将对齐后的线段添加到列表中

            vectors.append(align_lines)  # 将对齐后的线段列表添加到结果列表中

    # ego_vectors = []
    # for vector in vectors:
    #     ego_vector = []
    #     for line in vector:
    #         ego_line = line @ extrinsics_OA[:3, :3] - extrinsics_OA[:3, 3]
    #         ego_vector.append(ego_line)
    #
    #     ego_vectors.append(ego_vector)

    return vectors  # 返回结果列表
    # return ego_vectors

"""
根据车辆位姿和所有点线信息，计算相对运动对齐后的线段，并存储在列表vectors中
"""
def motion_align_relativity(all_point_lines, all_poses):
    vectors = []  # 用于存储结果的列表

    for i, all_point_info in enumerate(zip(all_point_lines, all_poses)):
        all_point_line, all_pose = all_point_info
        if i == 0:  # 对于第一帧的情况
            key_pose = all_pose  # 将当前帧的位姿设为关键位姿
            key_x_position, key_y_position, key_yaw = key_pose  # 将关键位姿中的位置和姿态信息提取出来
            vectors.append(all_point_line)  # 将当前帧的线段添加到结果列表中
        else:  # 对于不是第一帧的情况
            pose = all_pose  # 获取当前帧的位姿
            x_position, y_position, yaw = pose  # 将当前帧的位置和姿态信息提取出来
            R = np.array([math.cos(yaw - key_yaw), -math.sin(yaw - key_yaw), 0,
                          math.sin(yaw - key_yaw), math.cos(yaw - key_yaw), 0,
                          0, 0, 1]).reshape(3, 3)  # 计算旋转矩阵R
            T = np.array([y_position - key_y_position, x_position - key_x_position, 0])  # 计算平移向量T

            align_lines = []  # 用于存储对齐后的线段
            for line in all_point_line:  # 遍历当前帧的线段
                line = np.array(line)  # 将线段转换为NumPy数组类型
                align_lines.append(line)  # 将线段添加到对齐后的线段列表中

            vectors.append(align_lines)  # 将对齐后的线段列表添加到结果列表中

    return vectors  # 返回结果列表

"""
对给定的车道线数据进行多项式拟合，得到拟合后的车道线。
"""
def lane_fit(gt_lanes, poly_order=3, sample_step=10, interp=False):
    lanes = []  # 用于存储全部车道线的列表

    # 将所有车道线添加到lanes列表中
    for i in range(len(gt_lanes)):
        lanes += gt_lanes[i]

    lens = len(lanes) // 2  # lanes列表中车道线的数量的一半

    cat_lanes = []  # 用于存储连接的车道线的列表
    for i in range(lens):
        cat_lane = lanes[i] + lanes[i + lens]  # 将两个车道线连接起来
        cat_lanes.append(cat_lane)  # 将连接后的车道线添加到cat_lanes列表中

    fit_lanes = []  # 用于存储拟合后的车道线的列表
    for list_lane in cat_lanes:  # 遍历连接的车道线
        np_lane = np.array(list_lane)  # 将车道线转换为NumPy数组类型
        arrSortIndex = np.lexsort([np_lane[:, 0]])  # 根据x坐标对车道线进行排序
        np_lane = np_lane[arrSortIndex, :]  # 对车道线进行排序
        xs_gt = np_lane[:, 0]  # 车道线的x坐标
        ys_gt = np_lane[:, 1]  # 车道线的y坐标
        zs_gt = np_lane[:, 2]  # 车道线的z坐标

        # 在车道线上进行多项式拟合
        poly_params_xy = np.polyfit(xs_gt, ys_gt, deg=poly_order)  # 多项式拟合x-y平面
        poly_params_xz = np.polyfit(xs_gt, zs_gt, deg=poly_order)  # 多项式拟合x-z平面

        x_min, x_max = np.min(xs_gt), np.max(xs_gt)  # 车道线x坐标的最小值和最大值
        x_min = math.floor(x_min)  # 取最小值的下界
        x_max = math.ceil(x_max)  # 取最大值的上界
        x_sample = np.array(range(x_min, x_max, sample_step))  # 生成采样点坐标数组

        xs_out = np.array(x_sample, dtype=np.float32)  # 采样点的x坐标

        # 根据拟合参数计算采样点的y坐标和z坐标
        ys_out = np.polyval(poly_params_xy, xs_out)
        zs_out = np.polyval(poly_params_xz, xs_out)

        fit_lane = np.zeros((len(ys_out), 3))  # 创建拟合后车道线的数组，大小为(len(ys_out), 3)
        fit_lane[:, 0] = xs_out  # 将采样点的x坐标放入拟合后车道线数组的第一列
        fit_lane[:, 1] = ys_out  # 将采样点的y坐标放入拟合后车道线数组的第二列
        fit_lane[:, 2] = zs_out  # 将采样点的z坐标放入拟合后车道线数组的第三列

        mask_idex = fit_lane[:, 0] > 0  # 生成一个布尔掩码，用于筛选无效的拟合点
        if not any(mask_idex):  # 如果没有有效的拟合点，则跳过当前车道线
            continue

        fit_lane = fit_lane[mask_idex]  # 根据掩码筛选有效的拟合点

        fit_lanes.append(fit_lane)  # 将拟合后的车道线添加到列表fit_lanes中

    return fit_lanes  # 返回拟合后的车道线列表

"""
根据给定的点线信息绘制2D图形并保存为图片
"""
def get_lane_imu_img_2D(points, iego_lanes=None):
    filepath = "./ego_align.png"  # 图片保存的路径

    fig_2d = plt.figure(figsize=(6.4, 6.4))  # 创建一个2D图形窗口
    plt.grid(linestyle='--', color='y', linewidth=0.5)  # 设置网格线的样式

    x_major_locator = MultipleLocator(1)  # x轴主刻度间隔设置为1
    y_major_locator = MultipleLocator(4)  # y轴主刻度间隔设置为4
    ax = plt.gca()  # 获取当前图形对象
    ax.xaxis.set_major_locator(x_major_locator)  # 设置x轴刻度间隔
    ax.yaxis.set_major_locator(y_major_locator)  # 设置y轴刻度间隔

    for i, vector in enumerate(points):  # 遍历points列表中的向量
        if i == 0:
            color = 'b'
        if i == 1:
            color = 'r'
        if i == 2:
            color = 'g'

        for line in vector:  # 遍历向量中的线段
            x_data = []
            y_data = []
            for poi in line:  # 遍历线段中的点
                x_data.append(poi[1])
                y_data.append(poi[0])
            plt.plot(x_data, y_data, linestyle='-', color=color, linewidth=1)  # 画线段

    plt.xlabel('X: ')  # 设置x轴标签
    plt.ylabel('Y: distance')  # 设置y轴标签
    plt.title("Only show X_Y: align pic")  # 设置图形标题
    plt.savefig(filepath)  # 保存图形文件
    plt.cla()  # 清除当前的坐标轴
    plt.close()  # 关闭图形窗口

    return filepath  # 返回保存的图形文件路径

"""
给定的点线信息绘制2D图形并保存为图片
"""
def get_lane_imu_img_2D_fit(points, iego_lanes=None):
    filepath = "./ego_align_cat.png"  # 图片保存的路径

    fig_2d = plt.figure(figsize=(6.4, 6.4))  # 创建一个2D图形窗口
    plt.grid(linestyle='--', color='y', linewidth=0.5)  # 设置网格线的样式

    x_major_locator = MultipleLocator(1)  # x轴主刻度间隔设置为1
    y_major_locator = MultipleLocator(4)  # y轴主刻度间隔设置为4
    ax = plt.gca()  # 获取当前图形对象
    ax.xaxis.set_major_locator(x_major_locator)  # 设置x轴刻度间隔
    ax.yaxis.set_major_locator(y_major_locator)  # 设置y轴刻度间隔

    for i, vector in enumerate(points):  # 遍历points列表中的向量
        line = vector.tolist()  # 将当前向量转换为列表
        x_data = []
        y_data = []
        for poi in line:  # 遍历线段的点
            x_data.append(poi[1])
            y_data.append(poi[0])
        plt.plot(x_data, y_data, linestyle='-', color='b', linewidth=1)  # 绘制线段

    plt.xlabel('X: ')  # 设置x轴标签
    plt.ylabel('Y: distance')  # 设置y轴标签
    plt.title("Only show X_Y: align pic")  # 设置图形标题
    plt.savefig(filepath)  # 保存图形文件
    plt.cla()  # 清除当前的坐标轴
    plt.close()  # 关闭图形窗口

    return filepath  # 返回保存的图形文件路径


if __name__ == '__main__':
    Odometry_path = '/home/slj/Documents/workspace/ThomasVision/data/smart_lane/Odometry.txt'
    calib_path = '/home/slj/Documents/workspace/ThomasVision/data/smart_lane/front_new.json'
    data_annotation = '/home/slj/Documents/workspace/ThomasVision/data/smart_lane/data_annotation'
    data_raw = '/home/slj/Documents/workspace/ThomasVision/data/smart_lane/data_raw'
    lidar_calib_path = '/home/slj/Documents/workspace/ThomasVision/data/smart_lane/lidar_calib.json'

    with open(calib_path) as ft:
        calibs = json.load(ft)
        extrinsic_rotation_matrix = calibs['extrinsic_rotation_matrix']
        extrinsic_translation_matrix = calibs['extrinsic_translation_matrix']
        intrinsic_matrix = calibs['intrinsic_matrix']
        distortion_params = calibs['distortion_params']

    with open(lidar_calib_path) as lidar_ft:
        lidar_calibs = json.load(lidar_ft)
        lidar_extrinsic_rotation_matrix = lidar_calibs['extrinsic_rotation_matrix']
        lidar_extrinsic_translation_matrix = lidar_calibs['extrinsic_translation_matrix']


    extrinsics = np.zeros((4, 4))
    extrinsics[:3, :3] = extrinsic_rotation_matrix
    extrinsics[:3, 3] = extrinsic_translation_matrix
    extrinsics[3, 3] = 1.0

    intrinsic = np.zeros((3, 4))
    intrinsic[:3, :3] = intrinsic_matrix

    calib = np.matmul(intrinsic, extrinsics)

    data_files = os.listdir(data_annotation)
    data_files.sort()

    data_lines = {}
    with open(Odometry_path, 'r') as infile:
        for line in infile:
            data = line.strip().split(',')
            data_lines[data[0]] = data[1:]

    all_point_lines = []
    all_poses = []
    img_names = []
    i = 30
    for data_file in data_files[i:i+2]:
        img_names.append(data_file.split('.')[0])
        point_lines = []
        ann_path = os.path.join(data_annotation, data_file)
        with open(ann_path) as ft:
            data = json.load(ft)
            point_3ds = data['annotations']
            for point_3d in point_3ds:
                point_line = []
                points = point_3d['points']
                for point in points:
                    x, y, z = point['x'], point['y'], point['z']
                    if x < 0 :
                        continue
                    point_line.append([x, y, z])
                point_lines.append(point_line)
        all_point_lines.append(point_lines)
        data_name = data_file.split('.')[0][:15]
        data = data_lines[data_name]
        x_position, y_position, yaw = data
        x_position = float(x_position)
        y_position = float(y_position)
        yaw = float(yaw.split(';')[0])

        all_poses.append([x_position, y_position, yaw])

    # vectors = motion_align_relativity(all_point_lines, all_poses)
    vectors = motion_align(all_point_lines, all_poses)
    filepath = get_lane_imu_img_2D(vectors)

    # vectors = lane_fit(vectors)
    # filepath_1 = get_lane_imu_img_2D_fit(vectors)

    # perspective(vectors, extrinsics, intrinsic, distortion_params, img_names[0], data_raw)
    perspective_fit(vectors, extrinsics, intrinsic, distortion_params, img_names[0], data_raw)





# [0.003400596494017081 -0.007649317362956704 0.9999649701868492 0.0
# -0.9998360386253572 0.017757926699115575 0.0035359989444003737 0.0
# -0.017784352882268525 -0.9998130291738248 -0.0075876761702830375 0.0
# 0.02605885443814961 1.0919294769483647 -1.8699881027882832 1.0 ]




