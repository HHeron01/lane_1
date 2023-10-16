      
import json
# from utils.utils import *
import os.path as ops
import glob
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from mmdet3d.datasets.pipelines import Compose
from lane.mmdet3d.datasets.multiview_datasets.instance_bevlane.openlane_extract_AF import OpenLaneSegMask
from mmdet3d.datasets.multiview_datasets.image import img_transform, normalize_img
from mmdet.datasets import DATASETS
import cv2


# 使用@DATASETS.register_module()装饰器注册了一个新的数据集模块。
@DATASETS.register_module()
class OpenLane_Dataset_AF(Dataset):
    def __init__(self, images_dir, json_file_dir, data_config=None, grid_config=None,
                 test_mode=False, pipeline=None, CLASSES=None, use_valid_flag=False):
        # 初始化父类
        super(OpenLane_Dataset_AF, self).__init__()
        
        # 从grid_config中获取宽度和深度范围
        width_range = (grid_config['x'][0], grid_config['x'][1])  # (-19.2, 19.2)
        depth_range = (grid_config['y'][0], grid_config['y'][1])  # (0, 96)
        # 从grid_config中获取宽度和深度的分辨率
        self.width_res = grid_config['x'][2]  # 0.2
        self.depth_res = grid_config['y'][2]  # 0.3
        
        # 从data_config中获取源图像的宽度和高度
        self.IMG_ORIGIN_W, self.IMG_ORIGIN_H = data_config['src_size']  # (1920, 1280)
        # 从data_config中获取输入图像的宽度和高度
        self.input_w, self.input_h = data_config['input_size']  # (960, 1280)
        
        # 获取x和y的最小和最大范围
        self.x_min = grid_config['x'][0]
        self.x_max = grid_config['x'][1]
        self.y_min = grid_config['y'][0]
        self.y_max = grid_config['y'][1]
        
        # 在y范围内均匀采样
        self.y_samples = np.linspace(self.y_min, self.y_max, num=100, endpoint=False)
        
        # 定义z的偏移量
        self.zoff = 1.08
        self.use_valid_flag = use_valid_flag
        self.CLASSES = CLASSES
        # 根据test_mode设置是否为训练模式
        self.is_train = not test_mode  
        self.data_config = data_config
        # 创建网格
        self.grid = self.make_grid()
        
        # 定义图像目录和json文件目录
        self.images_dir = images_dir
        self.json_file_dir = json_file_dir
        # 初始化数据集
        self.samples = self.init_dataset(json_file_dir)
        # 初始化OpenLaneSegMask对象
        self.mask_extract = OpenLaneSegMask(width_range=width_range,
            depth_range=depth_range,
            width_res=self.width_res,
            depth_res=self.depth_res,
            data_config=self.data_config)
        
        self.downsample = 4
        
        # 如果提供了pipeline，则初始化Compose对象
        if pipeline is not None:
            self.pipeline = Compose(pipeline)
        
        # 初始化标志数组
        self.flag = np.zeros(len(self.samples), dtype=np.uint8)

    # 定义返回数据集长度的方法
    def __len__(self):
        return len(self.samples) 

    # 创建一个3D网格的方法
    def make_grid(self):
        xcoords = torch.linspace(self.x_min, self.x_max, int((self.x_max - self.x_min) / self.width_res))  # 191
        ycoords = torch.linspace(self.y_min, self.y_max, int((self.y_max - self.y_min) / self.depth_res))  # 320
        yy, xx = torch.meshgrid(ycoords, xcoords)
        return torch.stack([xx, yy, torch.full_like(xx, self.zoff)], dim=-1)

    # 根据可见性修剪3D车道的方法
    def prune_3d_lane_by_visibility(self, lane_3d, visibility):
        lane_3d = lane_3d[visibility > 0, ...]
        return lane_3d

    def prune_3d_lane_by_range(self, lane_3d, x_min, x_max):
        # 从3D车道线数据中筛选出y坐标在 (0, 200) 范围内的数据点
        lane_3d = lane_3d[np.logical_and(lane_3d[:, 1] > 0, lane_3d[:, 1] < 200), ...]
        # 从筛选后的数据中再次筛选出x坐标在 (x_min, x_max) 范围内的数据点
        lane_3d = lane_3d[np.logical_and(lane_3d[:, 0] > x_min, lane_3d[:, 0] < x_max), ...]
        # 返回剪裁后的3D车道线数据
        return lane_3d


    def data_filter(self, gt_lanes, gt_visibility, gt_category):
        # 根据可视性信息，对每条3D车道线进行修剪
        gt_lanes = [self.prune_3d_lane_by_visibility(np.array(gt_lane), np.array(gt_visibility[k])) for k, gt_lane in
                    enumerate(gt_lanes)]
        
        # 保留至少包含两个点的车道线和其对应的类别
        gt_category = [gt_category[k] for k, lane in enumerate(gt_lanes) if lane.shape[0] > 1]
        gt_lanes = [lane for lane in gt_lanes if lane.shape[0] > 1]

        # 只保留在特定高度范围内的车道线及其对应的类别
        gt_category = [gt_category[k] for k, lane in enumerate(gt_lanes)
                    if lane[0, 1] < self.y_samples[-1] and lane[-1, 1] > self.y_samples[0]]
        gt_lanes = [lane for lane in gt_lanes if lane[0, 1] < self.y_samples[-1] and lane[-1, 1] > self.y_samples[0]]

        # 根据x轴的范围对车道线进行修剪
        gt_lanes = [self.prune_3d_lane_by_range(np.array(lane), self.x_min, self.x_max) for lane in gt_lanes]

        # 再次保留至少包含两个点的车道线及其对应的类别
        gt_category = [gt_category[k] for k, lane in enumerate(gt_lanes) if lane.shape[0] > 1]
        gt_lanes = [lane for lane in gt_lanes if lane.shape[0] > 1]

        # 返回处理过的车道线类别和车道线数据
        return gt_category, gt_lanes

    def sample_augmentation(self):
        """
        对样本进行尺寸调整或增强。
        返回:
            resize (tuple): 调整后的宽度和高度与原始宽度和高度之比的比例因子。
            resize_dims (tuple): 调整后的宽度和高度的尺寸。
        说明:
            fW 和 fH 是期望的输入图像的宽度和高度。
            self.input_w 和 self.input_h 是从配置中获得的输入尺寸。
            self.IMG_ORIGIN_W 和 self.IMG_ORIGIN_H 是原始图像的宽度和高度。
        """

        # 获取期望的输入图像的宽度和高度
        fW, fH = self.input_w, self.input_h
        # 计算调整后的宽度和高度与原始宽度和高度之比的比例因子
        resize = (fW / self.IMG_ORIGIN_W, fH / self.IMG_ORIGIN_H)
        # 获取调整后的宽度和高度的尺寸
        resize_dims = (fW, fH)
        
        return resize, resize_dims

    def get_seg_mask(self, gt_lanes_3d, gt_laned_2d, gt_category, gt_visibility):

        mask_bev, mask_2d = self.mask_extract(gt_lanes_3d, gt_laned_2d, gt_category, gt_visibility)
        return mask_bev, mask_2d

    def perspective(self, matrix, vector):
        """
        使用投影矩阵对向量应用透视投影。
        参数:
            matrix (torch.Tensor): 用于透视投影的矩阵。
            vector (torch.Tensor): 需要进行透视投影的3D向量。
        返回:
            tuple: 包含两个元素的元组:
                - 投影后的2D向量
                - 一个表示向量是否在有效的可见范围内的mask（即是否在摄像机前面）。
        说明:
            1. 首先，我们对3D向量进行操作，使其可以和4x4的投影矩阵进行矩阵乘法。
            2. 接着，我们将3D向量通过矩阵乘法转换为齐次坐标。
            3. 最后，我们将齐次坐标转换为普通的2D坐标。
        """

        # 为向量增加一个维度以进行矩阵乘法操作
        vector = vector.unsqueeze(-1)
        
        # 通过矩阵乘法计算齐次坐标
        homogeneous = torch.matmul(matrix[..., :-1], vector) + matrix[..., [-1]]
        # 齐次坐标来实现透视投影
        """
        在二维和三维空间中，齐次坐标的概念尤其重要。
        考虑二维空间：一个普通的二维点由(x, y)表示。而在齐次坐标中，该点可表示为(X, Y, W)，其中x = X/W且y = Y/W。W是齐次坐标的缩放因子或齐次分量。
        同理，在三维空间中，点(x, y, z)在齐次坐标中可以表示为(X, Y, Z, W)。
        """
        homogeneous = homogeneous.squeeze(-1)
        
        # 判断向量是否在摄像机前面（即z坐标大于0）
        b = (homogeneous[..., -1] > 0).unsqueeze(-1)
        b = torch.cat((b, b, b), -1)
        b[..., -1] = True

        # 乘以mask来处理不在摄像机前面的点
        homogeneous = homogeneous * b.float()

        # 将齐次坐标转换为普通的2D坐标
        return homogeneous[..., :-1] / homogeneous[..., [-1]], b.float()

    def get_data_info(self, index, debug=True):
        # 从提供的索引位置获取样本的JSON文件名
        label_json = self.samples[index]

        # 使用os.path.join来构建JSON文件的完整路径
        label_file_path = ops.join(self.json_file_dir, label_json)

        # 初始化列表来存储图像、变换矩阵等相关数据
        # 用于存储图像的列表
        imgs = []
        # 用于存储摄像机外部变换矩阵的列表
        trans = []
        # 用于存储摄像机旋转矩阵的列表
        rots = []
        # 用于存储摄像机内部参数矩阵的列表
        intrins = []
        # 用于存储预处理后的平移矩阵的列表
        post_trans = []
        # 用于存储预处理后的旋转矩阵的列表
        post_rots = []
        # 用于存储摄像机外部参数矩阵的列表
        extrinsics = []
        # 用于存储相机镜头畸变参数的列表
        undists = []

        # 打开并读取指定的JSON文件内容
        with open(label_file_path, 'r') as fr:
            info_dict = json.loads(fr.read())

        # 使用os.path.join来构建与JSON中记录的相对路径对应的图像文件的完整路径
        image_path = ops.join(self.images_dir, info_dict['file_path'])
        # '/workspace/openlane_all/images/training/segment-10017090168044687777_6380_000_6400_000_with_camera_labels/155008346754599000.jpg'
        # 检查图像文件是否存在
        assert ops.exists(image_path), '{:s} not exist'.format(image_path)

        # 使用Python Imaging Library（PIL）库来打开图像
        img = Image.open(image_path)  # (1920, 1280)
        # 使用OpenCV库读取相同的图像文件
        image = cv2.imread(image_path)      
        # 调整OpenCV图像的大小以适配预定的输入尺寸
        image = cv2.resize(image, (self.input_w, self.input_h))  # (960, 640)
        # 从JSON数据中提取摄像机的外部参数并转化为NumPy数组
        extrinsic = np.array(info_dict['extrinsic'])
        # 从JSON数据中提取摄像机的内部参数并转化为NumPy数组
        intrinsic = np.array(info_dict['intrinsic'])
        # 从JSON数据中提取地面真实的车道线信息
        gt_lanes_packeds = info_dict['lane_lines']
        # 使用预定义的方法进行样本增强，并返回resize参数和其维度
        resize, resize_dims = self.sample_augmentation()
        
        # 根据上述参数对PIL图像进行尺寸和方向的变换
        img, post_rot, post_tran = img_transform(img, resize, resize_dims)

        # 定义两个旋转矩阵，它们用于在不同坐标系间转换
        R_vg = np.array([[0, 1, 0],
                        [-1, 0, 0],
                        [0, 0, 1]], dtype=float)
        R_gc = np.array([[1, 0, 0],
                        [0, 0, 1],
                        [0, -1, 0]], dtype=float)

        # 对摄像机的外部参数矩阵进行坐标系转换
        """
        将相机坐标系到全球坐标系的旋转变换、相机坐标系到相机中心的变换和全球坐标系到栅格地面坐标系的旋转变换进行连续相乘
        R_vg 是相机坐标系（Camera Coordinate System）到全球坐标系（Global Coordinate System）的旋转变换矩阵。
        extrinsic[:3, :3] 是 extrinsic 矩阵的前3行前3列，表示相机坐标系到相机中心的变换，即外参。
        R_gc 是全球坐标系到栅格地面坐标系（Grid Ground Coordinate System）的旋转变换矩阵。
        np.linalg.inv(R_vg) 表示 R_vg 的逆矩阵。
        np.matmul(np.matmul(np.matmul(np.linalg.inv(R_vg), extrinsic[:3, :3]), R_vg), R_gc) 表示将旋转矩阵的逆矩阵与外参相乘，然后再与 R_vg 和 R_gc 相乘。
        最后，将乘积结果赋值给 extrinsic[:3, :3]，即更新了外参的旋转部分。
        """
        extrinsic[:3, :3] = np.matmul(np.matmul(
            np.matmul(np.linalg.inv(R_vg), extrinsic[:3, :3]),R_vg), R_gc)
        
        # 更新外部参数矩阵的平移部分
        extrinsic[0:2, 3] = 0.0

        gt_lanes_2d, gt_lanes_3d, gt_visibility, gt_category = [], [], [], []

        for j, gt_lane_packed in enumerate(gt_lanes_packeds):
            # A GT lane can be either 2D or 3D
            # if a GT lane is 3D, the height is intact from 3D GT, so keep it intact here too
            lane2d = np.array(gt_lane_packed['uv'], dtype=np.float32)
            lane3d = np.array(gt_lane_packed['xyz'], dtype=np.float32)
            lane_visibility = np.array(gt_lane_packed['visibility'])

            lane3d = np.vstack((lane3d, np.ones((1, lane3d.shape[1]))))
            cam_representation = np.linalg.inv(
            np.array([[0, 0, 1, 0],
                    [-1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, 0, 1]], dtype=float))     # 定义相机表示矩阵的逆矩阵

        # 进行逆透视变换：将3D车道线坐标从相机坐标系转换到全局坐标系
        lane3d = np.matmul(extrinsic, np.matmul(cam_representation, lane3d))
        # 提取3D车道线的前三维坐标，并转置为行优先形式
        lane3d = lane3d[0:3, :].T
        # 将2D车道线坐标转置为行优先形式
        lane2d = lane2d.T

        gt_lanes_3d.append(lane3d)
        gt_lanes_2d.append(lane2d)
        gt_visibility.append(lane_visibility)
        gt_category.append(gt_lane_packed['category'])

        gt_category, gt_lanes_3d = self.data_filter(gt_lanes_3d, gt_visibility, gt_category)
        
        img = normalize_img(img)   # 对图像进行归一化处理
        # 将相机的平移向量、旋转矩阵、外参矩阵、内参矩阵、后处理平移向量、后处理旋转矩阵、图像数据和失真系数依次添加到对应的列表中
        trans.append(torch.Tensor(extrinsic[:3, 3]))
        rots.append(torch.Tensor(extrinsic[:3, :3]))
        extrinsics.append(torch.tensor(extrinsic).float())
        intrins.append(torch.cat((torch.Tensor(intrinsic), torch.zeros((3, 1))), dim=1).float())
        post_trans.append(post_tran)
        post_rots.append(post_rot)
        imgs.append(img)
        # 失真系数被定义为一个长度为7的零向量 torch.zeros(7)，表示不考虑任何畸变
        undists.append(torch.zeros(7))

        # 将列表转化为张量，对应维度上的元素进行堆叠
        imgs, trans, rots, intrins, post_trans, post_rots, undists, extrinsics = torch.stack(imgs), torch.stack(trans), torch.stack(rots), torch.stack(intrins), torch.stack(
            post_trans), torch.stack(post_rots), torch.stack(undists), torch.stack(extrinsics)

        extrinsics = torch.linalg.inv(extrinsics)   # 对外参矩阵取逆，由相机坐标系到全球坐标系的变换转为全球坐标系到相机坐标系的变换

        # 获取车道线的语义分割遮罩
        mask_bev, mask_2d = self.get_seg_mask(gt_lanes_3d, gt_lanes_2d, gt_category, gt_visibility)
        # # (mask_seg_bev, mask_haf_bev, mask_vaf_bev, mask_offset_bev, mask_z_bev),\
        # (mask_seg_2d,  mask_haf_2d,  mask_vaf_2d)

        # 将车道线遮罩转化为张量
        mask_bev, mask_2d = self.mask_toTensor(mask_bev, mask_2d)

        
        '''
        # if debug:
        #     visu_path = './vis_pic'
        #     calib = np.matmul(intrins, extrinsics)
        #     for gt_lane in gt_lanes:
        #         gt_lane = torch.tensor(gt_lane).float()
        #         img_points, _ = self.perspective(calib, gt_lane)

        #         post_img_points = []
        #         for img_point in img_points:
        #             img_point = torch.matmul(post_rots[0, :2, :2], img_point) + post_trans[0, :2]
        #             post_img_points.append(img_point.detach().cpu().numpy())
        #         post_img_points = np.array(post_img_points)
        #         x_2d, y_2d = post_img_points[:, 0].astype(np.int32), post_img_points[:, 1].astype(np.int32)
        #         for k in range(1, img_points.shape[0]):
        #             image = cv2.line(image, (x_2d[k - 1], y_2d[k - 1]),
        #                              (x_2d[k], y_2d[k]), (0, 0, 255), 4)
            # cv2.imwrite(visu_path + "/img.jpg", image)

        '''
        
        input_dict = dict(
            imgs=imgs,
            trans=trans,
            rots=rots,
            extrinsics=extrinsics,
            intrins=intrins,
            undists=undists,
            post_trans=post_trans,
            post_rots=post_rots,
            mask_bev=mask_bev,
            mask_2d=mask_2d,
            gt_lanes_3d=gt_lanes_3d,
            gt_lanes_2d=gt_lanes_2d,
            grid=self.grid,
            drop_idx=torch.tensor([]),
            file_path=info_dict['file_path'],
        )

        return input_dict

    def mask_toTensor(self, mask_bev, mask_2d):

        mask_seg_bev, mask_haf_bev, mask_vaf_bev, mask_offset_bev, mask_z_bev = mask_bev
        mask_seg_bev[mask_seg_bev > 0] = 1
        mask_seg_bev = torch.from_numpy(mask_seg_bev).contiguous().float().unsqueeze(0)
        mask_haf_bev = torch.from_numpy(mask_haf_bev).contiguous().float()
        mask_vaf_bev = torch.from_numpy(mask_vaf_bev).permute(2, 0, 1).contiguous().float()
        mask_offset_bev  = torch.from_numpy(mask_offset_bev).permute(2, 0, 1).contiguous().float()
        mask_z_bev  = torch.from_numpy(mask_z_bev).permute(2, 0, 1).contiguous().float()

        if mask_2d is not None:
            mask_seg_2d,  mask_haf_2d,  mask_vaf_2d = mask_2d
            mask_seg_2d[mask_seg_2d > 0] = 1
            mask_seg_2d = torch.from_numpy(mask_seg_2d).contiguous().float().unsqueeze(0)
            mask_haf_2d = torch.from_numpy(mask_haf_2d).contiguous().float()
            mask_vaf_2d = torch.from_numpy(mask_vaf_2d).permute(2, 0, 1).contiguous().float()

            return (mask_seg_bev, mask_haf_bev, mask_vaf_bev, mask_offset_bev, mask_z_bev),\
                (mask_seg_2d,  mask_haf_2d,  mask_vaf_2d)
        
        return (mask_seg_bev, mask_haf_bev, mask_vaf_bev, mask_offset_bev, mask_z_bev), None

    def init_dataset(self, json_file_dir):
        filter_samples = []  # '/workspace/openlane_all/images/validation/segment-11048712972908676520_545_000_565_000_with_camera_labels/152268469123823000.jpg'
        samples = glob.glob(json_file_dir + '**/*.json', recursive=True)
        print("[INFO] init datasets...")
        for i, sample in tqdm(enumerate(samples)):
            label_file_path = ops.join(self.json_file_dir, sample)
            with open(label_file_path, 'r') as fr:
                info_dict = json.loads(fr.read())
            image_path = ops.join(self.images_dir, info_dict['file_path'])
            if not ops.exists(image_path):
                # print('{:s} not exist'.format(image_path))
                continue
            # if i < 1014:
            #     continue
            filter_samples.append(sample)
            if len(filter_samples) > 63:
                break
            # print("image_path:", image_path)

        # return samples
        return filter_samples

    def __getitem__(self, idx):
        input_dict = self.get_data_info(idx, debug=False)
        data = self.pipeline(input_dict)
        return data
        # return input_dict

if __name__ == '__main__':
    data_config = {
        'cams': [],
        'Ncams': 1,
        'input_size': (960, 640),
        'src_size': (1920, 1280),
        'thickness': 5,
        'angle_class': 36,

        # Augmentation
        'resize': (-0.06, 0.11),
        'rot': (-5.4, 5.4),
        'flip': True,
        'crop_h': (0.0, 0.0),
        'resize_test': 0.00,

    }

    grid_config = {
        'x': [-10.0, 10.0, 0.15],
        'y': [3.0, 103.0, 0.5],
        'z': [-5, 3, 8],
        'depth': [1.0, 60.0, 1.0],
    }
    # images_dir = '/home/slj/Documents/workspace/mmdet3d/data/openlane/example/image'
    # json_file_dir = '/home/slj/Documents/workspace/mmdet3d/data/openlane/example/annotations/segment-260994483494315994_2797_545_2817_545_with_camera_labels'

    images_dir = '/home/slj/data/openlane/openlane_all/images'
    json_file_dir = '/home/slj/data/openlane/openlane_all/lane3d_300/training/'

    dataset = OpenLane_Dataset_AF(images_dir, json_file_dir, data_config=data_config, grid_config=grid_config,
                 test_mode=False, pipeline=None, CLASSES=None, use_valid_flag=True)

    for idx in tqdm(range(dataset.__len__())):
        input_dict = dataset.__getitem__(idx)
        print(idx)

    