# Copyright (c) OpenMMLab. All rights reserved.
import pickle

import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

from tools.data_converter import nuscenes_converter as nuscenes_converter

map_name_from_general_to_detection = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
}
classes = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]


def get_gt(info):
    """从信息中生成真实标签。
    参数：
        info（字典）：生成真实标签所需的信息。
    返回：
        Tensor：真实边界框。
        Tensor：真实标签。
"""
    # 从信息中获取自车坐标系到全局坐标系的旋转和平移信息
    ego2global_rotation = info['cams']['CAM_FRONT']['ego2global_rotation']
    ego2global_translation = info['cams']['CAM_FRONT']['ego2global_translation']
    # 将平移信息转换为负数
    trans = -np.array(ego2global_translation)
    # 反转旋转信息
    rot = Quaternion(ego2global_rotation).inverse
    # 初始化真实边界框和标签的列表
    gt_boxes = list()
    gt_labels = list()

    # 遍历每个物体实例信息
    for ann_info in info['ann_infos']:
        # 使用车辆坐标系
        if map_name_from_general_to_detection[ann_info['category_name']] not in classes or \
                ann_info['num_lidar_pts'] + ann_info['num_radar_pts'] <= 0:
            continue
            
        # 创建一个 Box 对象
        # 表示一个边界框（Bounding Box）
        box = Box(
            ann_info['translation'],
            ann_info['size'],
            Quaternion(ann_info['rotation']),
            velocity=ann_info['velocity'],
        )
        
        # 进行平移和旋转变换
        box.translate(trans)
        box.rotate(rot)
        
        # 获取边界框的中心坐标、宽高深度以及偏航角和速度
        box_xyz = np.array(box.center)
        box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
        box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
        box_velo = np.array(box.velocity[:2])
        
        # 将边界框的信息拼接到列表中
        gt_box = np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])
        gt_boxes.append(gt_box)
        
        # 获取边界框对应的类别标签并添加到列表中
        gt_labels.append(
            classes.index(
                map_name_from_general_to_detection[ann_info['category_name']]
            )
        )

    return gt_boxes, gt_labels


def nuscenes_data_prep(root_path, info_prefix, version, max_sweeps=10):
    """
    准备与nuScenes数据集相关的数据。
    相关数据包括记录基本信息、2D标注以及地面真实数据库的.pkl文件。

    参数:
        root_path (str): 数据集根目录的路径。
        info_prefix (str): info文件名的前缀。
        version (str): 数据集版本。
        max_sweeps (int, 可选): 输入连续帧的数量。默认值为10。
    """
    nuscenes_converter.create_nuscenes_infos(
        root_path, info_prefix, version=version, max_sweeps=max_sweeps)

def add_ann_adj_info(extra_tag):
    nuscenes_version = 'v1.0-mini'
    dataroot = './data/nuscenes-mini/'
    nuscenes = NuScenes(nuscenes_version, dataroot)
    
    # 遍历训练集和验证集
    for set in ['train', 'val']:
        # 加载对应的info文件
        dataset = pickle.load(
            open('./data/nuscenes-mini/%s_infos_%s.pkl' % (extra_tag, set), 'rb'))
        
        # 遍历info列表
        for id in range(len(dataset['infos'])):
            # 每隔10个info打印进度
            if id % 10 == 0:
                print('%d/%d' % (id, len(dataset['infos'])))
            info = dataset['infos'][id]
            
            # 获取扫描连续帧的信息
            sample = nuscenes.get('sample', info['token'])
            ann_infos = list()
            # 遍历每个注释
            for ann in sample['anns']:
                ann_info = nuscenes.get('sample_annotation', ann)
                # 获取注释对应的速度信息
                velocity = nuscenes.box_velocity(ann_info['token'])
                if np.any(np.isnan(velocity)):
                    velocity = np.zeros(3)
                ann_info['velocity'] = velocity
                ann_infos.append(ann_info)
            # 将注释信息添加到对应的info中
            dataset['infos'][id]['ann_infos'] = ann_infos
            dataset['infos'][id]['ann_infos'] = get_gt(dataset['infos'][id])
            dataset['infos'][id]['scene_token'] = sample['scene_token']
            
        # 将更新后的dataset保存到文件中
        with open('./data/nuscenes-mini/%s_infos_%s.pkl' % (extra_tag, set),
                  'wb') as fid:
            pickle.dump(dataset, fid)


if __name__ == '__main__':
    dataset = 'nuscenes'
    version = 'v1.0-mini'
    train_version = f'{version}'
    root_path = './data/nuscenes-mini'
    extra_tag = 'bevdetv2-nuscenes'
    nuscenes_data_prep(
        root_path=root_path,
        info_prefix=extra_tag,
        version=train_version,
        max_sweeps=0)

    print('add_ann_infos')
    add_ann_adj_info(extra_tag)
