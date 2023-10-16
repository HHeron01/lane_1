# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.parallel import DataContainer as DC

from mmdet3d.core.bbox import BaseInstance3DBoxes
from mmdet3d.core.points import BasePoints
from mmdet.datasets.pipelines import to_tensor
from ..builder import PIPELINES


@PIPELINES.register_module()
class DefaultFormatBundle(object):
    """默认格式捆绑。
    它简化了格式化常见字段的流程，包括 "img"、"proposals"、"gt_bboxes"、
    "gt_labels"、"gt_masks" 和 "gt_semantic_seg"。这些字段的格式如下。

    - img: (1) 转置，(2) 转为张量，(3) 转为 DataContainer（stack=True）
    - proposals: (1) 转为张量，(2) 转为 DataContainer
    - gt_bboxes: (1) 转为张量，(2) 转为 DataContainer
    - gt_bboxes_ignore: (1) 转为张量，(2) 转为 DataContainer
    - gt_labels: (1) 转为张量，(2) 转为 DataContainer
    - gt_masks: (1) 转为张量，(2) 转为 DataContainer（cpu_only=True）
    - gt_semantic_seg: (1) 在 dim-0 上添加维度，(2) 转为张量，
                       (3) 转为 DataContainer（stack=True）
    """

    def __init__(self, ):
        return

    def __call__(self, results):
        """调用函数来转换和格式化results中的常见字段。
        参数:
            results (dict): 包含要转换数据的结果字典。
        返回:
            dict: 结果字典包含使用默认捆绑格式化的数据。
        """
        if 'img' in results:
            if isinstance(results['img'], list):
                # 处理单帧中的多个图像
                imgs = [img.transpose(2, 0, 1) for img in results['img']]
                imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
                results['img'] = DC(to_tensor(imgs), stack=True)
            else:
                # 转置图像通道并转为张量，然后转为 DataContainer
                img = np.ascontiguousarray(results['img'].transpose(2, 0, 1))
                results['img'] = DC(to_tensor(img), stack=True)
        
        # 针对以下字段进行格式化
        for key in [
                'proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels',
                'gt_labels_3d', 'attr_labels', 'pts_instance_mask',
                'pts_semantic_mask', 'centers2d', 'depths'
        ]:
            """
            'proposals': 包含目标检测模型生成的候选物体提议框（bounding boxes）的数据。
            'gt_bboxes': 包含图像中真实目标的边界框坐标，通常作为训练数据的一部分。
            'gt_bboxes_ignore': 用于排除某些边界框不参与训练的数据，通常用于处理难以分类的目标。
            'gt_labels': 包含与真实目标关联的类别标签，用于目标分类任务。
            'gt_labels_3d': 包含与真实目标关联的三维类别标签，通常在三维感知任务中使用。
            'attr_labels': 包含目标属性标签的数据，用于描述目标的特定属性或特征。
            'pts_instance_mask': 用于实例分割任务，包含了每个点云点对应的实例分割掩码。
            'pts_semantic_mask': 用于语义分割任务，包含了每个点云点对应的语义分割掩码。
            'centers2d': 包含图像中目标的中心点坐标，通常用于目标中心点回归任务。
            'depths': 包含点云或深度图像中的深度信息，通常在三维感知任务中使用。
            """
            if key not in results:
                continue
            if isinstance(results[key], list):
                # 转为张量并转为 DataContainer
                results[key] = DC([to_tensor(res) for res in results[key]])
            else:
                # 转为张量并转为 DataContainer
                results[key] = DC(to_tensor(results[key]))
        
        # 针对3D边界框字段进行格式化
        if 'gt_bboxes_3d' in results:
            if isinstance(results['gt_bboxes_3d'], BaseInstance3DBoxes):
                # 如果是 BaseInstance3DBoxes，则转为 DataContainer（仅在CPU上操作）
                results['gt_bboxes_3d'] = DC(
                    results['gt_bboxes_3d'], cpu_only=True)
            else:
                # 转为张量并转为 DataContainer
                results['gt_bboxes_3d'] = DC(
                    to_tensor(results['gt_bboxes_3d']))

        # 针对 gt_masks 字段进行格式化，转为 DataContainer（仅在CPU上操作）
        if 'gt_masks' in results:
            results['gt_masks'] = DC(results['gt_masks'], cpu_only=True)
        
        # 针对 gt_semantic_seg 字段进行格式化，添加维度并转为张量，最后转为 DataContainer（进行堆叠）
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None, ...]), stack=True)

        return results

    def __repr__(self):
        return self.__class__.__name__

@PIPELINES.register_module()
class Collect3D(object):
    """从加载器中收集与特定任务相关的数据。
    这通常是数据加载器管道的最后阶段。通常，键被设置为“img”，“proposals”，“gt_bboxes”，“gt_bboxes_ignore”，“gt_labels”和/或“gt_masks”的某个子集。
    始终会填充“img_meta”项。 “img_meta”字典的内容取决于“meta_keys”。默认情况下，包括以下内容：
        - 'img_shape'：输入到网络的图像的形状，表示为元组（h，w，c）。请注意，如果批处理张量大于此形状，则图像可能在底部/右侧进行零填充。
        - 'scale_factor'：表示预处理比例的浮点数
        - 'flip'：指示是否使用了图像翻转变换的布尔值
        - 'filename'：图像文件的路径
        - 'ori_shape'：图像的原始形状，表示为元组（h，w，c）
        - 'pad_shape'：填充后的图像形状
        - 'lidar2img'：从激光雷达到图像的变换
        - 'depth2img'：从深度到图像的变换
        - 'cam2img'：从相机到图像的变换
        - 'pcd_horizontal_flip'：指示点云是否水平翻转的布尔值
        - 'pcd_vertical_flip'：指示点云是否垂直翻转的布尔值
        - 'box_mode_3d'：3D框模式
        - 'box_type_3d'：3D框类型
        - 'img_norm_cfg'：标准化信息的字典：
            - 'mean'：每个通道的均值减法
            - 'std'：每个通道的标准差除数
            - 'to_rgb'：指示是否将BGR转换为RGB的布尔值
        - 'pcd_trans'：点云变换
        - 'sample_idx'：样本索引
        - 'pcd_scale_factor'：点云比例因子
        - 'pcd_rotation'：应用于点云的旋转
        - 'pts_filename'：点云文件的路径。
    参数：
        keys (Sequence[str]）：要在“data”中收集的结果的键。
        meta_keys (Sequence[str]，可选）：要转换为“mmcv.DataContainer”并在“data[img_metas]”中收集的元数据键。
        默认值：（'filename'，'ori_shape'，'img_shape'，'lidar2img'，
                'depth2img'，'cam2img'，'pad_shape'，'scale_factor'，'flip'，
                'pcd_horizontal_flip'，'pcd_vertical_flip'，'box_mode_3d'，
                'box_type_3d'，'img_norm_cfg'，'pcd_trans'，
                'sample_idx'，'pcd_scale_factor'，'pcd_rotation'，'pts_filename'）
"""

    def __init__(
        self,
        keys,
        meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                   'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
                   'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
                   'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx',
                   'pcd_scale_factor', 'pcd_rotation', 'pcd_rotation_angle',
                   'pts_filename', 'transformation_3d_flow', 'trans_mat',
                   'affine_aug')):
        """
        图像信息：
        'filename'：图像文件的路径或文件名。
        'ori_shape'：原始图像的形状（高度、宽度、通道数）。
        'img_shape'：输入网络的图像形状（高度、宽度、通道数）。
        'pad_shape'：经过填充后的图像形状（高度、宽度、通道数）。
        'scale_factor'：预处理尺度因子。
        'flip'：图像是否进行了水平翻转。
        'img_norm_cfg'：图像的归一化配置，包括均值、标准差等。
        
        坐标变换信息：
        'lidar2img'：从激光雷达坐标到图像坐标的变换矩阵。
        'depth2img'：从深度图到图像坐标的变换矩阵。
        'cam2img'：从相机坐标到图像坐标的变换矩阵。
        
        数据增强信息：
        'pcd_horizontal_flip'：点云是否进行了水平翻转。
        'pcd_vertical_flip'：点云是否进行了垂直翻转。
        'box_mode_3d'：3D 边界框的模式。
        'box_type_3d'：3D 边界框的类型。
        'pcd_trans'：点云的变换信息。
        'sample_idx'：样本索引。
        'pcd_scale_factor'：点云的尺度因子。
        'pcd_rotation'：点云的旋转信息。
        'pcd_rotation_angle'：点云旋转的角度信息。
        
        其他信息：
        'pts_filename'：点云文件的路径或文件名。
        'transformation_3d_flow'：3D 变换流信息。
        'trans_mat'：仿射变换矩阵。
        'affine_aug'：仿射数据增强信息。
        """
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """调用函数来收集results中的键。在“meta_keys”中的键将被转换为:obj:`mmcv.DataContainer`。
        参数：
            results (dict)：结果字典，包含要收集的数据。
        返回值：
            dict：结果字典包含以下键
                - 在“self.keys”中的键
                - “img_metas”
        """
        
        data = {}  # 创建一个空字典用于存储结果数据
        img_metas = {}  # 创建一个空字典用于存储图像元信息
        
        # 遍历指定的元信息键（meta_keys），如果在results中存在，就将其存储在img_metas中
        for key in self.meta_keys:
            if key in results:
                img_metas[key] = results[key]

        # 将img_metas转换为DataContainer类型并存储在data字典中
        data['img_metas'] = DC(img_metas, cpu_only=True)
        
        # 遍历需要收集的数据键（keys），将它们从results中存储在data字典中
        for key in self.keys:
            data[key] = results[key]
        
        return data  # 返回包含收集数据的字典

    def __repr__(self):
        """str: 返回描述该模块的字符串。"""
        return self.__class__.__name__ + f'(keys={self.keys}, meta_keys={self.meta_keys})'

@PIPELINES.register_module()
class DefaultFormatBundle3D(DefaultFormatBundle):
    """默认的3D数据格式化处理类。
    这个类继承自DefaultFormatBundle类，用于处理3D数据的格式化。
    Args:
        class_names (list): 类别名称列表。
        with_gt (bool): 是否包含ground-truth信息，默认为True。
        with_label (bool): 是否包含标签信息，默认为True。
    """

    def __init__(self, class_names, with_gt=True, with_label=True):
        # 调用父类的构造函数
        super(DefaultFormatBundle3D, self).__init__()
        # 设置类的成员变量
        self.class_names = class_names  # 类别名称列表
        self.with_gt = with_gt  # 是否包含ground-truth信息
        self.with_label = with_label  # 是否包含标签信息

    def __call__(self, results):
        """调用函数来转换和格式化results中的常见字段。
        Args:
            results (dict): 结果字典，包含要转换的数据。
        Returns:
            dict: 结果字典包含使用默认格式化处理的数据。
        """
        # 格式化3D数据
        if 'points' in results:
            assert isinstance(results['points'], BasePoints)
            # 将3D点云数据转换为DataContainer对象
            results['points'] = DC(results['points'].tensor)

        # 针对每个键在列表['voxels', 'coors', 'voxel_centers', 'num_points']上进行循环
        for key in ['voxels', 'coors', 'voxel_centers', 'num_points']:
            if key not in results:
                continue
            # 将数据转换为tensor，并创建DataContainer对象，不进行stack操作
            results[key] = DC(to_tensor(results[key]), stack=False)

        if self.with_gt:
            # 清除GT边界框信息
            if 'gt_bboxes_3d_mask' in results:
                gt_bboxes_3d_mask = results['gt_bboxes_3d_mask']
                # 根据3D GT边界框的掩码清除不需要的边界框信息
                results['gt_bboxes_3d'] = results['gt_bboxes_3d'][gt_bboxes_3d_mask]
                if 'gt_names_3d' in results:
                    results['gt_names_3d'] = results['gt_names_3d'][gt_bboxes_3d_mask]
                if 'centers2d' in results:
                    results['centers2d'] = results['centers2d'][gt_bboxes_3d_mask]
                if 'depths' in results:
                    results['depths'] = results['depths'][gt_bboxes_3d_mask]
            if 'gt_bboxes_mask' in results:
                gt_bboxes_mask = results['gt_bboxes_mask']
                if 'gt_bboxes' in results:
                    # 根据GT边界框的掩码清除不需要的边界框信息
                    results['gt_bboxes'] = results['gt_bboxes'][gt_bboxes_mask]
                # 根据GT边界框的掩码清除不需要的GT名称信息
                results['gt_names'] = results['gt_names'][gt_bboxes_mask]
            if self.with_label:
                if 'gt_names' in results and len(results['gt_names']) == 0:
                    # 如果GT名称为空，则设置空的标签和属性标签
                    results['gt_labels'] = np.array([], dtype=np.int64)
                    results['attr_labels'] = np.array([], dtype=np.int64)
                elif 'gt_names' in results and isinstance(
                        results['gt_names'][0], list):
                    # 如果GT名称是一个列表的列表（多视角设置），则为每个子列表创建标签
                    results['gt_labels'] = [
                        np.array([self.class_names.index(n) for n in res],
                                dtype=np.int64) for res in results['gt_names']
                    ]
                elif 'gt_names' in results:
                    # 否则，为每个GT名称创建标签
                    results['gt_labels'] = np.array([
                        self.class_names.index(n) for n in results['gt_names']
                    ],
                                                    dtype=np.int64)
                if 'gt_names_3d' in results:
                    # 创建3D GT名称的标签
                    results['gt_labels_3d'] = np.array([
                        self.class_names.index(n) for n in results['gt_names_3d']
                    ],
                                                    dtype=np.int64)
        # 调用父类的__call__方法来继续处理结果
        results = super(DefaultFormatBundle3D, self).__call__(results)
        # 返回处理后的结果
        return results

    def __repr__(self):
        """str: 返回描述该模块的字符串。"""
        repr_str = self.__class__.__name__
        repr_str += f'(class_names={self.class_names}, '
        repr_str += f'with_gt={self.with_gt}, with_label={self.with_label})'
        return repr_str
