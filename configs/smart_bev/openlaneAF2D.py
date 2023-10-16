      
# Copyright (c) Phigent Robotics. All rights reserved.

dist_params = dict(backend='nccl')
# 设置分布式参数，指定使用的后端为'nccl'
log_level = 'INFO'
# 设置日志级别为'INFO'
work_dir = './result/work_AF_test'
# 设置工作目录为'./result/work_AF_test'
load_from =  None
# 设置加载模型的路径为 None，即不加载已经训练好的模型
resume_from = None
# 设置从断点恢复的路径为 None，即不从任何断点处恢复
no_validate = True
# 设置为 True，表示不进行验证步骤

#for IterBasedRunner
# runner = dict(type='IterBasedRunner', max_epochs=24) #IterBasedRunner
# workflow = [('train', 20), ('val', 5)] #for IterBasedRunner
# max_loops = 50 #for IterBasedRunner

#for EpochBasedRunner
runner = dict(type='EpochBasedRunner', max_epochs=3)
workflow = [('train', 1), ('val', 1)]#, ('test', 1)]
max_loops = 3

checkpoint_config = dict(interval=500)
# 检查点保存的配置，设置间隔为500步保存一次检查点
# 模型的权重以及一些其他相关的训练参数，可以用于恢复训练、评估模型或在未来进行推理。
opencv_num_threads = 0
# 禁用OpenCV的多线程以避免系统负载过重
mp_start_method = 'fork'
# 将多进程的启动方法设置为'fork'以加快训练速度
# windows一般为'spawn'
class_names = []
# 类别名称，这里初始化为空列表

data_config = {
    'cams': [],             # 这是一个相机列表，当前为空。它可能用于存储有关每个相机的信息或参数。
    'Ncams': 1,             # 这是相机的数量。目前配置只有一个相机。
    'input_size': (960, 640),   # 这是模型的输入图像大小。宽度是960像素，高度是640像素。
    'src_size': (1920, 1280),   # 这是原始图像的大小或解析度，可能在处理之前。宽度是1920像素，高度是1280像素。
    'thickness': 2,         # 这可能是一些图像处理的参数，但具体的上下文不明。
    'angle_class': 36,      # 这可能指的是角度的分类或离散化的数量。

    # 数据增强参数
    'resize': (-0.06, 0.11),   # 调整图像大小的范围。这可以是一种数据增强方法，允许图像在给定的范围内缩小或放大。
    'rot': (-5.4, 5.4),       # 旋转图像的角度范围，为了数据增强。
    'flip': False,            # 是否进行图像翻转的标志。
    'crop_h': (0.0, 0.0),     # 图像垂直裁剪的范围。目前它是0，意味着没有裁剪。
    'resize_test': 0.00,      # 这可能是在测试阶段调整图像大小的参数。
}

grid_config = {
    'x': [-19.2, 19.2, 0.2],    # x轴的范围和间距。从-19.2到19.2，每0.2一个间隔。
    'y': [0.0, 96.0, 0.3],     # y轴的范围和间距。从0.0到96.0，每0.3一个间隔。
    'z': [-5, 3, 8],           # z轴的范围和间距。这个配置有些奇怪，因为从-5到3的范围与8的间隔不太匹配。
    'grid_height': 6,          # 网格的高度。
    'depth': [1.0, 60.0, 1.0], # 深度的范围和间距。从1.0到60.0，每1.0一个间隔。
    'grid_res': [0.2, 0.2, 0.6], # 3D网格的分辨率，分别为x、y、z方向。
    'offset': [-19.2, 0.0, 1.08], # 网格的起始偏移量。
    'grid_size': [38.4, 96],   # 网格的大小，这可能是x和y方向上的总长度。
    'depth_discretization': 'LID', # 深度的离散化方法。
    'dbound': [4.0, 70.0, 0.6]  # 深度边界的范围和间隔。
}

# model
model = dict(
    type='LaneAF2DDetector',  # 模型的类型或名称，它叫做"LaneAF2DDetector"，可能是一个2D车道检测器。
        img_backbone=dict(
        pretrained='torchvision://resnet34',  # 使用预训练的resnet34模型。
        type='ResNet',  # 主干网络的类型是ResNet。
        depth=34,  # ResNet的深度，这里使用的是ResNet-34。
        num_stages=4,  # ResNet的阶段数量。
        out_indices=(0, 1, 2, 3),  # 输出的阶段索引。
        frozen_stages=-1,  # 冻结的阶段，-1意味着不冻结任何阶段。
        norm_cfg=dict(type='BN', requires_grad=True),  # 使用批量归一化(Batch Normalization)。
        norm_eval=False,  # 在评估模式下不使用BN。
        with_cp=True,  # 是否使用checkpoint，有助于节省显存。
        style='pytorch'  # 使用PyTorch风格的ResNet。
    ),
        img_neck=dict(
        type='CustomFPN',  # 使用自定义的特征金字塔网络(Feature Pyramid Network, FPN)。
        in_channels=[64, 128, 256, 512],  # 输入通道数，与ResNet-34的输出匹配。
        out_channels=256,  # FPN的输出通道数。
        num_outs=1,  # 输出的数量。
        start_level=0,  # 开始的层级。
        out_ids=[0]  # 输出的id。
    ),
        lane_head=dict(
        type='LaneAF2DHead',  # 车道头部的类型。
        in_channel=256,  # 输入通道数，与FPN的输出匹配。
        num_classes=1,  # 类别数。这里设置为1，可能是二进制分类（车道/非车道）。
        seg_loss=dict(type='Lane_FocalLoss'),  # 用于分割的损失函数，这里使用特定的"FocalLoss"。
        haf_loss=dict(type='RegL1Loss'),  # 可能与车道的水平对齐有关的损失函数。
        vaf_loss=dict(type='RegL1Loss'),  # 可能与车道的垂直对齐有关的损失函数。
    ))

# Data
dataset_type = 'OpenLane_Dataset_AF' #'IterBasedRunner', EpochBasedRunner
images_dir = '/workspace/openlane_all/images'
json_file_dir_train = '/workspace/openlane_all/lane3d_300/training/'
json_file_dir_val = '/workspace/openlane_all/lane3d_300/validation/'
file_client_args = dict(backend='disk')
"""
LoadAnnotationsSeg类型的数据加载步骤用于加载标注数据，其中包括了类别信息、语义分割标签等。bda_aug_conf参数为None，表示没有使用数据增强。classes参数指定了类别名称。is_train参数为False，表示这个步骤用于测试和评估。

Collect3D类型的数据收集步骤用于将处理后的数据组合成一个字典，其中包括了以下键值对：
'img_inputs'：用于输入网络的图像数据。
'maps_bev'：Bird's Eye View（BEV）地图数据。
'maps_2d'：2D地图数据。
'gt_lanes_3d'：3D车道线标签数据。
'gt_lanes_2d'：2D车道线标签数据。
'file_path'：文件路径信息。
"""
train_pipeline = [
    dict(
        type='LoadAnnotationsSeg',
        bda_aug_conf=None,
        classes=class_names,
        is_train=False),
    # 返回 img_inputs=(imgs, intrins, extrinsics, post_rots, post_trans, undists,
                                # bda_rot, rots, trans, grid, drop_idx)
    # maps_bev,maps_2d,gt_lanes_3d,gt_lanes_2d,file_path
    dict(
        type='Collect3D', 
        keys=['img_inputs', 'maps_bev', 'maps_2d'], 
        meta_keys=['gt_lanes_3d', 'gt_lanes_2d', 'file_path'])
]
test_pipeline = [
    dict(
        type='LoadAnnotationsSeg',
        bda_aug_conf=None,
        classes=class_names,
        is_train=False),

    dict(
        type='Collect3D', 
        keys=['img_inputs', 'maps_bev', 'maps_2d'], 
        meta_keys=['gt_lanes_3d', 'gt_lanes_2d', 'file_path'])
]
eval_pipeline = [
    dict(
        type='LoadAnnotationsSeg',
        bda_aug_conf=None,
        classes=class_names,
        is_train=False),

    dict(
        type='Collect3D', 
        keys=['img_inputs', 'maps_bev', 'maps_2d'], 
        meta_keys=['gt_lanes_3d', 'gt_lanes_2d', 'file_path'])
]

# 定义输入模态性（Input Modality），用于指定哪些传感器数据将被使用
input_modality = dict(
    use_lidar=False,     # 是否使用激光雷达数据
    use_camera=True,     # 是否使用摄像头数据
    use_radar=False,     # 是否使用雷达数据
    use_map=False,       # 是否使用地图数据
    use_external=False   # 是否使用外部数据
)

# 定义共享数据配置，其中包括数据集类型和模态性配置
share_data_config = dict(
    type=dataset_type,    # 数据集的类型
    # modality=input_modality,  # 输入模态性配置（可选择是否启用）
)

# 定义测试数据配置，包括图像目录、JSON文件目录、数据配置、网格配置、数据处理流程等
test_data_config = dict(
    images_dir=images_dir,            # 图像数据的目录
    json_file_dir=json_file_dir_val,  # JSON文件数据的目录
    data_config=data_config,          # 数据配置
    grid_config=grid_config,          # 网格配置
    pipeline=test_pipeline,           # 数据处理流程
    CLASSES=None,                     # 类别信息（在测试模式下一般不需要）
    test_mode=True,                   # 是否处于测试模式
    use_valid_flag=False,             # 是否使用验证标志（一般在测试模式下不使用）
)

# 定义数据配置字典，包括每个GPU的样本数、每个GPU的工作进程数、是否洗牌等
data = dict(
    samples_per_gpu=32,          # 每个GPU的样本数
    workers_per_gpu=0,          # 每个GPU的工作进程数
    shuffle=True,               # 是否在数据加载时洗牌

    # 训练数据配置，包括图像目录、JSON文件目录、数据配置、网格配置、数据处理流程等
    train=dict(
        images_dir=images_dir,            # 图像数据的目录
        json_file_dir=json_file_dir_train, # JSON文件数据的目录
        data_config=data_config,          # 数据配置
        grid_config=grid_config,          # 网格配置
        pipeline=train_pipeline,           # 数据处理流程
        CLASSES=None,                     # 类别信息（在训练模式下一般不需要）
        test_mode=False,                  # 是否处于测试模式
        use_valid_flag=False,             # 是否使用验证标志（一般在训练模式下不使用）
    ),

    # 验证数据配置，使用了之前定义的测试数据配置
    val=test_data_config,

    # 测试数据配置，也使用了之前定义的测试数据配置
    test=test_data_config
)

# evaluation = dict(interval=1, pipeline=eval_pipeline)

for key in ['train', 'val', 'test']:
    data[key].update(share_data_config)

# 定义优化器配置字典，包括优化器的类型、学习率和权重衰减等参数
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=1e-07)
# 定义优化器配置字典，包括梯度裁剪的配置，这里设置了梯度裁剪的最大范数和裁剪类型
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))

# 定义学习率配置字典，包括学习率策略、预热策略和预热参数等
lr_config = dict(
    policy='step',                # 学习率策略，这里是按步骤调整学习率
    warmup='linear',              # 预热策略，这里是线性预热
    warmup_iters=200,             # 预热迭代次数
    warmup_ratio=0.001,           # 预热比例
    step=[24,]                    # 学习率调整的步骤，这里在第24个迭代时调整学习率
)

log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

custom_hooks = [
    # dict(
    #     type='MEGVIIEMAHook',
    #     init_updates=10560,
    #     priority='NORMAL',
    # ),
    dict(
        type='BevLaneVisAllHook',
    ),
]

# unstable
# fp16 = dict(loss_scale='dynamic')
