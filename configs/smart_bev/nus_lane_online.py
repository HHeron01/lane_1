# Copyright (c) Phigent Robotics. All rights reserved.

dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
no_validate = True


#for IterBasedRunner
# runner = dict(type='IterBasedRunner', max_epochs=24) #IterBasedRunner
# workflow = [('train', 20), ('val', 5)] #for IterBasedRunner
# max_loops = 50 #for IterBasedRunner

#for EpochBasedRunner
runner = dict(type='EpochBasedRunner', max_epochs=24)
workflow = [('train', 1), ('val', 1)]#, ('test', 1)]
max_loops = 24

checkpoint_config = dict(interval=500)
# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams': 6,
    'input_size': (128, 128),
    'src_size': (900, 1600),
    'thickness': 5,
    'angle_class': 36,

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,

}

# Model
grid_config = {
    'x': [-15.0, 15.0, 0.15],
    'y': [-30.0, 30.0, 0.15],
    'z': [-5, 3, 8],
    'depth': [1.0, 60.0, 1.0],
}

voxel_size = [0.1, 0.1, 0.2]

numC_Trans = 80

model = dict(
    type='BEVLane',
    img_backbone=dict(
        pretrained=None,#'torchvision://resnet50',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=512,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='LSSViewTransformer',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=512,
        out_channels=numC_Trans,
        downsample=16),
    img_bev_encoder_backbone=dict(
        type='CustomResNet',
        numC_input=numC_Trans,
        num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8]),
    img_bev_encoder_neck=dict(
        type='FPN_LSS',
        in_channels=numC_Trans * 8 + numC_Trans * 2,
        out_channels=256),
    bev_lane_head=dict(
        type='BevLaneHead',
        in_channel=256,
        num_classes=4,
        off_loss=dict(type='OffsetLoss'),
        seg_loss=dict(type='Lane_FocalLoss'),
        haf_loss=dict(type='RegL1Loss'),
        vaf_loss=dict(type='RegL1Loss'),
    ))

# Data
dataset_type = 'Nus_online_SegDataset'#'IterBasedRunner', EpochBasedRunner
data_root = 'data/nuscenes-mini/'
file_client_args = dict(backend='disk')

# only for bev features aug conf
bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

train_pipeline = [
    dict(
        type='LoadAnnotationsBevseg',
        bda_aug_conf=bda_aug_conf,
        classes=class_names),

    dict(
        type='Collect3D', keys=['img_inputs', 'maps'])
]

test_pipeline = [
    dict(
        type='LoadAnnotationsBevseg',
        bda_aug_conf=None,
        classes=class_names,
        is_train=False),
    dict(type='Collect3D', keys=['img_inputs', 'maps'])
]

eval_pipeline = [
    dict(
        type='LoadAnnotationsBevseg',
        bda_aug_conf=None,
        classes=class_names,
        is_train=False),
    dict(type='Collect3D', keys=['img_inputs', 'maps'])
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    # modality=input_modality,
)

test_data_config = dict(
    pipeline=test_pipeline,
    data_root=data_root,
    data_config=data_config,
    grid_config=grid_config,
    CLASSES=None,
    test_mode=True,
    use_valid_flag=False,
    # box_type_3d='LiDAR',
    version='v1.0-mini',
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    shuffle=True,
    train=dict(
        data_root=data_root,
        pipeline=train_pipeline,
        data_config=data_config,
        grid_config=grid_config,
        CLASSES=None,
        # classes=class_names,
        test_mode=False,
        use_valid_flag=False,
        # modality=None,
        # box_type_3d='LiDAR',
        version='v1.0-mini',
    ),
    val=test_data_config,
    test=test_data_config)

evaluation = dict(interval=1, pipeline=eval_pipeline)

for key in ['train', 'val', 'test']:
    data[key].update(share_data_config)

# Optimizer
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=1e-07)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[24,])

log_config = dict(
    interval=4,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
    dict(
        type='BevLaneVisAllHook',
    ),
]



# unstable
# fp16 = dict(loss_scale='dynamic')
