# Copyright (c) Phigent Robotics. All rights reserved.

dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_0207'
load_from = '/home/slj/Documents/workspace/mmdet3d/work_0206_1/epoch_60_ema.pth'
resume_from = None
no_validate = True


#for IterBasedRunner
# runner = dict(type='IterBasedRunner', max_epochs=24) #IterBasedRunner
# workflow = [('train', 20), ('val', 5)] #for IterBasedRunner
# max_loops = 50 #for IterBasedRunner

#for EpochBasedRunner
runner = dict(type='EpochBasedRunner', max_epochs=60)
workflow = [('train', 1), ('val', 1)]#, ('test', 1)]
max_loops = 60

checkpoint_config = dict(interval=500)
# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

# For nuScenes we usually do 10-class detection
class_names = []

data_config = {
    'cams': [],
    'Ncams': 1,
    'input_size': (960, 640),
    'src_size': (1920, 1280),
    'thickness': 2,
    'angle_class': 36,

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': False,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,

}

# Model
grid_config = {
    'x': [-19.2, 19.2, 0.2],
    'y': [0.0, 96.0, 0.3],
    'z': [-5, 3, 8],
    'grid_height': 6,
    'depth': [1.0, 60.0, 1.0],
    'grid_res': [0.2, 0.3, 0.6],
    'offset': [-19.2, 0.0, 1.08],
    'grid_size': [38.4, 96],
    'depth_discretization': 'LID',
    'dbound': [4.0, 70.0, 0.6]
}

# voxel_size = [0.1, 0.1, 0.2]

numC_Trans = 64

res_downsample = 16

model = dict(
    type='AnchorBEVLane',
    img_backbone=dict(
        pretrained=None,#'torchvision://resnet50',
        type='ResNet',
        depth=34,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[256, 512],#[1024, 2048],
        out_channels=64,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
    #     type='Lift',
    #     grid_cfg=grid_config,
    #     img_size=data_config['input_size'],
    #     downsample=res_downsample,
    #     channel_in=512,
    #     channel_out=64,
    # ),
        type='OFTV3',
        grid_res=grid_config['grid_res'],
        grid_height=grid_config['grid_height'],
        scale=res_downsample,
        add_coor=False,
        debug=None,

    ),
        # type='LSSViewTransformer',
        # grid_config=grid_config,
        # input_size=data_config['input_size'],
        # in_channels=512,
        # out_channels=numC_Trans,
        # downsample=16),
    img_bev_encoder_backbone=dict(
        type='CustomResNet',
        numC_input=numC_Trans,
        num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8],
        oft=True,
        height=9
    ),
    img_bev_encoder_neck=dict(
        type='FPN_LSS',
        in_channels=numC_Trans * 8 + numC_Trans * 2,
        out_channels=256,
        extra_upsample=4,
    ),
    bev_lane_head=dict(
        type='Gen_BevLaneHead',
        in_channel=256,
        num_classes=1,
        grid_config=grid_config,
    ))

# Data
dataset_type = 'OpenLane_Anchor_Dataset'#'IterBasedRunner', EpochBasedRunner
images_dir = '/home/slj/data/openlane/openlane_all/images'
json_file_dir = '/home/slj/data/openlane/openlane_all/lane3d_300/training/'
file_client_args = dict(backend='disk')

# only for bev features aug conf
bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

train_pipeline = [
    dict(
        type='LoadAnnotationsBevAnchor',
        bda_aug_conf=bda_aug_conf,
        classes=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'targets'], meta_keys=['file_path', 'origin_lanes'], )
]

test_pipeline = [
    dict(
        type='LoadAnnotationsBevAnchor',
        bda_aug_conf=None,
        classes=class_names,
        is_train=False),
    dict(
        type='Collect3D', keys=['img_inputs', 'targets'], meta_keys=[ 'file_path', 'origin_lanes'])
]

eval_pipeline = [
    dict(
        type='LoadAnnotationsBevAnchor',
        bda_aug_conf=None,
        classes=class_names,
        is_train=False),
    dict(
        type='Collect3D', keys=['img_inputs', 'targets'], meta_keys=['file_path', 'origin_lanes'])
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
    images_dir=images_dir,
    json_file_dir=json_file_dir,
    data_config=data_config,
    grid_config=grid_config,
    pipeline=test_pipeline,
    CLASSES=None,
    test_mode=True,
    use_gen_anchor=True,
    use_valid_flag=False,
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    shuffle=True,
    train=dict(
        images_dir=images_dir,
        json_file_dir=json_file_dir,
        data_config=data_config,
        grid_config=grid_config,
        pipeline=train_pipeline,
        CLASSES=None,
        use_gen_anchor=True,
        # classes=class_names,
        test_mode=False,
        use_valid_flag=False,
    ),
    val=test_data_config,
    test=test_data_config)

# evaluation = dict(interval=1, pipeline=eval_pipeline)

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

#/home/slj/Documents/workspace/mmdet3d/data/openlane/example/image/validation/segment-260994483494315994_2797_545_2817_545_with_camera_labels
