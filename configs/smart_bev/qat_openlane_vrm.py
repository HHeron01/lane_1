# Copyright (c) Phigent Robotics. All rights reserved.
# import albumentations as A
import torch
from torch import nn
dist_params = dict(backend='nccl')
log_level = 'INFO'
# work_dir = './work_dirs_0410_qat'
work_dir = './work_dirs_0424'
load_from = '/home/slj/Documents/workspace/ThomasVision/work_dirs_0421/epoch_29_ema.pth'
resume_from = None
no_validate = True


#for IterBasedRunner
# runner = dict(type='IterBasedRunner', max_epochs=24) #IterBasedRunner
# workflow = [('train', 20), ('val', 5)] #for IterBasedRunner
# max_loops = 50 #for IterBasedRunner

#for EpochBasedRunner
runner = dict(type='EpochBasedRunner', max_epochs=100)
workflow = [('train', 20), ('val', 1)]#, ('test', 1)]
max_loops = 100

checkpoint_config = dict(interval=500)
# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

input_shape = (576, 1024) #h, w
output_2d_shape = (144, 256)

# train_trans = A.Compose([
#     A.Resize(height=input_shape[0], width=input_shape[1]),
#     A.MotionBlur(p=0.2),
#     A.RandomBrightnessContrast(),
#     A.ColorJitter(p=0.1),
#     A.Normalize(),
#     ToTensorV2()
# ])

'''Post-processing parameters '''
post_conf = -0.1  # Minimum confidence on the segmentation map for clustering
post_emb_margin = 6.0 # embeding margin of different clusters
post_min_cluster_size = 15 # The minimum number of points in a cluster

''' BEV range '''
x_range = (3, 103)
y_range = (-12, 12)

# y_range = (3, 103)
# x_range = (-12, 12)
meter_per_pixel = 0.5 # grid size
bev_shape = (int((x_range[1] - x_range[0]) / meter_per_pixel), int((y_range[1] - y_range[0]) / meter_per_pixel))


''' virtual camera config '''
vc_config = {}
vc_config['use_virtual_camera'] = True
vc_config['vc_intrinsic'] = [[2081.5212033927246, 0.0, 934.7111248349433],
                                    [0.0, 2081.5212033927246, 646.3389987785433],
                                    [0.0, 0.0, 1.0]]
vc_config['vc_extrinsics'] = [[-0.002122161262459438, 0.010697496358766389, 0.9999405282331697, 1.5441039498273286],
            [-0.9999378331046326, -0.010968621415360667, -0.0020048117763292747, -0.023774034344867204],
            [0.010946522625388108, -0.9998826195688676, 0.01072010851209982, 2.1157397903843567],
            [0.0, 0.0, 0.0, 1.0]]
vc_config['vc_image_shape'] = (1920, 1280) #w, h
# vc_config['vc_image_shape'] = (1024, 1920) #w, h

numC_Trans = 64
res_downsample = 16

model = dict(
    type='VRM_BEVLane',
    img_backbone=dict(
        pretrained=None,
        type='ResNet',
        # type='ResNetV1d',
        depth=34,
        num_stages=4,
        out_indices=(2, 3),  #16倍/32倍下采样
        frozen_stages=-1,
        # norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=False,
        style='pytorch',
        # zero_init_residual=False,
        ),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[256, 512],#[1024, 2048],
        out_channels=64,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='VRM',
    ),
    bev_lane_head=dict(
            type='LaneHeadResidual_Instance_with_offset_z',
            output_size=bev_shape,
            output_2d_shape=output_2d_shape,
            lane_2d_pred=False,
            input_channel=64,
            bce=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0])),
            iou_loss=dict(type='IoULoss'),
            poopoo=dict(type='LanePushPullLoss',
                        var_weight=1.0,
                        dist_weight=1.,
                        margin_var=1.0,
                        margin_dist=5.0,
                        ignore_label=255,
                        ),
            mse_loss=nn.MSELoss(),
            bce_loss=nn.BCELoss(),
        )
    # bev_lane_head=dict(
    #     type='BevLaneHead',
    #     in_channel=64,
    #     num_classes=1,
    #     off_loss=dict(type='RegL1Loss'),#type='OffsetLoss'),
    #     seg_loss=dict(type='Lane_FocalLoss'),
    #     haf_loss=dict(type='RegL1Loss'),
    #     vaf_loss=dict(type='RegL1Loss'),
    #     z_loss=dict(type='RegL1Loss'),
    # )
)

# Data
dataset_type = 'Virtual_Cam_OpenLane_Dataset' #'OpenLane_Dataset'
images_dir = '/home/slj/data/openlane/openlane_all/images'
json_file_dir = '/home/slj/data/openlane/openlane_all/lane3d_300/training/'
# json_file_test_dir = '/home/slj/data/openlane/openlane_all/lane3d_300/validation/'
json_file_test_dir = '/home/slj/data/openlane/openlane_all/lane3d_300/training/'
# json_file_dir = '/home/slj/data/openlane/openlane_all/lane3d_1000/training/'
file_client_args = dict(backend='disk')


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


train_pipeline = [
    dict(
        type='Collect3D', keys=['image_ori', 'image', 'extrinsic', 'intrin', 'post_tran', 'post_rot', 'ipm_gt_segment', 'ipm_gt_instance',
                                'ipm_gt_offset', 'ipm_gt_z', 'image_gt_segment', 'image_gt_instance'
                                ], meta_keys=['res_points_d', 'file_path'])
]

test_pipeline = [
    dict(
        type='Collect3D', keys=['image_ori', 'image', 'extrinsic', 'intrin', 'post_tran', 'post_rot', 'ipm_gt_segment', 'ipm_gt_instance',
                                'ipm_gt_offset', 'ipm_gt_z', 'image_gt_segment', 'image_gt_instance'
                                ], meta_keys=['res_points_d', 'file_path'])
]

eval_pipeline = [
    dict(
        type='Collect3D', keys=['image_ori', 'image', 'extrinsic', 'intrin', 'post_tran', 'post_rot', 'ipm_gt_segment', 'ipm_gt_instance',
                                'ipm_gt_offset', 'ipm_gt_z', 'image_gt_segment', 'image_gt_instance'
                                ], meta_keys=['res_points_d', 'file_path'])
]

test_data_config = dict(
        image_paths=images_dir,
        gt_paths=json_file_test_dir,
        x_range=x_range,
        y_range=y_range,
        meter_per_pixel=meter_per_pixel,
        virtual_camera_config=vc_config,
        input_shape=input_shape,
        output_2d_shape=output_2d_shape,
        test_mode=True,
        CLASSES=None,
        pipeline=test_pipeline,
        use_valid_flag=False,
    )


data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    shuffle=True,
    train=dict(
        image_paths=images_dir,
        gt_paths=json_file_dir,
        x_range=x_range,
        y_range=y_range,
        input_shape=input_shape,
        output_2d_shape=output_2d_shape,
        meter_per_pixel=meter_per_pixel,
        virtual_camera_config=vc_config,
        pipeline=train_pipeline,
        test_mode=False,
        CLASSES=None,
        use_valid_flag=False,
    ),
    val=test_data_config,
    test=test_data_config)



for key in ['train', 'val', 'test']:
    data[key].update(share_data_config)

# Optimizer
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=1e-08)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[24, ])

# optimizer = dict(type="AdamW", lr=1e-3, weight_decay=0)
# optimizer_config = dict(grad_clip=None)
# lr_config = dict(
#     policy="step", warmup="linear", warmup_iters=500, warmup_ratio=0.001, step=[7, 10]
# )

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
