# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

import mmdet
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.apis import inference
from mmdet3d.core.hook.vrm_openlane_vis_func import Vrm_LaneVisFunc
from tqdm import tqdm
import cv2
import numpy as np
import json

if mmdet.__version__ > '2.23.0':
    # If mmdet version > 2.23.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
else:
    from mmdet3d.utils import setup_multi_processes

try:
    # If mmdet version > 2.23.0, compat_cfg would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    # parser.add_argument('--checkpoint', default='/home/slj/Documents/workspace/mmdet3d/work_dirs_0105/epoch_17_ema.pth', help='checkpoint file')
    parser.add_argument('--checkpoint', default='/home/slj/data/ThomasVision_info/work_dirs_0426/epoch_80_ema.pth')
    parser.add_argument('--show',
                        default=True,
                        help='show or false')
    parser.add_argument('--json_path',
                        default='./work_dirs_0605/vaild_json',
                        help='json for eval')
    parser.add_argument(
        '--show_dir', default='work_dirs_0605/infer_dir/', help='directory where results will be saved')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--no-aavt',
        action='store_true',
        help='Do not align after view transformer.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg = compat_cfg(cfg)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    # model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    # print("cfg:", cfg.model)
    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

    # if not distributed:
    model = MMDataParallel(
        model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
    model.eval()
    vis_func = Vrm_LaneVisFunc()
    for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        with torch.no_grad():
            bev_net_out, net_out_2d = model(return_loss=False, **data)
        if args.show:
            img_metas = data.get('img_metas')

            file_path = img_metas.data[0][0].get('file_path')

            trans_matrix = img_metas.data[0][0].get('trans_matrix')

            gt_lanes = img_metas.data[0][0].get('res_points_d')

            axis_gt_lanes = [] #1
            vaild_gt_lanes = []
            for gt_lane in gt_lanes:
                # pred_in_persformer = np.array([-1 * lane[1], lane[0], lane[2]])
                axis_gt_lane = np.array([-1 * gt_lane[:, 0], gt_lane[:, 1], gt_lane[:, 2]])
                axis_gt_lanes.append(axis_gt_lane.T)
                vaild_gt_lanes.append(axis_gt_lane.T.tolist())

            img_name = file_path.split('/')[-1].split('.')[0]
            img_root = file_path.split('/')[:-1]
            # print("img_root:", img_root)

            visu_path = args.show_dir
            img_path = os.path.join(visu_path, *img_root)
            # print("img_path:", img_path)
            if not os.path.exists(img_path):
                os.makedirs(img_path)

            image_oris, imgs, intrins, extrinsics, post_rots, post_trans = data.get(
                "image_ori"), data.get("image"), data.get("intrin"), data.get(
                "extrinsic"), data.get("post_rot"), data.get("post_tran"),

            # map_input = runner.data_batch.get("maps")
            gt_mask, gt_instance, mask_offset, mask_z = data.get("ipm_gt_segment"), data.get(
                "ipm_gt_instance"), data.get("ipm_gt_offset"), data.get("ipm_gt_z"),

            # net_out = runner.outputs.get('net_out')
            binary_seg, embedding, offset_feature, z_feature, topdown = bev_net_out

            ori_img = cv2.imread('/home/slj/data/openlane/openlane_all/images/' + file_path)
            ori_img = cv2.warpPerspective(ori_img, trans_matrix, (1920, 1280)) #self.vc_image_shape)

            img = vis_func.get_cam_imgs(image_oris)
            ipm_img = vis_func.get_cam_imgs(imgs)
            # img = cv2.resize(img, (1024, 576))
            img_draw_pred = img.copy()
            img_draw_gt = img.copy()
            # ground truth
            mask_gt = gt_mask[0].detach().cpu().numpy()

            # pred
            bevlines = vis_func.post_process(bev_net_out)

            ipm_img_show, iego_lanes = vis_func.img_post_process(net_out_2d, ori_img, post_rots[0],
                                             post_trans[0], extrinsics[0], intrins[0])

            # ipm_lanes = vis_func.imageview2ego(img_lanes)
            axis_bevlines = [] #2
            vaild_bevlines = []
            for bevline in bevlines:
                # pred_in_persformer = np.array([-1 * lane[1], lane[0], lane[2]])
                axis_bevline = np.array([-1 * bevline[:, 0], bevline[:, 1], bevline[:, 2]])
                axis_bevlines.append(axis_bevline.T)
                vaild_bevlines.append(bevline.tolist())

            # if not os.path.exists(args.json_path):
            #     os.makedirs(args.json_path)
            #
            # with open(os.path.join(args.json_path, img_name + '.json'), 'w') as f1:
            #     json.dump([vaild_bevlines, vaild_gt_lanes], f1)

            mask_pred = binary_seg[0].detach().cpu().numpy()
            mask_pred[mask_pred > 0] = 1.0

            heat_img = vis_func.get_pred_bev_feat(topdown)
            img_draw_pred = vis_func.draw_pic_2d(img_draw_pred, axis_bevlines, intrins[0], extrinsics[0], post_rots[0],
                                             post_trans[0], pred=True)
            img_draw_gt = vis_func.draw_pic_2d(img_draw_gt, gt_lanes, intrins[0], extrinsics[0], post_rots[0],
                                           post_trans[0], pred=False)

            # filepath_2d = vis_func.get_lane_imu_img_2D(axis_gt_lanes, bevlines)
            filepath_2d = vis_func.get_lane_imu_img_2D(axis_gt_lanes, bevlines, iego_lanes)
            imu_2d_compare = cv2.imread(filepath_2d)

            filepath_3d = vis_func.get_lane_imu_img_3D(axis_gt_lanes, bevlines, iego_lanes)
            imu_3d_compare = cv2.imread(filepath_3d)

            disp_img = vis_func.get_disp_img(img_draw_pred, img_draw_gt, mask_pred[0] * 100, mask_gt[0] * 100,
                                         imu_2d_compare, imu_3d_compare, ipm_img_show)

            cv2.imwrite(img_path + '/' + str(img_name) + '_disp_img_plus.jpg', disp_img)

            # cv2.imwrite(img_path + '/' + str(img_name) + '_disp_ipm_img.jpg', ipm_img_show)
            # cv2.imshow("frame", disp_img)
            # cv2.waitKey(10)




if __name__ == '__main__':
    main()
