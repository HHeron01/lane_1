    
# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings

import mmcv
import torch
import json
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
from mmdet3d.core.hook.openlane_vis_func import LaneVisFunc
from tqdm import tqdm
import cv2

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
    parser.add_argument('--checkpoint', 
                        default='./result/lift_workdirs/work01/epoch_25_ema.pth')
    parser.add_argument('--show',
                        default=False,
                        help='show or false')
    parser.add_argument(
        '--show_dir', 
        default='./result/lift_workdirs/work01/vis_img/', 
        help='directory where results will be saved')
    parser.add_argument('--out_path', 
                        default='./result/lift_workdirs/work01/', 
                        help='output result file')
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
        samples_per_gpu=8, workers_per_gpu=8, dist=distributed, shuffle=True)

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
    train_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('train_dataloader', {})
    }

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.val)
    data_loader = build_dataloader(dataset, **test_loader_cfg)
    # dataset = build_dataset(cfg.data.train)
    # data_loader = build_dataloader(dataset, **train_loader_cfg)

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
    vis_func = LaneVisFunc()
    for i, data_batch in tqdm(enumerate(data_loader), total=len(data_loader)):

        with torch.no_grad():
            net_out = model(return_loss=False, **data_batch)

        bevlines = vis_func.post_process(net_out)  # 后处理
        batch_size = len(bevlines)

        img_metas = data_batch.get('img_metas')
        img_inputs = data_batch.get('img_inputs')
        map_input = data_batch.get('maps')
        gt_mask, mask_haf, mask_vaf, mask_offset, mask_z = map_input
        imgs, intrins, extrinsics, post_rots, post_trans, undists, bda_rot, rots, trans, grid, drop_idx = img_inputs
        binary_seg, embedding, haf, vaf, offset_feature, z_feature, topdown = net_out

        result = {}

        infer_list = []
        for batch_idx in range(0, batch_size, 8):
            file_path = img_metas.data[0][batch_idx].get('file_path') # 文件路径            
            img_path = '/'.join(file_path.split('/')[1:])
            infer_list.append(img_path)  
            json_name = img_path.split('/')[-1].split('.')[0] + '.json' # json文件名
            img_root = img_path.split('/')[:-1]  
            json_path = os.path.join(args.out_path, 'json_file', *img_root)  # json路径
            os.makedirs(json_path, exist_ok=True)

            result["intrinsic"]=[]
            result["extrinsic"]=[]
            result["file_path"]=file_path
            result["lane_lines"]=[]

            bevline = bevlines[batch_idx]
            for lane in bevline:
                pred_lane = lane['coordinates']
                if len(pred_lane)>0:
                    xyz = []
                    uv = []
                    visibility=[]
                    category=0
                    attribute=0
                    track_id=0
                    for point in pred_lane:
                        x,y,z = point[0], point[1], point[2]
                        xyz.append([float(x), float(y), float(z)])
                    lane = dict(xyz=xyz, uv=uv, visibility=visibility, category=category, attribute=attribute, track_id=track_id)
                    result["lane_lines"].append(lane)
            
            with open(os.path.join(json_path, json_name), 'w') as f:
                f.write(json.dumps(result, indent=2))
            
            if args.show:
                file_path = img_metas.data[0][batch_idx].get('file_path')
                gt_lanes = img_metas.data[0][batch_idx].get('gt_lanes')
                img_name = file_path.split('/')[-1].split('.')[0]
                img_root = file_path.split('/')[:-1]
                vis_img_path = args.show_dir
                # vis_img_path = os.path.join(vis_img_path, *img_root)
                if not os.path.exists(vis_img_path):
                    os.makedirs(vis_img_path)
                
                img = vis_func.get_cam_imgs(imgs[batch_idx])
                img_draw_pred = img.copy()
                img_draw_gt = img.copy()
                mask_gt = gt_mask[batch_idx].detach().cpu().numpy()

                mask_pred = binary_seg[batch_idx].detach().cpu().numpy()
                mask_pred[mask_pred > 0] = 1.0


                bevline = bevlines[batch_idx]
                bevlines_img = vis_func.get_lines_img(mask_pred, bevline)
                points_img = vis_func.get_points_img(mask_gt, gt_lanes)
                heat_img = vis_func.get_pred_bev_feat(topdown[batch_idx])
                img_draw_pred = vis_func.draw_pic_bev_2d(img_draw_pred, bevline, intrins[0], extrinsics[0], post_rots[0], post_trans[0], pred=True)
                img_draw_gt = vis_func.draw_pic_bev_2d(img_draw_gt, gt_lanes, intrins[0], extrinsics[0], post_rots[0], post_trans[0], pred=False)

                filepath_2d = vis_func.get_lane_imu_img_2D(gt_lanes, bevline)
                imu_2d_compare = cv2.imread(filepath_2d)

                filepath_3d = vis_func.get_lane_imu_img_3D(gt_lanes, bevline)
                imu_3d_compare = cv2.imread(filepath_3d)

                disp_img = vis_func.get_disp_img(heat_img, img_draw_pred, img_draw_gt, mask_pred * 100, mask_gt * 100,
                                            bevlines_img, points_img, imu_2d_compare, imu_3d_compare)
                cv2.imwrite(os.path.join(vis_img_path, str(img_name) + '_disp_img_plus.jpg'), disp_img)
        
        with open(os.path.join(args.out_path, 'infer_list.txt'), 'a') as f:
                f.write('\n'.join(infer_list) + '\n')      # 写入文件    

if __name__ == '__main__':
    main()

    