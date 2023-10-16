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
        description='MMDet测试（和评估）模型')
    parser.add_argument('config', help='测试配置文件的路径')
    parser.add_argument('checkpoint', help='检查点文件')
    parser.add_argument('--out', help='以pickle格式输出结果文件')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='是否融合卷积和批归一化操作，这会稍微增加推断速度')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='（已弃用，请使用--gpu-id）要使用的GPU的ID（仅适用于非分布式训练）')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='要使用的GPU的ID（仅适用于非分布式测试）')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='格式化输出结果，而不进行评估。当您想将结果格式化为特定格式并提交给测试服务器时，此选项非常有用')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='评估指标，取决于数据集，例如COCO的"bbox"，"segm"，"proposal"，PASCAL VOC的"mAP"，"recall"')
    parser.add_argument('--show', action='store_true', help='显示结果')
    parser.add_argument(
        '--show-dir', help='保存结果的目录')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='是否使用GPU收集结果')
    parser.add_argument(
        '--no-aavt',
        action='store_true',
        help='视图变换器后不进行对齐操作')
    parser.add_argument(
        '--tmpdir',
        help='用于从多个工作进程收集结果的临时目录，在未指定gpu-collect时可用')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='是否为CUDNN后端设置确定性选项')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='对使用的配置进行一些设置覆盖，以xxx=yyy格式的键值对将合并到配置文件中。如果要覆盖的值是一个列表，格式应为key="[a,b]"或key=a,b。它还允许嵌套的列表/元组值，例如 key="[(a,b),(c,d)]"。请注意，引号是必需的，不允许有空格。')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='自定义评估选项，以xxx=yyy格式的键值对将作为dataset.evaluate()函数的关键字参数（已弃用），请改用--eval-options')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='自定义评估选项，以xxx=yyy格式的键值对将作为dataset.evaluate()函数的关键字参数')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='作业启动器')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options和--eval-options不能同时指定，--options已弃用，请使用--eval-options')
    if args.options:
        warnings.warn('--options已弃用，请使用--eval-options')
        args.eval_options = args.options
    return args

def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('请至少指定一个操作（保存/评估/格式化/显示结果/保存结果）使用参数 "--out"、"--eval"、'
         '"--format-only"、"--show"或"--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval和--format_only不能同时指定')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('输出文件必须是pkl格式的文件')

    # 解析命令行参数并进行必要的校验
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # 从配置文件中加载配置，并根据命令行参数进行必要的修改
    cfg = compat_cfg(cfg)

    # 兼容处理配置文件的更新
    setup_multi_processes(cfg)

    # 设置多进程参数，用于多进程测试
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None

    # 设置cudnn_benchmark，优化模型的加速运算
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids`已弃用，请使用`--gpu-id`。因为在非分布式测试中，'
                      '我们仅支持单GPU模式。现在使用gpu_ids中的第一个GPU。')
    else:
        cfg.gpu_ids = [args.gpu_id]

    # 设置要使用的GPU的ID
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # 初始化分布式环境，用于多GPU的并行测试
    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

    # 设置测试数据加载器的参数
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)

    # 设置测试数据集的模式，并根据配置文件修改数据预处理器
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)
    # 对于连接的测试数据集，设置测试模式，并根据配置文件修改数据预处理器

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }
    # 设置测试数据加载器的配置，包括默认参数和配置文件中的参数
    
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)
    # 设置随机种子，用于结果的可复现性
    
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)
    # 构建数据集和数据加载器
    
    if not args.no_aavt:
        if '4D' in cfg.model.type:
            cfg.model.align_after_view_transfromation = True
    # 如果命令行参数中没有指定不进行轴对齐和视角转换操作，并且模型类型中包含'4D'字符串，
    # 则设置模型的align_after_view_transformation为True
    
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    # 构建模型，加载配置中的模型参数
    
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    # 构建fp16模型，用于模型的半精度运算

    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # 加载模型的checkpoint文件

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # 如果指定了融合卷积和批归一化操作，则对模型进行融合处理

    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # 从checkpoint中获取类别信息，用于模型预测结果的解析和显示

    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        model.PALETTE = dataset.PALETTE
    # 从checkpoint中获取调色板信息，用于分割任务的可视化

    if not distributed:
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
    # 如果不是分布式环境，在单个GPU上进行模型的测试

    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)
    # 在多个GPU上进行模型的测试

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)

    # 在主进程中保存模型测试的结果
    kwargs = {} if args.eval_options is None else args.eval_options
    if args.format_only:
        dataset.format_results(outputs, **kwargs)

    # 格式化输出模型测试结果
    if args.eval:
        eval_kwargs = cfg.get('evaluation', {}).copy()
        for key in [
                'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                'rule'
        ]:
            eval_kwargs.pop(key, None)
        eval_kwargs.update(dict(metric=args.eval, **kwargs))
        print(dataset.evaluate(outputs, **eval_kwargs))
    # 进行模型的评估，包括计算指标和打印结果

if __name__ == '__main__':
    main()
