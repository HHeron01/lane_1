# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import time

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector
from tools.misc.fuse_conv_bn import fuse_module


"""
这个脚本用于基准测试（benchmark）模型的性能。它通过解析命令行参数来指定配置文件路径、checkpoint 文件等信息。然后，它根据配置文件构建数据加载器和模型，并加载 checkpoint。接下来，它对指定数量的样本进行性能评估，并计算推理速度（fps）。最后，它输出评估结果。
"""
def parse_args():
    # 创建一个参数解析器对象
    parser = argparse.ArgumentParser(description='MMDet benchmark a model')
    
    # 添加命令行参数，用于指定测试配置文件的路径
    parser.add_argument('config', help='test config file path')
    
    # 添加命令行参数，用于指定checkpoint文件
    parser.add_argument('checkpoint', help='checkpoint file')
    
    # 添加可选的命令行参数，默认值为400，用于指定要进行性能评估的样本数量
    parser.add_argument('--samples', default=400, help='samples to benchmark')
    
    # 添加可选的命令行参数，默认值为50，用于指定打印日志的间隔
    parser.add_argument(
        '--log-interval', default=50, help='interval of logging')
    
    # 添加可选的命令行参数，如果指定该参数，则将卷积和批标准化进行融合，这会稍微提高推理速度
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    
    # 添加可选的命令行参数，如果指定该参数，则不使用预计算加速
    parser.add_argument(
        '--no-acceleration',
        action='store_true',
        help='Omit the pre-computation acceleration')
    
    # 解析命令行参数
    args = parser.parse_args()
    return args


def main():
    # 解析命令行参数
    args = parse_args()

    # 从配置文件中加载配置
    cfg = Config.fromfile(args.config)
    
    # 设置 cudnn_benchmark，以提高训练速度
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    
    # 设置模型的预训练权重为None，关闭模型的预训练
    cfg.model.pretrained = None
    
    # 将数据加载器设置为测试模式
    cfg.data.test.test_mode = True

    # 构建数据集的数据加载器
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # 构建模型并加载checkpoint
    cfg.model.train_cfg = None
    # 将模型的训练配置设置为 None，关闭训练配置，模型将作为测试模型运行
    cfg.model.align_after_view_transfromation=True
    # 设置模型在视角转换后的对齐操作为 True
    if not args.no_acceleration:
        cfg.model.img_view_transformer.accelerate=True
    # 如果命令行参数 "--no-acceleration" 没有指定，将模型图像视角转换的加速设置为 True
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    # 根据配置文件中的模型配置和测试配置，构建检测器模型对象
    fp16_cfg = cfg.get('fp16', None)
    # 获取配置文件中的 fp16 相关配置
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    # 如果 fp16 相关配置存在，则对模型进行 fp16 化处理
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    # 加载预训练的权重到模型中，根据命令行参数指定的权重文件路径加载，加载后将权重映射到 CPU 上进行计算
    # 如果指定了 fuse-conv-bn 参数，则进行卷积和批标准化的融合
    if args.fuse_conv_bn:
        model = fuse_module(model)

    model = MMDataParallel(model, device_ids=[0])

    model.eval()

    # 前几次迭代可能非常慢，因此跳过它们
    num_warmup = 5
    pure_inf_time = 0

    # 对几个样本进行性能评估，并取其平均值
    for i, data in enumerate(data_loader):
        # 遍历数据加载器中的每个数据
        inputs = [d.cuda() for d in data['img_inputs'][0]]
        # 将输入数据转移到 GPU 上进行计算
        
        with torch.no_grad():
            feat_prev, inputs = model.module.extract_img_feat(inputs, pred_prev=True, img_metas=None)
        # 使用模型的 extract_img_feat 方法提取图像特征，其中 pred_prev 和 img_metas 是方法的输入参数。
        # 使用 torch.no_grad() 来关闭梯度计算，节省内存和加快推理速度

        data['img_inputs'][0] = inputs
        # 将处理后的输入数据更新到字典中
        torch.cuda.synchronize()
        # 确保前面的 GPU 计算完成
        start_time = time.perf_counter()
        # 记录开始时间

        with torch.no_grad():
            model(return_loss=False, rescale=True, sequential=True, feat_prev=feat_prev, **data)
        # 使用模型进行推理，其中的参数 return_loss、rescale、sequential 和 feat_prev 是方法的输入参数，**data 是字典参数的展开形式

        torch.cuda.synchronize()
        # 确保 GPU 推理完成
        elapsed = time.perf_counter() - start_time
        # 计算推理时间

        if i >= num_warmup:
            pure_inf_time += elapsed
            # 在热身迭代（前几次迭代）之后，累加推理时间

        if (i + 1) % args.log_interval == 0:
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(f'Done image [{i + 1:<3}/ {args.samples}], fps: {fps:.1f} img / s')
            # 满足日志间隔条件时，计算当前的推理速度（帧率）并打印日志信息

        if (i + 1) == args.samples:
            pure_inf_time += elapsed
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(f'Overall fps: {fps:.1f} img / s')
            break
            # 当达到指定的样本数量时，累加推理时间并计算总体的推理速度，并输出总体的推理速度


if __name__ == '__main__':
    main()
