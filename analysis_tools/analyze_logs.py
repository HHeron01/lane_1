# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
from collections import defaultdict

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def cal_train_time(log_dicts, args):
    # 对于每个日志字典进行遍历，获取日志信息
    for i, log_dict in enumerate(log_dicts):
        print(f'{"-" * 5}Analyze train time of {args.json_logs[i]}{"-" * 5}')
        
        # 存储所有的训练时间
        all_times = []
        
        # 对于每个epoch，提取训练时间并将其添加到all_times列表中
        for epoch in log_dict.keys():
            if args.include_outliers:
                all_times.append(log_dict[epoch]['time'])
            else:
                all_times.append(log_dict[epoch]['time'][1:])
        
        # 转换为numpy数组方便计算
        all_times = np.array(all_times)
        
        # 计算每个epoch的平均时间
        epoch_ave_time = all_times.mean(-1)
        
        # 找到最慢和最快的epoch及其对应的平均时间
        slowest_epoch = epoch_ave_time.argmax()
        fastest_epoch = epoch_ave_time.argmin()
        
        # 计算平均时间的标准差
        std_over_epoch = epoch_ave_time.std()
        
        # 输出最慢和最快的epoch以及其平均时间
        print(f'slowest epoch {slowest_epoch + 1}, '
              f'average time is {epoch_ave_time[slowest_epoch]:.4f}')
        print(f'fastest epoch {fastest_epoch + 1}, '
              f'average time is {epoch_ave_time[fastest_epoch]:.4f}')
        
        # 输出平均时间的标准差
        print(f'time std over epochs is {std_over_epoch:.4f}')
        
        # 输出所有迭代的平均时间
        print(f'average iter time: {np.mean(all_times):.4f} s/iter')
        print()


def plot_curve(log_dicts, args):
    # 设置绘图的后端
    if args.backend is not None:
        plt.switch_backend(args.backend)
    
    # 设置绘图的样式
    sns.set_style(args.style)
    
    # 如果未指定图例（legend），则使用{文件名}_{指标名}作为图例
    legend = args.legend
    if legend is None:
        legend = []
        for json_log in args.json_logs:
            for metric in args.keys:
                legend.append(f'{json_log}_{metric}')
    
    # 确保图例的数量与json_logs和keys的数量匹配
    assert len(legend) == (len(args.json_logs) * len(args.keys))
    
    # 获取指标列表
    metrics = args.keys

    num_metrics = len(metrics)
    
    # 遍历日志字典列表
    for i, log_dict in enumerate(log_dicts):
        epochs = list(log_dict.keys())
        
        # 遍历指标列表
        for j, metric in enumerate(metrics):
            print(f'plot curve of {args.json_logs[i]}, metric is {metric}')
            
            # 检查指标是否存在于日志字典中
            if metric not in log_dict[epochs[args.interval - 1]]:
                raise KeyError(
                    f'{args.json_logs[i]} does not contain metric {metric}')
            
            # 根据模式绘制曲线（根据epoch或iter）
            if args.mode == 'eval':
                if min(epochs) == args.interval:
                    x0 = args.interval
                else:
                    if min(epochs) % args.interval == 0:
                        x0 = min(epochs)
                    else:
                        x0 = min(epochs) + args.interval - min(epochs) % args.interval
                
                # 构造x轴数据
                xs = np.arange(x0, max(epochs) + 1, args.interval)
                ys = []
                
                # 构造y轴数据
                for epoch in epochs[args.interval - 1::args.interval]:
                    ys += log_dict[epoch][metric]
                
                if not log_dict[epoch][metric]:
                    xs = xs[:-1]
                
                ax = plt.gca()
                ax.set_xticks(xs)
                plt.xlabel('epoch')
                plt.plot(xs, ys, label=legend[i * num_metrics + j], marker='o')
            else:
                xs = []
                ys = []
                num_iters_per_epoch = log_dict[epochs[args.interval-1]]['iter'][-1]
                for epoch in epochs[args.interval - 1::args.interval]:
                    iters = log_dict[epoch]['iter']
                    if log_dict[epoch]['mode'][-1] == 'val':
                        iters = iters[:-1]
                    xs.append(np.array(iters) + (epoch - 1) * num_iters_per_epoch)
                    ys.append(np.array(log_dict[epoch][metric][:len(iters)]))
                xs = np.concatenate(xs)
                ys = np.concatenate(ys)
                plt.xlabel('iter')
                plt.plot(xs, ys, label=legend[i * num_metrics + j], linewidth=0.5)
            plt.legend()
        
        # 设置图表标题
        if args.title is not None:
            plt.title(args.title)
    
    # 显示图表或保存图表到指定路径
    if args.out is None:
        plt.show()
    else:
        print(f'save curve to: {args.out}')
        plt.savefig(args.out)
        plt.cla()


def add_plot_parser(subparsers):
    parser_plt = subparsers.add_parser(
        'plot_curve', help='parser for plotting curves')
    parser_plt.add_argument(
        'json_logs',
        type=str,
        nargs='+',
        help='path of train log in json format')
    parser_plt.add_argument(
        '--keys',
        type=str,
        nargs='+',
        default=['mAP_0.25'],
        help='the metric that you want to plot')
    parser_plt.add_argument('--title', type=str, help='title of figure')
    parser_plt.add_argument(
        '--legend',
        type=str,
        nargs='+',
        default=None,
        help='legend of each plot')
    parser_plt.add_argument(
        '--backend', type=str, default=None, help='backend of plt')
    parser_plt.add_argument(
        '--style', type=str, default='dark', help='style of plt')
    parser_plt.add_argument('--out', type=str, default=None)
    parser_plt.add_argument('--mode', type=str, default='train')
    parser_plt.add_argument('--interval', type=int, default=1)


def add_time_parser(subparsers):
    parser_time = subparsers.add_parser(
        'cal_train_time',
        help='parser for computing the average time per training iteration')
    parser_time.add_argument(
        'json_logs',
        type=str,
        nargs='+',
        help='path of train log in json format')
    parser_time.add_argument(
        '--include-outliers',
        action='store_true',
        help='include the first value of every epoch when computing '
        'the average time')


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Json Log')
    # currently only support plot curve and calculate average train time
    # 用于向主解析器添加子命令解析器
    subparsers = parser.add_subparsers(dest='task', help='task parser')
    add_plot_parser(subparsers)
    add_time_parser(subparsers)
    args = parser.parse_args()
    return args


def load_json_logs(json_logs):
    """
    将json_logs加载并转换为log_dict，其中epoch作为键，值为子字典。子字典的键是不同的指标，例如memory（内存）、bbox_mAP（边界框平均精确度）等，值是所有迭代中相应指标的值列表。
    """
    # 创建一个空的日志字典列表
    log_dicts = [dict() for _ in json_logs]
    
    # 遍历每个JSON日志文件
    for json_log, log_dict in zip(json_logs, log_dicts):
        # 打开JSON日志文件
        with open(json_log, 'r') as log_file:
            # 逐行读取日志文件
            for line in log_file:
                # 解析JSON行数据
                log = json.loads(line.strip())
                
                # 跳过没有 `epoch` 字段的行
                if 'epoch' not in log:
                    continue
                
                # 获取epoch值并从字典中删除该键值对
                epoch = log.pop('epoch')
                
                # 如果日志字典中没有该epoch的键，则创建一个新的空字典作为值
                if epoch not in log_dict:
                    log_dict[epoch] = defaultdict(list)
                
                # 将日志中的键值对添加到相应的epoch字典中
                for k, v in log.items():
                    log_dict[epoch][k].append(v)
    
    # 返回日志字典列表
    return log_dicts



def main():
    args = parse_args()

    json_logs = args.json_logs
    for json_log in json_logs:
        assert json_log.endswith('.json')

    log_dicts = load_json_logs(json_logs)

    eval(args.task)(log_dicts, args)


if __name__ == '__main__':
    main()
