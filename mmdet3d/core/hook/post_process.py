import json

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline


"""
mean_col_by_row_with_offset_z函数通过计算定义了一个二维数组seg中每个矩形框的列均值、行中心坐标和z值的平均值，并将结果存储在列表中。
bev_instance2points_with_offset_z函数使用上面的mean_col_by_row_with_offset_z函数来获取每个矩形框的列均值、行中心坐标和z值。然后，根据给定的参数生成每个矩形框的物理坐标，并使用样条插值进行拟合。最后，将结果存储在一个列表中，并返回。
这些函数主要用于处理和转换二维的BEV（鸟瞰图）数据。
"""
def mean_col_by_row_with_offset_z(seg, offset_y, z):
    assert (len(seg.shape) == 2)

    center_ids = np.unique(seg[seg > 0])
    lines = []
    for idx, cid in enumerate(center_ids):  # 对每个id循环
        cols, rows, z_val = [], [], []  # 初始化列、行、z值的列表
        for y_op in range(seg.shape[0]):  # 循环每一行
            condition = seg[y_op, :] == cid  # 找到该行中与当前id相等的位置
            x_op = np.where(condition)[0]  # 找到所有在该行中与当前id相等的位置
            z_op = z[y_op, :]
            offset_op = offset_y[y_op, :]
            if x_op.size > 0:  # 如果存在位置
                offset_op = offset_op[x_op]
                z_op = np.mean(z_op[x_op])  # 计算位置的平均z值
                z_val.append(z_op)
                x_op_with_offset = x_op + offset_op
                x_op = np.mean(x_op_with_offset)  # 计算位置的平均x值
                cols.append(x_op)
                rows.append(y_op + 0.5)  # 行加上0.5，是为了获得中心位置
        lines.append((cols, rows, z_val))  # 将每一行的结果加入到lines列表中
    return lines


def bev_instance2points_with_offset_z(ids: np.ndarray, max_x=50, meter_per_pixal=(0.2, 0.2), offset_y=None, Z=None):
    center = ids.shape[1] / 2  # 计算中心位置
    lines = mean_col_by_row_with_offset_z(ids, offset_y, Z)  # 调用mean_col_by_row_with_offset_z方法，得到lines列表
    points = []  # 初始化points列表
    for y, x, z in lines:  # 对每一行进行循环
        x = np.array(x)[::-1]  # 将x、y、z列表转换为数组，并进行翻转
        y = np.array(y)[::-1]
        z = np.array(z)[::-1]

        x = max_x / meter_per_pixal[0] - x  # 计算x的物理坐标
        y = y * meter_per_pixal[1]  # 计算y的物理坐标
        y -= center * meter_per_pixal[1]  # 将y坐标以中心点为原点
        x = x * meter_per_pixal[0]  # 计算x的物理坐标

        y *= -1.0  # 将y坐标翻转

        if len(x) < 2:  # 如果列表长度小于2，则跳过此次循环
            continue
        spline = CubicSpline(x, y, extrapolate=False)  # 创建样条曲线拟合x和y坐标
        points.append((x, y, z, spline))  # 将x、y、z和spline加入到points列表中
    return points



