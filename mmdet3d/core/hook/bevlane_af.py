import os
import cv2
import copy
import torch
import numpy as np
import torch.nn.functional as F
from typing import Union
import torchsnooper


class GenerateAF(object):
    def __init__(self, H_interv: int = 1) -> None:
        """
        初始化方法，设置水平采样间隔。
        Args:
            H_interv: 水平采样间隔，默认为1个像素
        """
        self.H_interv = H_interv

    def load_bin_seg(self, file_path):
        """
        加载二进制分割结果。
        
        Args:
            file_path: 图像文件路径
        """
        print('file_path=', file_path)
        self.img = cv2.imread(file_path)

    def get_bin_seg(self):
        """
        获取二进制分割结果。

        Returns:
            np.ndarray: 二进制分割图像
        """
        self.bin_seg = np.array(self.img[:, :, 0], dtype=np.uint8)
        return self.bin_seg

    def _line_center_coords(self, oneline_xmask: np.ndarray) -> np.ndarray:
        """
        计算一行中心点坐标。

        Args:
            oneline_xmask: 一行的x坐标掩码
        
        Returns:
            np.ndarray: 一行中心点的x坐标数组
        """
        x_sum = np.sum(oneline_xmask, axis=1).astype(np.float)  # 每一行x坐标求和
        nonzero_cnt = np.count_nonzero(oneline_xmask, axis=1).astype(np.int)  # 每一行非0元素的个数
        temp = np.ones((nonzero_cnt.shape[0], 2), dtype=int)
        temp[:, 0] = nonzero_cnt
        nonzero_cnt = np.max(temp, axis=1)  # 对每一行非0元素个数取最大值，用于除法计算
        center_x = x_sum / nonzero_cnt  # 计算每一行的中心点x坐标
        return center_x

    def __call__(self, lane_mask: Union[np.ndarray, torch.Tensor]):
        """
        生成水平机动辅助力（HAF）和垂直波动辅助力（VAF）。

        Args:
            lane_mask: 车道线掩码
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: 水平机动辅助力和垂直波动辅助力
        """
        laneline_mask = copy.deepcopy(lane_mask)
        if isinstance(laneline_mask, torch.Tensor):  # 将torch.Tensor转换为numpy数组
            laneline_mask = np.array(laneline_mask)
        if len(laneline_mask.shape) == 3 and laneline_mask.shape[2] == 1:  # 去除多余的维度
            laneline_mask = np.squeeze(laneline_mask, axis=2)

        haf = np.zeros_like(laneline_mask, dtype=np.float)
        vaf = np.zeros((*laneline_mask.shape, 2), dtype=np.float)

        mask_h, mask_w = laneline_mask.shape[0], laneline_mask.shape[1]
        x = np.arange(0, mask_w, 1)
        y = np.arange(0, mask_h, 1)
        x_mask, y_mask = np.meshgrid(x, y)  # 生成x和y的网格矩阵

        for idx in np.unique(laneline_mask):
            if idx == 0:
                continue
            oneline_mask = np.zeros_like(laneline_mask, dtype=np.int)
            oneline_mask[laneline_mask == idx] = 1  # 生成当前车道线的单行掩码
            oneline_xmask = oneline_mask * x_mask  # 获取单行的x坐标掩码

            center_x = self._line_center_coords(oneline_xmask)  # 计算单行中心点的x坐标
            center_x = np.expand_dims(center_x, 1).repeat(mask_w, axis=1)  # 在整行上进行广播，生成完整的中心点x坐标数组

            # 计算水平机动辅助力（HAF）
            valid_cx = oneline_mask * center_x  # 有效中心点的x坐标
            haf_oneline = valid_cx - oneline_xmask  # 水平机动辅助力（HAF）= 有效中心点x坐标 - 当前x坐标
            haf_oneline[haf_oneline > 0] = 1.0  # 大于0的值置为1
            haf_oneline[haf_oneline < 0] = -1.0  # 小于0的值置为-1

            rows, cols = haf_oneline.shape
            for i in range(rows):
                for j in range(cols):
                    if haf_oneline[i, j] > 0:
                        if j < 95:
                            haf_oneline[i, j+1] = 0
                        if j < 94:
                            haf_oneline[i, j+2] = -1
                    elif haf_oneline[i, j] < 0:
                        if j > 0:
                            haf_oneline[i, j-1] = 0
                        if j > 1:
                            haf_oneline[i, j-2] = 1

            # 计算垂直波动辅助力（VAF）
            vaf_oneline = np.zeros((*laneline_mask.shape, 2), dtype=np.float)
            center_x_down = np.zeros_like(laneline_mask, dtype=np.float)
            center_x_down[self.H_interv:, :] = center_x[0:mask_h - self.H_interv, :]
            valid_cx = oneline_mask * center_x_down


    @torchsnooper.snoop()
    def debug_vis(self, lane_mask: Union[np.ndarray, torch.Tensor], out_path: str) -> None:
        # 获取HAF和VAF
        haf, vaf = self.__call__(lane_mask)
        vaf = vaf[:, :, 0]  # 去除VAF的第三个维度

        # 创建空数组来存储可视化结果
        haf_vis = np.zeros((*haf.shape, 3), dtype=np.uint8)
        vaf_vis = np.zeros((*vaf.shape, 3), dtype=np.uint8)

        # 对HAF的每个像素点进行处理
        for row in range(haf.shape[0]):
            for col in range(haf.shape[1]):
                if haf[row, col] > 0:
                    # 当HAF大于0时，将当前像素点标记为红色
                    cv2.circle(haf_vis, (col, row), 1, (0, 0, 255))
                elif haf[row, col] < 0:
                    # 当HAF小于0时，将当前像素点标记为绿色
                    cv2.circle(haf_vis, (col, row), 1, (0, 255, 0))
                else:
                    pass

        # 将HAF的可视化结果保存为文件
        img_name = os.path.join(out_path, 'haf_vis.png')
        cv2.imwrite(filename=img_name, img=haf_vis)

        # 对VAF的每个像素点进行处理
        for row in range(vaf.shape[0]):
            for col in range(vaf.shape[1]):
                if vaf[row, col] > 0:
                    # 当VAF大于0时，绘制一条红色线段
                    cv2.line(vaf_vis, (col, row), (col + int(vaf[row, col]), row - self.H_interv), (0, 0, 255), thickness=1)
                elif vaf[row, col] < 0:
                    # 当VAF小于0时，绘制一条绿色线段
                    cv2.line(vaf_vis, (col, row), (col + int(vaf[row, col]), row - self.H_interv), (0, 255, 0), thickness=1)
                else:
                    pass

        # 将VAF的可视化结果保存为文件
        img_name = os.path.join(out_path, 'vaf_vis.png')
        cv2.imwrite(filename=img_name, img=vaf_vis)

        # 返回HAF和VAF的可视化结果
        return haf_vis, vaf_vis
