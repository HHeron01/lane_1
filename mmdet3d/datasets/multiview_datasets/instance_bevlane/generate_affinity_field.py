import os
import cv2
import copy
import torch
import numpy as np
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple, Union

# import torchsnooper

__all__ = ['GenerateHAFAndVAF']


class GenerateHAFAndVAF(object):
    """
    NOTE: Generate horizontal affinity field and vertical affinity field.
    """

    def __init__(self, H_interv: int = 1) -> None:
        """
        Args:
            H_interv : horizontal sampling interval, default 1 pixel
        """
        self.H_interv = H_interv

    def load_bin_seg(self, file_path, debug_visu=False):
        print('file_path=', file_path)
        self.img = cv2.imread(file_path)
        if debug_visu:
            print('img shape', self.img.shape)
            cv2.imshow("img", self.img)
            cv2.waitKey(3000)

    def get_bin_seg(self):
        self.bin_seg = np.array(self.img[:, :, 0], dtype=np.uint8)
        return self.bin_seg

    def _line_center_coords(self, oneline_xmask: np.ndarray) -> np.ndarray:
        x_sum = np.sum(oneline_xmask, axis=1).astype(float) # 240

        nonzero_cnt = np.count_nonzero(oneline_xmask, axis=1).astype(int)
        temp = np.ones((nonzero_cnt.shape[0], 2), dtype=int)   #240, 2
        temp[:, 0] = nonzero_cnt
        nonzero_cnt = np.max(temp, axis=1)  # (240, )

        center_x = x_sum / nonzero_cnt

        return center_x

    def __call__(self, lane_mask: Union[np.ndarray, torch.Tensor]):
        laneline_mask = copy.deepcopy(lane_mask)
        if isinstance(laneline_mask, torch.Tensor):
            laneline_mask = np.array(laneline_mask)
        if len(laneline_mask.shape) == 3 and laneline_mask.shape[2] == 1:
            laneline_mask = np.squeeze(laneline_mask, axis=2)

        haf = np.zeros_like(laneline_mask, dtype=float)
        vaf = np.zeros((*laneline_mask.shape, 2), dtype=float)

        mask_h, mask_w = laneline_mask.shape[0], laneline_mask.shape[1]
        x = np.arange(0, mask_w, 1)
        y = np.arange(0, mask_h, 1)
        x_mask, y_mask = np.meshgrid(x, y)  # (240,96)
        for idx in np.unique(laneline_mask):
            if idx == 0:
                continue
            oneline_mask = np.zeros_like(laneline_mask, dtype=int)   # laneline_mask 就是车道线序号的mask
            oneline_mask[laneline_mask == idx] = 1     # 这里就是车道线是否匹配上
            oneline_xmask = oneline_mask * x_mask       

            center_x = self._line_center_coords(oneline_xmask)
            center_x = np.expand_dims(center_x, 1).repeat(mask_w, axis=1)  # 计算所有列上的中点

            # calc haf
            valid_cx = oneline_mask * center_x
            haf_oneline = valid_cx - oneline_xmask
            haf_oneline[haf_oneline > 0] = 1.0
            haf_oneline[haf_oneline < 0] = -1.0
            #最靠近左边mask的为1, 增加标签
            rows, cols = haf_oneline.shape  # 获取 haf_oneline 数组的行数和列数
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
            haf += haf_oneline
            # calc vaf
            vaf_oneline = np.zeros((*laneline_mask.shape, 2), dtype=float)
            center_x_down = np.zeros_like(laneline_mask, dtype=float)
            center_x_down[self.H_interv:, :] = center_x[0:mask_h - self.H_interv, :]
            valid_cx = oneline_mask * center_x_down
            vertical_mask = np.bitwise_and(valid_cx > 0, oneline_mask > 0)
            vaf_oneline[:, :, 0][vertical_mask] = (valid_cx - oneline_xmask)[vertical_mask]
            vaf_oneline[:, :, 1][vertical_mask] = -self.H_interv
            # normalize
            vaf_oneline = F.normalize(torch.from_numpy(vaf_oneline), dim=2)
            vaf_oneline = np.array(vaf_oneline.float())

            vaf += vaf_oneline
        # print('haf.shape',haf.shape) (320, 192)
        # print('vaf.shape', vaf.shape) (320, 192)
        # rows, cols = haf.shape
        # haf_masked = np.zeros((rows, cols))
        # for i in range(rows):
        #     for j in range(cols):
        #         if haf[i, j] != 0:
        #             haf_masked[i, j] = 1
        #         elif j > 0 and j < cols-1 and haf[i, j-1] > 0 and haf[i, j+1] < 0:
        #             haf_masked[i, j] = 1

        return haf, vaf  # haf_masked

    def debug_vis(self, lane_mask: Union[np.ndarray, torch.Tensor], out_path: str) -> None:
        haf, vaf = self.__call__(lane_mask)
        vaf = vaf[:, :, 0]
        haf_vis = np.zeros((*haf.shape, 3), dtype=np.uint8)
        vaf_vis = np.zeros((*vaf.shape, 3), dtype=np.uint8)

        # NOTE: priorily draw outer line than inter line
        # NOTE: positive horizontal offset use warm color, negative horizontal offset use cool color
        for row in range(haf.shape[0]):
            for col in range(haf.shape[1]):
                if haf[row, col] > 0:
                    # cv2.line(haf_vis, (col, row), (col + int(haf[row, col]), row), (0, 0, 255), thickness=1)
                    cv2.circle(haf_vis, (col, row), 1, (0, 0, 255))
                elif haf[row, col] < 0:
                    # cv2.line(haf_vis, (col, row), (col + int(haf[row, col]), row), (0, 255, 0), thickness=1)
                    cv2.circle(haf_vis, (col, row), 1, (0, 255, 0))
                else:
                    pass

        img_name = os.path.join(out_path, 'haf_vis.png')
        cv2.imwrite(filename=img_name, img=haf_vis)

        for row in range(vaf.shape[0]):
            for col in range(vaf.shape[1]):
                if vaf[row, col] > 0:
                    cv2.line(vaf_vis, (col, row), (col + int(vaf[row, col]), row - self.H_interv), (0, 0, 255),
                             thickness=1)
                    # cv2.circle(haf_vis, (col, row - 1), 1, (0, 0, 255))
                elif vaf[row, col] < 0:
                    cv2.line(vaf_vis, (col, row), (col + int(vaf[row, col]), row - self.H_interv), (0, 255, 0),
                             thickness=1)
                    # cv2.circle(haf_vis, (col, row - 1), 1, (0, 255, 0))
                else:
                    pass
        img_name = os.path.join(out_path, 'vaf_vis.png')
        cv2.imwrite(filename=img_name, img=vaf_vis)
