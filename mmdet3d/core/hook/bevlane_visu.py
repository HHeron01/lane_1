import torch
from mmcv.runner.hooks import HOOKS, Hook
from PIL import Image
import cv2
import os
import numpy as np
from mmdet3d.core.hook.openlane_vis_func import LaneVisFunc
from mmdet3d.core.hook.anchor_openlane_vis_func import Anchor_LaneVisFunc
from mmdet3d.core.hook.vrm_openlane_vis_func import Vrm_LaneVisFunc

# 使用HOOKS.register_module()装饰器注册一个名为BevLaneVisAllHook的Hook类
@HOOKS.register_module()
class BevLaneVisAllHook(Hook):
    """
    用于可视化lane3d输出的Hook
    """
    def __init__(self) -> None:
        super().__init__()
        # 初始化函数，设置Hook的属性
        # 如果需要写入结果，则需要提供log_dir，这里注释掉了
        # assert log_dir is not None
        # self.log_dir = log_dir
        # 设置可视化的频率
        # self.freq = freq
        # self.dst_rank = dst_rank
        # self.use_offset = use_offset
        
        # 创建用于可视化的LaneVisFunc对象
        self.vis_func= LaneVisFunc()
        # 创建用于可视化的Anchor_LaneVisFunc对象
        self.anchor_vis_func = Anchor_LaneVisFunc()
        # 创建用于可视化的Vrm_LaneVisFunc对象
        self.vir_vis_func0 = Vrm_LaneVisFunc()                 
        
    # 在训练过程的每个epoch结束后调用
    def after_train_epoch(self, runner):
        # 如果runner的data_batch包含"maps"键
        if runner.data_batch.get("maps"):
            # 调用可视化函数self.vis_func进行可视化
            self.vis_func(runner)
        # 如果runner的data_batch包含"targets"键
        if runner.data_batch.get("targets"):
            # 调用可视化函数self.anchor_vis_func进行可视化
            self.anchor_vis_func(runner)
        # 如果runner的data_batch包含"maps_2d"键，并且不为None
        if runner.data_batch.get("maps_2d", None) is not None:
            # 创建一个新的LaneVisFunc对象，设置use_off_z和use_offset为False
            self.vis_func1 = LaneVisFunc(use_off_z=False, use_offset=False)
            # 调用vis_func_2d方法进行2D可视化
            self.vis_func1.vis_func_2d(runner)
        # 如果runner的data_batch包含"lables_2d"键，并且不为None
        if runner.data_batch.get("lables_2d", None) is not None:
            # 创建一个新的Anchor_LaneVisFunc对象，设置use_off_z和use_offset为False
            self.anchor_vis_func1 = Anchor_LaneVisFunc(use_off_z=False, use_offset=False)
            # 调用vis_2d_result方法进行2D结果可视化
            self.anchor_vis_func1.vis_2d_result(runner)
        # 如果runner的model_name为'BEVLane'或'VRM_BEVLane'
        if runner.model_name == 'BEVLane':
            # 调用vir_vis_func方法进行虚拟可视化
            self.vir_vis_func(runner)
        elif runner.model_name == 'VRM_BEVLane':
            # 调用vir_vis_func方法进行虚拟可视化
            self.vir_vis_func(runner)
    
    # 在验证过程的每个epoch结束后调用
    def after_val_epoch(self, runner):
        # 如果runner的data_batch包含"maps"键
        if runner.data_batch.get("maps"):
            # 调用可视化函数self.vis_func进行可视化
            self.vis_func(runner)
        # 如果runner的data_batch包含"targets"键
        if runner.data_batch.get("targets"):
            # 调用可视化函数self.anchor_vis_func进行可视化
            self.anchor_vis_func(runner)
        # 如果runner的data_batch包含"maps_2d"键，并且不为None
        if runner.data_batch.get("maps_2d", None) is not None:
            # 创建一个新的LaneVisFunc对象，设置use_off_z和use_offset为False
            self.vis_func1 = LaneVisFunc(use_off_z=False, use_offset=False)
            # 调用vis_func_2d方法进行2D可视化
            self.vis_func1.vis_func_2d(runner)
        # 如果runner的data_batch包含"lables_2d"键，并且不为None
        if runner.data_batch.get("lables_2d", None) is not None:
            # 创建一个新的Anchor_LaneVisFunc对象，设置use_off_z和use_offset为False
            self.anchor_vis_func1 = Anchor_LaneVisFunc(use_off_z=False, use_offset=False)
            # 调用visu_2d_result方法进行2D结果可视化
            self.anchor_vis_func1.visu_2d_result(runner)
        # 如果runner的model_name为'VRM_BEVLane'
        elif runner.model_name == 'VRM_BEVLane':
            # 调用vir_vis_func方法进行虚拟可视化
            self.vir_vis_func(runner)


    def vis(self, runner, mode='train'):
        pass
        # print('*' * 50)
        # *_, inputs = storager.get_data('inputs')
        # *_, results = storager.get_data('results')
        # print("results:", len(results))

        # NOTE results["prediction"], results["loss"]
        # NOTE predictions[1] is lanehead pred
        # print("results:", results["prediction"][1][1].shape)
        # disp_img = self.vis_func(inputs, results["prediction"][1])
        #
        # if self.write:
        #     print("write: epoc={}, train_iter={}".format(self.trainer.epoch, self.trainer.train_iter))
        #     img = Image.fromarray(disp_img)
        #     img.save(self.log_dir + '/' + str(self.trainer.epoch % 200).zfill(6) + '.jpg')