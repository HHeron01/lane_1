import cv2
import numpy as np
import glob
import os


img_dir_name = 'segment-2330686858362435307_603_210_623_210_with_camera_labels'
# root_path = '/home/slj/Documents/workspace/ThomasVision/work_dirs_0426/infer_dir/training'
# img_path = '/home/slj/Documents/workspace/ThomasVision/work_dirs_0426/infer_dir/training/'

# root_path = '/home/slj/Documents/workspace/ThomasVision/work_dirs_0424/infer_dir_0427_qat/training'
# img_path = '/home/slj/Documents/workspace/ThomasVision/work_dirs_0424/infer_dir_0427_qat/training/'

root_path = '/home/slj/data/ThomasVision_info/infer_dir_0113_1/training'
img_path = '/home/slj/data/ThomasVision_info/infer_dir_0113_1/training/'

img_path = img_path + img_dir_name + '/'
# 其它格式的图片也可以
# img_array = []
# for filename in glob.glob(img_path + '*.jpg'):
#     img = cv2.imread(filename)
#     height, width, layers = img.shape
#     size = (width, height)
#     img_array.append(img)
size = (2368, 1280)
# size = (3786, 1152)
# size = (1738, 1152)
# avi：视频类型，mp4也可以
save_path = os.path.join(root_path, img_dir_name + '.avi')
out = cv2.VideoWriter(save_path,
                      cv2.VideoWriter_fourcc(*'DIVX'),
                      15, size)

files = os.listdir(img_path)
files.sort()

for file in files:
    img = cv2.imread(img_path + file)
    print(img.shape)
    out.write(img)
out.release()

