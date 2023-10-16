import cv2
import numpy as np
import glob
import os


img_dir_name = '/home/slj/data/ros_bag/sensor/camera/front/h264'
save_path = '/home/slj/data/ros_bag/sensor/camera/front/h264/front.txt'

image_list = os.listdir(img_dir_name)

image_list.sort()
with open(save_path, 'w') as f:
    for image_name in image_list:
        f.write(img_dir_name + '/' + image_name + '\n')