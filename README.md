## Env  Preparation
1 .conda create --name openmmlab python=3.8 -y
2 .conda activate openmmlab
3. pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f   https://download.pytorch.org/whl/torch_stable.html
4. pip install openmim mim 
5. install mmcv-full 
6. mim install mmdet 
7. mim install mmsegmentation
8. git clone https://github.com/open-mmlab/mmdetection3d.git
9. cd mmdetection3d 
pip install -e .
10. python setup.py develop


## Data Preparation
Prepare nuScenes-mini dataset  and create the pkl for BEVDet by running:
*python tools/create_data_bevdet.py*
**ref: **
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
│   │   ├── nuscenes_database
│   │   ├── nuscenes_infos_train.pkl
│   │   ├── nuscenes_infos_trainval.pkl
│   │   ├── nuscenes_infos_val.pkl
│   │   ├── nuscenes_infos_test.pkl
│   │   ├── nuscenes_dbinfos_train.pkl
│   │   ├── nuscenes_infos_train_mono3d.coco.json
│   │   ├── nuscenes_infos_trainval_mono3d.coco.json
│   │   ├── nuscenes_infos_val_mono3d.coco.json
│   │   ├── nuscenes_infos_test_mono3d.coco.json

## Train bev task 
python tools/train.py configs/smart_bev/smart_bev.py 



ref: https://www.pudn.com/news/6340ee732aaf6043c9f982d6.html
