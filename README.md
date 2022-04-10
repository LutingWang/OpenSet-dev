# Setup

```shell
cd
git clone git@gitlab.alibaba-inc.com:wangluting.wlt/openset_detection.git
git clone git@github.com:open-mmlab/mmdetection.git
git clone git@github.com:LutingWang/todd.git

cd openset_detection
ln -s ${HOME}/mmdetection/mmdet .
ln -s ${HOME}/todd/todd .
ln -s ${HOME}/.cache/clip
```

# Data Preparation

`data` directory structure

```
DenseCLIP/data
└── coco
    ├── annotations
    │   ├── instances_train2017.json
    │   └── instances_val2017.json
    ├── train2017
    │   └── *.jpg
    └── val2017
        └── *.jpg
```

Generate GSL annotations

```shell
python tools/build_coco_zsl_dataset.py configs/_base_/datasets/coco_detection_clip.py --split 48_17
'''Outputs:
split: 48_17

Splitting training dataset
#annotations: 665387
#images: 107761
Saving to data/coco/annotations/instances_train2017_48_17.json

Splitting validation dataset
#annotations: 33152
#images: 4836
Saving to data/coco/annotations/instances_val2017_48_17.json
'''
python tools/build_coco_zsl_dataset.py configs/_base_/datasets/coco_detection_clip.py --split 65_15
'''Outputs:
split: 65_15

Splitting training dataset
#annotations: 811777
#images: 111338
Saving to data/coco/annotations/instances_train2017_65_15.json

Splitting validation dataset
#annotations: 36781
#images: 4952
Saving to data/coco/annotations/instances_val2017_65_15.json
'''
```

```
DenseCLIP/data
└── coco
    ├── annotations
    │   ├── instances_train2017.json
    │   ├── instances_train2017_48_17.json
    │   ├── instances_train2017_65_15.json
    │   ├── instances_val2017.json
    │   ├── instances_val2017_48_17.json
    │   └── instances_val2017_65_15.json
    ├── train2017
    │   └── *.jpg
    └── val2017
        └── *.jpg
```

# Proposals

```shell
sh tools/odps_test.sh configs/rpn/rpn_r101_fpn_1x_coco_train.py data/ckpts/rpn_r101_fpn_2x_coco_20200131-24e3db1a.pth 8 --out data/proposals/rpn_r101_fpn_coco_train.pkl
sh tools/odps_test.sh configs/rpn/rpn_r101_fpn_1x_coco_val.py data/ckpts/rpn_r101_fpn_2x_coco_20200131-24e3db1a.pth 8 --out data/proposals/rpn_r101_fpn_coco_test.pkl
```

# Extract features

For proposals

```shell
sh tools/odps_train.sh debug configs/feature_extractor/clip_proposal_feature_extractor.py 8 --seed 3407 --cfg-options log_config.interval=1
```

# Resources

Under `oss://mvap-data/zhax/wangluting/`

- mmdetection/CLIP pretrained model checkpoints
- coco dataset (with ZSL annotations)
- mmcv wheels
