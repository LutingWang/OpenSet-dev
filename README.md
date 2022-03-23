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

# Resources

Under `oss://mvap-data/zhax/wangluting/`

- mmdetection/CLIP pretrained model checkpoints
- coco dataset (with ZSL annotations)
- mmcv wheels
