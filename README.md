# Setup

```shell
cd
git clone git@github.com:openai/CLIP.git
git clone git@github.com:lvis-dataset/lvis-api.git
git clone git@github.com:open-mmlab/mmdetection.git
GIT_LFS_SKIP_SMUDGE=1 git clone git@github.com:LutingWang/todd.git

git clone git@gitlab.alibaba-inc.com:wangluting.wlt/openset_detection.git
cd openset_detection
ln -s ${HOME}/CLIP/clip .
ln -s ${HOME}/lvis-api/lvis .
ln -s ${HOME}/mmdetection/mmdet .
ln -s ${HOME}/todd/todd .
ln -s ${HOME}/.cache/clip pretrained/clip
ln -s ${HOME}/.cache/torch/hub/checkpoints/ pretrained/torchvision
```

# Data Preparation

## Generate GZSL annotations

```shell
python tools/build_zsl_dataset.py configs/_base_/datasets/lvis_v1_detection.py --split 866_337
```

## Generate Proposals

```shell
# for coco
sh tools/odps_test.sh configs/rpn/rpn_r101_fpn_1x_coco_train.py data/ckpts/rpn_r101_fpn_2x_coco_20200131-24e3db1a.pth 8 --out data/coco/proposals/rpn_r101_fpn_coco_train_48_17.pkl
sh tools/odps_test.sh configs/rpn/rpn_r101_fpn_1x_coco_val.py data/ckpts/rpn_r101_fpn_2x_coco_20200131-24e3db1a.pth 8 --out data/coco/proposals/rpn_r101_fpn_coco_val_48_17.pkl

# for lvis v1
...
```

## Generate Class Embeddings

```shell
# for coco
python tools/class_embeddings.py vild

# for lvis
python tools/class_embeddings.py vild --dataset lvis_v1
python tools/class_embeddings.py vild --dataset lvis_v1 --pretrained "ViT-B/32"
```

## Generate Proposal Embeddings

```shell
sh tools/odps_train.sh debug configs/feature_extractor/proposal_feature_extractor.py 8 --seed 3407 --cfg-options log_config.interval=4
```

## Directory Tree

Before

```
DenseCLIP/data
├── coco
│   ├── annotations
│   │   ├── instances_train2017.json
│   │   └── instances_val2017.json
│   ├── train2017
│   │   └── *.jpg
│   └── val2017
│       └── *.jpg
└── lvis_v1
    ├── annotations
    │   ├── lvis_v1_train.json
    │   └── lvis_v1_val.json
    ├── train2017 -> coco/train2017
    └── val2017 -> coco/val2017
```

After

```
DenseCLIP/data
├── coco
│   ├── annotations
│   │   ├── instances_train2017.json (~118k)
│   │   ├── instances_train2017_48_17_1.json (~118k - 10526 = 107761)
│   │   ├── instances_val2017.json (5k)
│   │   └── instances_val2017_48_17_1.json (5k - 164 = 4836)
│   ├── prompt
│   │   └── vild.pth
│   ├── proposals
│   │   ├── rpn_r101_fpn_coco_train.pkl (~118k)
│   │   └── rpn_r101_fpn_coco_val.pkl (5k)
│   ├── train2017
│   │   └── *.jpg
│   └── val2017
│       └── *.jpg
└── lvis_v1
    ├── annotations
    │   ├── lvis_v1_train.json
    │   └── lvis_v1_val.json
    ├── proposals
    │   ├── rpn_r101_fpn_lvis_train.pkl (~118k)
    │   └── rpn_r101_fpn_lvis_val.pkl (5k)
    ├── train2017 -> coco/train2017
    └── val2017 -> coco/val2017
```

# ViLD*

```shell
sh tools/odps_train.sh debug configs/mask_rcnn/detpro_lvis_mask_rcnn_r50_fpn_20e_lvis_v1.py 8 --seed 3407
```

# Resources

Under `oss://mvap-data/zhax/wangluting/`

- mmdetection/CLIP pretrained model checkpoints
- coco dataset (with GZSL annotations)
- mmcv wheels
