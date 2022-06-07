_base_ = [
    'rpn_r101_fpn_1x_coco_val.py',
]

data_root = 'data/coco/'
data = dict(
    test=dict(
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/'))
