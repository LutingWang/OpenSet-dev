_base_ = [
    'coco_detection_clip.py',
]

data_root = 'data/coco/'
data = dict(
    train=dict(
        type='CocoZSLSeenDataset',
        ann_file=data_root + 'annotations/instances_train2017_48_17_.json',
    ),
    val=dict(
        type='CocoGZSLDataset',
        ann_file=data_root + 'annotations/instances_val2017_48_17_.json',
    ),
    test=dict(
        type='CocoGZSLDataset',
        ann_file=data_root + 'annotations/instances_val2017_48_17_.json',
    ),
)