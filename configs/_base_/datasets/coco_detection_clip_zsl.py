_base_ = [
    'coco_detection_clip.py',
]

dataset_type = 'CocoZSLDataset'
data_root = 'data/coco/'
data = dict(
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017_48_17.json',
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017_48_17.json',
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017_48_17.json',
    ),
)