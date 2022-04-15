_base_ = [
    'coco_detection_clip.py',
]

data_root = 'data/coco/'
data = dict(
    # samples_per_gpu=2,
    # workers_per_gpu=6,
    train=dict(
        type='CocoZSLSeenDataset',
        ann_file=data_root + 'annotations/instances_train2017_48_17_1.json',
        lmdb_file='local_data/embeddings6.lmdb',
    ),
    val=dict(
        type='CocoGZSLDataset',
        ann_file=data_root + 'annotations/instances_val2017_48_17_1.json',
    ),
    test=dict(
        type='CocoGZSLDataset',
        ann_file=data_root + 'annotations/instances_val2017_48_17_1.json',
    ),
)