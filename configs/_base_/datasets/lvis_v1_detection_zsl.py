_base_ = [
    'lvis_v1_detection.py',
]

ann_file_root = 'data/lvis_v1/'
data = dict(
    train=dict(dataset=dict(
        type='LVISV1ZSLSeenDataset',
        ann_file=ann_file_root + 'annotations/lvis_v1_train_seen_1.json',
    )),
    val=dict(type='LVISV1GZSLDataset'),
    test=dict(type='LVISV1GZSLDataset'),
)