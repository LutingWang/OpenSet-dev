_base_ = [
    'coco_detection.py',
]

dataset_type = 'LVISV1Dataset'
ann_file_root = 'data/lvis_v1/'
img_prefix = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 800)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
data = dict(
    train=dict(
        _delete_=True,
        type='ClassBalancedDataset',
        oversample_thr=1e-3,
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file_root + 'annotations/lvis_v1_train.json',
            img_prefix=img_prefix,
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_root + 'annotations/lvis_v1_val.json',
        img_prefix=img_prefix),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_root+ 'annotations/lvis_v1_val.json',
        img_prefix=img_prefix))
