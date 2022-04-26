_base_ = [
    'lvis_v1_detection.py',
]

ann_file_root = 'data/lvis_v1/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadEmbeddings', data_root='data/lvis_v1/proposal_embeddings4/'),
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
    dict(type='ToTensor', keys=['bboxes', 'bbox_embeddings']),
    dict(
        type='ToDataContainer', 
        fields=[dict(key='bboxes'), dict(key='bbox_embeddings')]),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'bboxes', 'bbox_embeddings']),
]
data = dict(
    train=dict(dataset=dict(
        type='LVISV1ZSLSeenDataset',
        ann_file=ann_file_root + 'annotations/lvis_v1_train_866_337_4.json',
        pipeline=train_pipeline,
    )),
    val=dict(type='LVISV1GZSLDataset'),
    test=dict(type='LVISV1GZSLDataset'))
