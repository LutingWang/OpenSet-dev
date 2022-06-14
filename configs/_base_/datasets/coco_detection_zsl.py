_base_ = [
    'coco_detection.py',
]

data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadPthEmbeddings', data_root='data/coco/proposal_embeddings8/'),
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
    train=dict(
        type='CocoZSLSeenDataset',
        pipeline=train_pipeline,
        ann_file=data_root + 'annotations/instances_train2017_48_17_2.json',
        proposal_file=data_root + 'proposals/rpn_r101_fpn_coco_train.pkl',
    ),
    val=dict(
        type='CocoGZSLDataset',
        ann_file=data_root + 'annotations/instances_val2017_48_17_2.json',
    ),
    test=dict(
        type='CocoGZSLDataset',
        ann_file=data_root + 'annotations/instances_val2017_48_17_2.json',
    ),
)