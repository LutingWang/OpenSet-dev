_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection_zsl.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py',
]

data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'proposals', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'proposals']),
        ])
]
data = dict(
    train=dict(
        proposal_file=data_root + 'proposals/rpn_r101_fpn_coco_train.pkl',
        pipeline=train_pipeline,
    ),
    val=dict(
        proposal_file=data_root + 'proposals/rpn_r101_fpn_coco_val.pkl',
        pipeline=test_pipeline,
    ),
    test=dict(
        proposal_file=data_root + 'proposals/rpn_r101_fpn_coco_val.pkl',
        pipeline=test_pipeline,
    ),
)
model = dict(
    rpn_head=None,
    roi_head=dict(
        bbox_head=dict(
            type='Shared2FCBBoxHeadZSL',
            class_embeddings='data/prompt/CocoDataset.pth',
            num_classes=48,
        )
    )
)
