_base_ = [
    'detpro_faster_rcnn_r50_fpn_1x_lvis_v1.py',
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='LoadPthEmbeddings', data_root='data/lvis_v1/proposal_embeddings4/'),
    # dict(type='LoadZipEmbeddings', data_root='data/lvis_v1/proposal_embeddings.zip/data/lvis_clip_image_embedding/', task_name='train2017'),
    dict(
        type='LoadPthEmbeddings', 
        data_root='data/lvis_v1/proposal_embeddings10/', 
        min_bbox_area=32*32, 
        detpro=True, 
        sampling_ratio=0.5,
    ),
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
data = dict(train=dict(dataset=dict(pipeline=train_pipeline)))
model = dict(
    type='TwoStageDetector',
    freeze_neck=True,
)
optimizer = dict(lr=0.02)
lr_config = dict(
    _delete_=True,
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[6])
runner = dict(max_epochs=8)