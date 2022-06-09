_base_ = [
    '../vild/vild_ens_mask.py',
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadImageEmbeddingFromFile',
        data_root='data/lvis_v1/image_embeddings1'),
    dict(
        type='LoadPthEmbeddings',
        data_root='data/lvis_v1/proposal_embeddings10/',
        min_bbox_area=1024,
        detpro=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 800)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='ToTensor', keys=['bboxes', 'bbox_embeddings']),
    dict(
        type='ToDataContainer',
        fields=[dict(key='bboxes'),
                dict(key='bbox_embeddings')]),
    dict(
        type='Collect',
        keys=[
            'img', 'image_embeddings', 'gt_bboxes', 'gt_labels', 'gt_masks',
            'bboxes', 'bbox_embeddings'
        ]),
]
data = dict(
    train=dict(
        dataset=dict(
            pipeline=train_pipeline,
        ),
    ),
)
model = dict(
    type='CAFE',
    neck=dict(
        _delete_=True,
        type='CAFENeck',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        mil_classifier=dict(
            type='DyHeadClassifier',
            kappa=35,
            tau=0.07,
            loss_mil=dict(type='FocalWithLogitsLoss', weight=32),
            loss_image_kd=dict(type='L1Loss', weight=256)),
        pre=dict(
            hidden_dim=512,
        ),
        post=dict(
            refine_level=2,
            refine_layers=3,
            post_loss=dict(
                type='CrossEntropyLoss',
                weight=dict(
                    type='WarmupScheduler',
                    iter_=200,
                ),
            ),
        ),
    ),
)
