_base_ = [
    'detpro_faster_rcnn_r50_fpn_1x_lvis_v1.py',
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadImageEmbeddingFromFile', data_root='data/lvis_v1/image_embeddings1'),
    # dict(type='LoadPthEmbeddings', data_root='data/lvis_v1/proposal_embeddings4/'),
    # dict(type='LoadZipEmbeddings', data_root='data/lvis_v1/proposal_embeddings.zip/data/lvis_clip_image_embedding/', task_name='train2017'),
    dict(type='LoadPthEmbeddings', data_root='data/lvis_v1/proposal_embeddings10/', min_bbox_area=32*32, detpro=True),
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
    dict(type='Collect', keys=['img', 'image_embeddings', 'gt_bboxes', 'gt_labels', 'bboxes', 'bbox_embeddings']),
]
data = dict(train=dict(dataset=dict(pipeline=train_pipeline)))
model = dict(
    type='GLIPFasterRCNN',
    class_embeddings='data/lvis_v1/prompt/detpro_ViT-B-32.pt', 
    backbone=dict(
        type='GLIPResNet',
        custom_plugins=dict(
            in_channels=[256, 512, 1024, 2048],
            embedding_dim=512,
            hidden_dim=512,
        ),
    ),
    glip_neck=dict(
        in_channels=256,
        num_levels=5,
        refine_level=2,
        refine=dict(
            type='StandardFusionDyHead',
            num_layers=3, 
            mil_classifier=dict(
                type='DyHeadClassifier',
                kappa=35, 
                logits_weight=False,
            ),
            loss_mil=dict(
                type='BCEWithLogitsLoss',
                weight=1,
            ),
            loss_image_kd=dict(
                type='L1Loss',
                weight=128,
            ),
        ),
    ),
    loss_ds=dict(
        type='DSLoss',
        weight=128,
        pred_features=256,
        target_features=512,
    )
)
